import os 
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from paths import SAVE_DIR, PROJECT_ROOT, HF_CACHE_DIR; os.environ["HF_HOME"] = HF_CACHE_DIR
import string
import warnings
from pathlib import Path

import random
import hydra
import torch
import torch.nn as nn 
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf 
from copy import deepcopy
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator

from lm_dataset.load_dataset import LM_DATASETS, load_dataset_from_config 
from model.util import load_model_from_config
from model.sharing_strategy import SHARING_STRATEGY
from model.relaxation.util import relax_weight_sharing
from util.config import preprocess_config
from util.tokenizer import load_tokenizer_from_config 
from util.trainer_pt import MoRTrainer
from util.callback import FixedStoppingCallback, EvalCallback, PeftSaveCallback, DatasetSaveCallback, ScalingLawsSaveCallback
from util.misc import print_trainable_parameters, get_latest_checkpoint_path, print_rank_zero, get_launcher_type; print_rank_zero()


class LoopAttnTrainer(Trainer):
    """Stock Trainer plus: (1) loop-attn params train in their own group at
    lr * loop_attn.lr_scale with no weight decay — at the backbone lr (3e-3)
    the content query w destabilizes: run dlcbveb62a797f13 exploded at step
    ~1250 (grad_norm 1024) after 1250 healthy steps; (2) loop-attn diagnostics
    in the logs — beta per head, newest_frac (1.0 = the read IS carry-last,
    i.e. a no-op), read_delta and logit_std."""

    def __init__(self, *args, cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop_attn_lr_scale = cfg.loop_attn.get("lr_scale", 0.1) if cfg is not None else 0.1

    def get_decay_parameter_names(self, model):
        return [n for n in super().get_decay_parameter_names(model) if "beta_logit" not in n]

    def create_optimizer(self):
        if self.optimizer is None:
            opt_model = self.model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            la_names = {n for n, _ in opt_model.named_parameters() if "loop_attn" in n}
            groups = [
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n not in la_names and p.requires_grad],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n not in la_names and p.requires_grad],
                 "weight_decay": 0.0},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in la_names and p.requires_grad],
                 "weight_decay": 0.0,
                 "lr": self.args.learning_rate * self.loop_attn_lr_scale},
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            print(f"loop_attn optimizer group: {len(groups[2]['params'])} params at lr {groups[2]['lr']:.2e}")
        return self.optimizer

    def log(self, logs, start_time=None):
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        la = getattr(getattr(m, "model", m), "loop_attn", None)
        if la is not None:
            for i, b in enumerate(la.beta_logit.sigmoid().detach().float().cpu().tolist()):
                logs[f"loop_attn_beta{i}"] = round(b, 4)
            for k, v in la.stats.items():
                logs[f"loop_attn_{k}"] = round(float(v), 4)
        super().log(logs, start_time)


@hydra.main(config_path="conf/pretrain", config_name="yymmdd_pretrain")
def main(cfg: DictConfig):
    cfg = preprocess_config(cfg)
    
    if cfg.wandb and cfg.get("wandb_run_id") is None:
        characters = string.ascii_letters + string.digits
        wandb_run_id = "".join(random.choices(characters, k=8))
        raise KeyError(f"wandb_run_id is not set. Please set wandb_run_id as {wandb_run_id} in the config file and run again.")
                
    # wandb settings
    if cfg.get("wandb"):
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity # name your W&B team
        os.environ["WANDB_PROJECT"] = cfg.wandb_project # name your W&B project
        if cfg.get("wandb_watch") is not None:
            os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
        os.environ ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = cfg.get("WANDB_MODE", "online")
        if os.environ["WANDB_MODE"] == "offline":
            os.environ["WANDB_DIR"] = PROJECT_ROOT
        os.environ["WANDB_SAVE_CODE"] = "false"
        os.environ["WANDB LOG MODEL"] = "false"
    
    launcher_type = get_launcher_type()
    
    print ("Loading tokenizers...")
    tokenizer = load_tokenizer_from_config(cfg)

    print ("Loading dataset...")
    train_dataset = load_dataset_from_config(cfg, tokenizer)
    if cfg.resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint_path(cfg, resume_step=cfg.resume_step if ("resume_step" in cfg and cfg.resume_step is not None) else None)
        train_dataset.load_state_dict(
            torch.load(os.path.join(str(latest_checkpoint), "dataset.pt"))
        )

    print ("Loading models...")
    model = load_model_from_config(cfg)
    
    if cfg.recursive.get("enable"):
        # KV cache sharing strategy
        model, lora_init_dict = SHARING_STRATEGY[cfg.model](cfg, model)
        if "loop_attn" in cfg and cfg.loop_attn.get("enable"):
            model.install_loop_attn(cfg)
            model.model.loop_attn.collect = True   # cheap diagnostics, logged by LoopAttnTrainer
            print(f"loop_attn installed: start={model.model.loop_attn_start} "
                  f"ends={sorted(model.model.loop_attn_ends)} heads={model.model.loop_attn.heads}")

    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)
        
    if cfg.get("relaxation") and cfg.relaxation.get("enable"):
        model = relax_weight_sharing(cfg, model, lora_init_dict=lora_init_dict)
        
        if cfg.resume_from_checkpoint:
            if cfg.relaxation.get("enable"):
                latest_checkpoint = get_latest_checkpoint_path(cfg, resume_step=cfg.resume_step if ("resume_step" in cfg and cfg.resume_step is not None) else None)                
                state_dict = torch.load(os.path.join(str(latest_checkpoint), "pytorch_model.bin"))
                model.get_base_model().load_state_dict(state_dict)
                
    if "mor" in cfg and cfg.mor.get("enable"):            
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
        else:
            raise ValueError(f"Unknown MoR type {cfg.mor.type}.")
        
    print_trainable_parameters(model)
        
    report_to = []
    if cfg.wandb:
        report_to.append("wandb")
    if cfg.tensorboard:
        report_to.append("tensorboard")
    
    train_args = TrainingArguments(
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine_with_min_lr"),
        lr_scheduler_kwargs=dict(cfg.get("lr_scheduler_kwargs", {"min_lr_rate": 0.1,})),
        learning_rate=cfg.learning_rate,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        output_dir=os.path.join(SAVE_DIR, "pretrain", cfg.output_dir),
        max_steps=cfg.num_train_steps,
        warmup_steps=cfg.num_warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_safetensors=False if launcher_type == "accelerate" else True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=cfg.dataloader_num_workers,
        bf16=cfg.precision == "bf16",
        fp16=cfg.precision == "fp16",
        overwrite_output_dir=True,
        report_to=report_to,
        run_name=cfg.wandb_run_name,
        logging_dir=cfg.tensorboard_dir,
        deepspeed=cfg.deepspeed if launcher_type == "deepspeed" else None,
        log_on_each_node=False,
    )
    
    callbacks = []
    fixed_save_steps = cfg.fixed_save_steps if ("fixed_save_steps" in cfg and cfg.fixed_save_steps) else None
    if cfg.stop_steps is not None:
        callbacks.append(FixedStoppingCallback(cfg.stop_steps))
    if "evaluation" in cfg and cfg.evaluation.enable:
        callbacks.append(EvalCallback(cfg, tokenizer))
    if cfg.relaxation.get("enable") and cfg.relaxation.method in ["lora", "dora", "adaption_prompt"]:
        callbacks.append(PeftSaveCallback(cfg.save_steps, fixed_save_steps=fixed_save_steps))
    if all(ds in LM_DATASETS for ds in cfg.dataset.split(',')):
        callbacks.append(DatasetSaveCallback(cfg.save_steps, fixed_save_steps=fixed_save_steps))
    if fixed_save_steps is not None:
        callbacks.append(ScalingLawsSaveCallback(fixed_save_steps,))
        
    if "mor" in cfg and cfg.mor.get("enable"):
        trainer = MoRTrainer(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks, cfg=cfg,)
    elif "loop_attn" in cfg and cfg.loop_attn.get("enable"):
        trainer = LoopAttnTrainer(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks, cfg=cfg,)
    else:
        trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks,)
    
    train_result = trainer.train(
        resume_from_checkpoint=cfg.resume_from_checkpoint
    )
    metrics = train_result.metrics
    trainer.log_metrics("pretrain", metrics)
    trainer.save_metrics("pretrain", metrics)
    trainer.save_state()
    trainer.save_model()
    
    if cfg.relaxation.get("enable"):
        trainer.model.base_model.model.save_pretrained(train_args.output_dir, safe_serialization=False)

    # Explicit teardown: without this, the atexit NCCL destructor segfaults on
    # PPU workers (rank exits -11 after training fully completed and saved).
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()