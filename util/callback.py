import os 
import sys
import json
import pickle
import warnings 
import subprocess
from collections import defaultdict 
from omegaconf import OmegaConf, open_dict

import wandb 
import torch
from torch.utils.tensorboard import SummaryWriter 
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# lm_eval pulls in vllm at import time, which conflicts with this node's
# transformers install; only EvalCallback needs it, so import lazily there.
try:
    import lm_eval
    from lm_eval import evaluator, utils
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table
except ImportError as _lm_eval_err:
    lm_eval = None
    _LM_EVAL_IMPORT_ERROR = _lm_eval_err
from transformers.trainer_callback import CallbackHandler


class ValLossCallback(TrainerCallback):
    """Held-out LM loss during training: mean per-sequence CE over the first
    `samples` packed sequences of `dataset` (same recipe and numbers as
    evaluate_fineweb_test.py), on rank 0 at step 0 and every `eval_steps`.
    Config block (nothing happens unless it is present and enabled):
        val_loss: {enable: true, eval_steps: 100, samples: 500, dataset: fineweb_test}
    Runs the unwrapped model under no_grad (fake-quant modules included), the
    other ranks wait at their next all-reduce (500 x 2048 tokens ~ 1 min on
    one PPU). Prints `=== val_loss step=N ...` and writes tensorboard val/loss."""
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__()
        v = cfg.val_loss
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.eval_steps = int(v.get("eval_steps", 100))
        self.samples = int(v.get("samples", 500))
        self.dataset = str(v.get("dataset", "fineweb_test"))
        self.at_begin = bool(v.get("at_begin", True))
        self._batch = None
        self.writer = SummaryWriter(cfg.tensorboard_dir) if cfg.get("tensorboard") else None

    @staticmethod
    def _is_rank0():
        return int(os.environ.get("RANK") or 0) == 0

    def _load(self):
        # identical construction to evaluate_fineweb_test.load_dataset_from_config for one LM dataset
        from datasets import load_dataset
        from lm_dataset.load_dataset import LM_DATASETS
        from lm_dataset.language_modeling_dataset import LanguageModelingDataset
        from lm_dataset.data_preprocessing import AddLabels, RemoveIndex
        ds = LanguageModelingDataset(load_dataset(**LM_DATASETS[self.dataset], streaming=True), self.tokenizer,
                                     max_length=self.cfg.max_length, transforms=[AddLabels(), RemoveIndex()],
                                     global_shuffling=False, local_shuffling=False,
                                     add_bos_token=self.cfg.get("add_bos_token", False))
        out = []
        for i, sample in enumerate(ds):
            if i >= self.samples:
                break
            out.append({k: sample[k].clone() for k in ("input_ids", "attention_mask", "labels")})
        self._batch = out

    @torch.no_grad()
    def _run(self, model, step):
        if not self._is_rank0() or model is None:
            return
        import time
        if self._batch is None:
            self._load()
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        t0, total = time.time(), 0.0
        for s in self._batch:
            out = model(input_ids=s["input_ids"].unsqueeze(0).to(device),
                        attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                        labels=s["labels"].unsqueeze(0).to(device),
                        use_cache=False, return_dict=True)
            total += out.loss.item()
        if was_training:
            model.train()
        avg = total / max(1, len(self._batch))
        print(f"=== val_loss step={step} {self.dataset} n={len(self._batch)} loss={avg:.4f} ({time.time() - t0:.0f}s)")
        if self.writer is not None:
            self.writer.add_scalar("val/loss", avg, step)
            self.writer.flush()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.at_begin and state.global_step == 0:
            self._run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            self._run(model, state.global_step)


class FixedStoppingCallback(TrainerCallback):
    """
    This callback is used when you want to set a certain num_train_steps for the learning rate scheduler 
    (e.g. "get linear schedule with warmup) but you want to stop training before that number of steps is reached.
    """
    def __init__(self, stop_steps: int):
        super().__init__()
        self.stop_steps = stop_steps
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.stop_steps:
            print(f"Stopping training at stop_steps={self.stop_steps}")
            control.should_training_stop = True
            
            
class EvalCallback(TrainerCallback):
    """
    This callback is used to evaluate the model on the zero-shot task. This is done by running lm_eval at the end of each epoch.
    The results are logged to wandb and tensorboard.
    """
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__()
        if lm_eval is None:
            raise ImportError("EvalCallback requires lm_eval, which failed to import") from _LM_EVAL_IMPORT_ERROR
        self.cfg = cfg
        self.tokenizer = tokenizer
        if self.cfg.get("tensorboard"):
            self.writer = SummaryWriter(self.cfg.tensorboard_dir)
    
    def get_tasks(self, cfg):
        task_manager = TaskManager("INFO")
        tasks = cfg.tasks
                                   
        if tasks is None:
            print("Need to specify task to evaluate.")
            sys.exit()
        elif tasks == "list":
            print(task_manager.list_all_tasks())
            sys.exit()
        elif tasks == "list_groups":
            print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
            sys.exit()
        elif tasks == "list_tags":
            print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
            sys.exit()
        elif tasks == "list_subtasks":
            print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
            sys.exit()
        else:
            if os.path.isdir(tasks):
                import glob

                task_names = []
                yaml_path = os.path.join(tasks, "*.yaml")
                for yaml_file in glob.glob(yaml_path):
                    config = utils.load_yaml_config(yaml_file)
                    task_names.append(config)
            else:
                task_list = tasks.split(",")
                task_names = task_manager.match_tasks(task_list)
                for task in [task for task in task_list if task not in task_names]:
                    if os.path.isfile(task):
                        config = utils.load_yaml_config(task)
                        task_names.append(config)
                task_missing = [
                    task for task in task_list if task not in task_names and "*" not in task
                ]  # we don't want errors if a wildcard ("*") task name was used

                if task_missing:
                    missing = ", ".join(task_missing)
                    print(
                        f"Tasks were not found: {missing}\n"
                        f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                    )
                    raise ValueError(
                        f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                    )
            return task_names

    def get_model(self, args, **kwargs):
        model = kwargs["model"]
            
        eval_cfg = self.cfg.get("evaluation")
        if eval_cfg.get("device") is None:
            OmegaConf.set_struct(eval_cfg, True)
            with open_dict(eval_cfg):
                eval_cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            warnings.warn(f"If you want to specify the device, use eval_cfg.device")
        if eval_cfg.get("batch_size") is None:
            OmegaConf.set_struct(eval_cfg, True)
            with open_dict(eval_cfg):
                eval_cfg.batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size
            warnings.warn(f"If you want to specify the batch_size, use eval_cfg.batch_size")

        model = lm_eval.api.registry.get_model("hf")(
            pretrained=model, tokenizer=self.tokenizer, 
            device=eval_cfg.device, batch_size=eval_cfg.batch_size,
            add_bos_token=self.cfg.get("add_bos_token", False),
            max_length=self.cfg.get("max_length", None),
        )
        return model, eval_cfg

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and state.global_step % self.cfg.evaluation.eval_steps == 0:
            model, eval_cfg = self.get_model(args, **kwargs)
            
            task_names = self.get_tasks(eval_cfg)
            
            results = evaluator.simple_evaluate(
                model=model, 
                tasks=task_names, 
                batch_size=eval_cfg.batch_size, 
                device=eval_cfg.device,
                num_fewshot=eval_cfg.num_fewshot,
            )
            
            if results is not None:
                print(make_table(results))
                if "groups" in results:
                    print(make_table(results, "groups"))
                                    
                if self.cfg.get("wandb"):
                    wandb_log = defaultdict(float)
                    for task_name, metrics in results["results"].items():
                        for metric_name, value in metrics.items():
                            if "std" not in metric_name:
                                wandb_log[f"eval_fewshot/{task_name}_{metric_name.replace('.', '_')}_multiple_ckpt"] = value
                    wandb.log(wandb_log, step=state.global_step)
                    
                if self.cfg.get("tensorboard"):
                    for task_name, metrics in results["results"].items():
                        for metric_name, value in metrics.items():
                            if "std" not in metric_name:
                                tag = f"eval_fewshot/{task_name}_{metric_name.replace('.', '_')}_multiple_ckpt"
                                self.writer.add_scalar(tag, value, global_step=state.global_step)


class PeftSaveCallback(TrainerCallback):
    """
    This callback is used to save the base model of peft model.
    """
    def __init__(self, save_steps: int, fixed_save_steps: str = None):
        super().__init__()
        self.save_steps = save_steps
        self.fixed_save_steps = []
        if fixed_save_steps is not None:
            self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.save_steps == 0 or state.global_step in self.fixed_save_steps) and state.global_step != 0:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            kwargs["model"].base_model.model.save_pretrained(output_dir, safe_serialization=False)
            
            
class DatasetSaveCallback(TrainerCallback):
    def __init__(self, save_steps: int, fixed_save_steps: str = None):
        super().__init__()
        self.save_steps = save_steps
        self.fixed_save_steps = []
        if fixed_save_steps is not None:
            self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.save_steps == 0 or state.global_step in self.fixed_save_steps) and state.global_step != 0:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            dataset = kwargs["train_dataloader"].dataset
            state_dict = dataset.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "dataset.pt"))


class MorSaveCallback(TrainerCallback):
    def __init__(self, save_steps: int, fixed_save_steps: str = None):
        super().__init__()
        self.save_steps = save_steps
        self.fixed_save_steps = []
        if fixed_save_steps is not None:
            self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]  # list
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.save_steps == 0 or state.global_step in self.fixed_save_steps) and state.global_step != 0:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            training_step = None
            for layer_idx in range(len(kwargs['model'].model.layers)):
                if hasattr(kwargs['model'].model.layers[layer_idx], "training_step"):
                    training_step = kwargs['model'].model.layers[layer_idx].training_step
                    break
            if training_step is not None:
                with open(os.path.join(output_dir, "training_step.pickle"), 'wb') as f:
                    pickle.dump(training_step, f)
            
            if "sam_optimizer" in kwargs and kwargs["sam_optimizer"] is not None:
                torch.save(kwargs["sam_optimizer"].state_dict(), os.path.join(output_dir, "sam_optimizer.pt"))
            if "sam_lr_scheduler" in kwargs and kwargs["sam_lr_scheduler"] is not None:
                torch.save(kwargs["sam_lr_scheduler"].state_dict(), os.path.join(output_dir, "sam_lr_scheduler.pt"))
                    
                
class MoRCallbackHandler(CallbackHandler):
    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler, sam_optimizer=None, sam_lr_scheduler=None,):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None
        
        self.sam_optimizer = sam_optimizer
        self.sam_lr_scheduler = sam_lr_scheduler
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False
        kwargs["sam_optimizer"] = self.sam_optimizer
        kwargs["sam_lr_scheduler"] = self.sam_lr_scheduler
        return self.call_event("on_save", args, state, control, **kwargs)


class ScalingLawsSaveCallback(TrainerCallback):
    """
    This callback is used to save the model during scaling laws experiments.
    """
    def __init__(self, fixed_save_steps: str):
        super().__init__()
        self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]  # list
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.fixed_save_steps and state.global_step != 0:
            control.should_save = True         
