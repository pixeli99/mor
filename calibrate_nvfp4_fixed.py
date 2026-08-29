"""Calibrate the frozen per-(boundary, block) NVFP4 scale (`nvfp4_state.scale: fixed`)
and/or audit recursion-boundary amax drift on an existing checkpoint. Zero training.

Model build follows evaluate_fineweb_test.py exactly (same SHARING_STRATEGY /
residual_scale / loop_attn / install_nvfp4_state path); the nvfp4 block is
forced to {scale: fixed, calibrate: true} so the pass is un-quantized and the
module only records per-token block amax + token norm at every boundary.

  python calibrate_nvfp4_fixed.py --exp_name fw10b_360m_rec3_sdpa_nvfp4fixed \
      --n_seq 64 --audit_json logs/e0/audit_rec3.json
  python calibrate_nvfp4_fixed.py --exp_name ifm730_huginn_carry_sdpa \
      --ckpt_dir results/pretrain/ifm730_huginn_carry_sdpa/checkpoint-7000 \
      --max_length 2048 --no_save --audit_json logs/e0/audit_huginn_carry.json

Calibration data: first --n_seq packed sequences of --calib_dataset (default
fineweb_edu = shards 000-012, streaming, no shuffling -> all from shard 000).
Never fineweb_test (shard 013).
"""
import os
import json
import time
import argparse
from copy import deepcopy

from paths import SAVE_DIR, PROJECT_ROOT, HF_CACHE_DIR; os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
import torch
from omegaconf import OmegaConf

from model.util import load_model_from_config
from model.sharing_strategy import SHARING_STRATEGY
from model.nvfp4_state import save_fixed_scale, FIXED_PERCENTILES
from util.tokenizer import load_tokenizer_from_config
from util.config import preprocess_config
from util.misc import get_torch_dtype
from evaluate_fineweb_test import load_dataset_from_config


def build_model(exp_name, ckpt_dir=None):
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "conf/pretrain", f"{exp_name}.yaml"))
    cfg = preprocess_config(cfg)
    cfg.resume_from_checkpoint = False
    OmegaConf.set_struct(cfg, False)
    cfg.nvfp4_state = OmegaConf.create({"enable": True, "scale": "fixed", "calibrate": True})
    assert cfg.recursive.get("enable"), "nvfp4_state needs a recursive model"
    assert not (cfg.get("relaxation") and cfg.relaxation.get("enable")), "relaxation not supported here"
    assert not ("mor" in cfg and cfg.mor.get("enable")), "mor not supported here"

    model = load_model_from_config(cfg)
    model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
    if cfg.recursive.get("residual_scale"):
        model.install_residual_scale(cfg)
        print(f"residual_scale {cfg.recursive.residual_scale} on looped blocks")
    if "loop_attn" in cfg and cfg.loop_attn.get("enable"):
        model.install_loop_attn(cfg)
    model.install_nvfp4_state(cfg)
    print(f"nvfp4_state installed: scale={model.model.nvfp4_state.scale_rule} calibrate={model.model.nvfp4_state.calibrate} "
          f"start={model.model.nvfp4_start} ends={sorted(model.model.nvfp4_ends)}")
    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)

    ckpt_dir = ckpt_dir or os.path.join(SAVE_DIR, "pretrain", exp_name)
    pt, st = os.path.join(ckpt_dir, "pytorch_model.bin"), os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(pt):
        state_dict = torch.load(pt, map_location="cpu")
    elif os.path.exists(st):
        from safetensors.torch import load_file
        state_dict = load_file(st, device="cpu")
    else:
        raise FileNotFoundError(f"no pytorch_model.bin / model.safetensors in {ckpt_dir}")
    model.load_state_dict(state_dict)
    print(f"loaded checkpoint {ckpt_dir}")
    return cfg, model, ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--ckpt_dir", default=None, help="override results/pretrain/<exp> (e.g. a checkpoint-N dir)")
    ap.add_argument("--calib_dataset", default="fineweb_edu")
    ap.add_argument("--n_seq", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=None, help="override cfg.max_length")
    ap.add_argument("--out", default=None, help="fixed-scale .pt (default results/pretrain/<exp>/nvfp4_fixed_scale.pt)")
    ap.add_argument("--no_save", action="store_true", help="audit only, do not write the .pt")
    ap.add_argument("--audit_json", default=None)
    args = ap.parse_args()
    assert args.calib_dataset != "fineweb_test", "never calibrate on the held-out shard"

    cfg, model, ckpt_dir = build_model(args.exp_name, args.ckpt_dir)
    cfg_ds = deepcopy(cfg)
    cfg_ds.dataset = args.calib_dataset
    if args.max_length:
        cfg_ds.max_length = args.max_length
    tokenizer = load_tokenizer_from_config(cfg_ds)
    dataset = load_dataset_from_config(cfg_ds, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device, dtype=get_torch_dtype(cfg))
    model.eval()
    q = model.model.nvfp4_state

    t0 = time.time()
    losses = []
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= args.n_seq:
                break
            out = model(input_ids=sample["input_ids"].unsqueeze(0).to(device),
                        attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
                        labels=sample["labels"].unsqueeze(0).to(device),
                        use_cache=False, return_dict=True)
            losses.append(out.loss.item())
    n_tok = sum(x.shape[0] for x in q.calib_amax[0])
    print(f"{len(losses)} seqs, {n_tok} tokens, mean loss {sum(losses)/len(losses):.4f}, {time.time()-t0:.0f}s")

    tables = q.calibration_tables(FIXED_PERCENTILES)
    ends = sorted(model.model.nvfp4_ends)
    meta = dict(exp=args.exp_name, ckpt_dir=ckpt_dir, ends=ends, hidden=model.config.hidden_size,
                residual_scale=float(cfg.recursive.get("residual_scale") or 1.0),
                calib_dataset=args.calib_dataset, n_seq=len(losses), max_length=int(cfg_ds.max_length),
                n_tokens=n_tok, calib_loss=sum(losses) / len(losses), tokenizer=cfg.tokenizer)
    for s in tables["stats"]:
        s["layer"] = ends[s["k"]]
        s.update({f"clip_{k}": v[s["k"]] for k, v in tables["clip_frac"].items()})
        print(f"  k={s['k']} layer={s['layer']:3d} amax median/p99/p99.9/max = {s['amax_median']:8.2f} {s['amax_p99']:8.2f} "
              f"{s['amax_p999']:8.2f} {s['amax_max']:8.2f}  |h| mean {s['norm_mean']:8.2f}  "
              f"clip(p99.9/p99) {s['clip_p99.9']:.4f}/{s['clip_p99']:.4f}")

    if not args.no_save:
        out = args.out or os.path.join(SAVE_DIR, "pretrain", args.exp_name, "nvfp4_fixed_scale.pt")
        save_fixed_scale(out, tables, meta)
        print(f"fixed scale tables {sorted(tables['percentiles'])} shape {tuple(tables['percentiles']['max'].shape)} -> {out}")
    if args.audit_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.audit_json)), exist_ok=True)
        with open(args.audit_json, "w") as f:
            json.dump(dict(meta, stats=tables["stats"]), f, indent=2)
        print(f"audit -> {args.audit_json}")


if __name__ == "__main__":
    main()
