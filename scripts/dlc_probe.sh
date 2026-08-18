#!/bin/bash
# DLC image probe for the MoR text pipeline: import table, data assert, 12-step
# 2-GPU training. Everything must be green before a multi-hour job is submitted.
set -uo pipefail
cd /cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1

echo "=== probe: $(python3 -V 2>&1) on $(hostname)"

# Vendored deps only if the image lacks them (same policy as fprm's dlc_v2.sh).
if ! python3 -c "import transformers, datasets, accelerate, hydra, omegaconf, wandb" 2>/dev/null; then
  PYV=$(python3 -c "import sys;print(f'{sys.version_info.major}{sys.version_info.minor}')")
  if [ "$PYV" = "312" ]; then
    export PYTHONPATH=/cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions/work/mordeps312
    echo "deps: injected mordeps312"
  else
    echo "deps: MISS - python $PYV has no vendored tree (only 312). Build work/mordeps$PYV."
  fi
else
  echo "deps: image provides everything"
fi

ok=1
for m in torch transformers datasets accelerate hydra omegaconf wandb tensorboard pyarrow numpy; do
  python3 -c "import $m;print(f'  OK   $m {getattr($m,\"__version__\",\"?\")}')" 2>/dev/null || { echo "  MISS $m"; ok=0; }
done
python3 -c "import torch;print(f'  torch cuda={torch.cuda.is_available()} n={torch.cuda.device_count()}')" || ok=0
[ $ok -eq 1 ] || { echo "PROBE FAILED: imports"; exit 1; }

DATA=/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/fineweb_edu/sample/10BT
ls "$DATA"/000_00000.parquet >/dev/null 2>&1 || { echo "DATASET MISSING: $DATA"; exit 1; }
echo "=== data ok: $(ls "$DATA" | wc -l) shards"

accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  pretrain.py --config-name smoke_360m_rec3_sdpa \
  name=dlc_probe total_batch_size=32 num_train_steps=12 stop_steps=12 num_warmup_steps=2 logging_steps=2
rc=$?
echo "=== probe train rc=$rc"
exit $rc
