#!/bin/bash
# DLC worker script: one 360M fineweb run on 16 PPUs with resume-on-crash retries.
#   bash dlc_train360m.sh <config-name>     e.g. fw10b_360m_vanilla_sdpa
# Success is judged by the artifact (pretrain_results.json is written only after
# a completed train), NOT the exit code: PPU workers can segfault in NCCL
# teardown after fully successful training.
set -uo pipefail
CONF=${1:?usage: dlc_train360m.sh <config-name>}

cd /cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1

DATA=/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/fineweb_edu/sample/10BT
ls "$DATA"/000_00000.parquet >/dev/null 2>&1 || { echo "DATASET MISSING: $DATA"; exit 1; }

OUT=results/pretrain/$CONF
NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())")
echo "=== $CONF on $NGPU GPUs, $(hostname), $(date)"

fails=0
while [ $fails -lt 8 ]; do
    if [ -f "$OUT/pretrain_results.json" ]; then
        echo "=== COMPLETE (artifacts present) $(date)"; exit 0
    fi
    resume=(resume_from_checkpoint=false)
    compgen -G "$OUT/checkpoint-*" > /dev/null && resume=(resume_from_checkpoint=true)
    echo "=== attempt $((fails+1)) $(date) ${resume[*]}"
    accelerate launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
        pretrain.py --config-name "$CONF" "${resume[@]}"
    rc=$?
    if [ -f "$OUT/pretrain_results.json" ]; then
        echo "=== COMPLETE (rc=$rc ignored, artifacts present) $(date)"; exit 0
    fi
    fails=$((fails+1))
    echo "=== exit rc=$rc without artifacts (failure $fails/8), retry in 60s"
    sleep 60
done
echo "=== GAVE UP after 8 failures $(date)"
exit 1
