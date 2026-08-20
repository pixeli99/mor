#!/bin/bash
# DLC worker script: multi-node training via accelerate; the SAME command runs
# on every pod. DLC injects WORLD_SIZE (=pod count), RANK (=pod rank),
# MASTER_ADDR, MASTER_PORT; accelerate re-exports WORLD_SIZE as the global rank
# count for pretrain.py (util/config.py splits total_batch_size by it).
#   bash dlc_train_multinode.sh <config-name> [extra hydra overrides...]
# Success is judged by the artifact (pretrain_results.json), NOT the exit code:
# PPU workers can segfault in NCCL teardown after fully successful training.
# On a mid-run crash every pod's launcher dies (rendezvous breaks), each retry
# loop relaunches and blocks in rendezvous until all pods rejoin.
set -uo pipefail
CONF=${1:?usage: dlc_train_multinode.sh <config-name> [overrides...]}
shift

cd /cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1

ls /jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li >/dev/null 2>&1 || { echo "JFS DATA MOUNT MISSING"; exit 1; }

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
ADDR=${MASTER_ADDR:-127.0.0.1}
PORT=${MASTER_PORT:-29500}
NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())")
TOTAL=$((NGPU * NNODES))

OUT_NAME=$CONF
for ov in "$@"; do case $ov in output_dir=*) OUT_NAME=${ov#output_dir=};; esac; done
OUT=results/pretrain/$OUT_NAME
echo "=== $CONF node $NODE_RANK/$NNODES ($NGPU gpu/node, $TOTAL total) master=$ADDR:$PORT out=$OUT $(hostname) $(date)"

fails=0
while [ $fails -lt 8 ]; do
    if [ -f "$OUT/pretrain_results.json" ]; then
        echo "=== COMPLETE (artifacts present) $(date)"; exit 0
    fi
    resume=(resume_from_checkpoint=false)
    compgen -G "$OUT/checkpoint-*" > /dev/null && resume=(resume_from_checkpoint=true)
    echo "=== attempt $((fails+1)) $(date) ${resume[*]}"
    accelerate launch --num_processes "$TOTAL" --num_machines "$NNODES" \
        --machine_rank "$NODE_RANK" --main_process_ip "$ADDR" --main_process_port "$PORT" \
        --mixed_precision bf16 --dynamo_backend no \
        pretrain.py --config-name "$CONF" "${resume[@]}" "$@"
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
