#!/bin/bash
# Local 4-PPU run of fw10b_360m_rec3_sdpa with resume-on-crash supervision.
# Relaunch after a reboot with:
#   cd /cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions && \
#   setsid nohup bash scripts/run_local_fw10b.sh > logs/fw10b_360m_rec3.log 2>&1 < /dev/null & disown
set -u
cd /cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1

CONF=fw10b_360m_rec3_sdpa
OUT=results/pretrain/$CONF
fails=0
while [ $fails -lt 8 ]; do
    resume=(resume_from_checkpoint=false)
    if compgen -G "$OUT/checkpoint-*" > /dev/null; then
        resume=(resume_from_checkpoint=true)
    fi
    echo "=== attempt $((fails+1)) $(date) ${resume[*]} ==="
    accelerate launch --config_file acc_configs/default_config.yaml \
        pretrain.py --config-name $CONF "${resume[@]}"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== COMPLETE rc=0 $(date) ==="
        exit 0
    fi
    fails=$((fails+1))
    echo "=== exit rc=$rc (failure $fails/8), retry in 60s ==="
    sleep 60
done
echo "=== GAVE UP after 8 failures $(date) ==="
exit 1
