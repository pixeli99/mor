#!/bin/bash
# Poll two DLC fine-tune jobs every 5 min: job-level Status (top-level JSON key,
# not the pod one) + last loss / val_loss lines from master-0. On a terminal
# state write TERMINAL; if the artifacts exist run the fineweb_test 2000-sample
# eval on an idle local GPU (0% util, <500 MiB), else write NEED_EVAL.
#   bash scripts/qatft_monitor.sh <jobid_amax> <jobid_bf16>
cd /cpfs01/shared/public/users/pengxiang.li/mixture_of_recursions
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
LOG=logs/e0/qatft-monitor.log
DLCW=/cpfs01/shared/public/users/pengxiang.li/dlcw
JA=${1:?jobid amax}; JB=${2:?jobid bf16}
declare -A EXP=([$JA]=fw10b_360m_rec3_qatft_amax_sdpa [$JB]=fw10b_360m_rec3_ft_bf16_sdpa)
declare -A TAG=([$JA]=amax [$JB]=bf16)
declare -A DONE
status() { timeout 120 $DLCW get job "$1" 2>/dev/null | sed '1d' | python3 -c "import json,sys; print(json.load(sys.stdin).get('Status'))" 2>/dev/null; }
lastlines() { timeout 240 $DLCW logs "$1" "$1-master-0" -n 40000 2>/dev/null | grep -E "^\{'loss'|=== val_loss|=== COMPLETE|=== GAVE UP|=== attempt|Traceback|Error" | tail -3 | tr '\n' '|'; }
free_gpu() { nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<500 {print $1; exit}'; }
echo "$(date '+%F %T') monitor start pid $$ jobs amax=$JA bf16=$JB" >> $LOG
while :; do
  alldone=1
  for j in $JA $JB; do
    [ -n "${DONE[$j]:-}" ] && continue
    alldone=0
    st=$(status "$j"); ll=$(lastlines "$j"); exp=${EXP[$j]}; tag=${TAG[$j]}
    echo "$(date '+%F %T') $j $tag Status=$st | $ll" >> $LOG
    case "$st" in
      Succeeded|Failed|Stopped)
        if [ -f results/pretrain/$exp/pretrain_results.json ] && [ -f results/pretrain/$exp/pytorch_model.bin ]; then
          g=$(free_gpu)
          if [ -n "$g" ]; then
            echo "$(date '+%F %T') $j $tag TERMINAL $st artifacts present, eval on GPU $g" >> $LOG
            CUDA_VISIBLE_DEVICES=$g python evaluate_fineweb_test.py --exp_names $exp --sample_number 2000 --output_file logs/e0/qatft_$tag.json > logs/e0/qatft_$tag.log 2>&1
            echo "$(date '+%F %T') $j $tag EVAL_DONE $(tr -d '\n ' < logs/e0/qatft_$tag.json 2>/dev/null)" >> $LOG
          else
            echo "$(date '+%F %T') $j $tag TERMINAL $st artifacts present, NEED_EVAL (no idle GPU): CUDA_VISIBLE_DEVICES=<g> python evaluate_fineweb_test.py --exp_names $exp --sample_number 2000 --output_file logs/e0/qatft_$tag.json" >> $LOG
          fi
        else
          echo "$(date '+%F %T') $j $tag TERMINAL $st WITHOUT artifacts (no pretrain_results.json) -> check master-0 log" >> $LOG
        fi
        DONE[$j]=1;;
    esac
  done
  [ $alldone = 1 ] && { echo "$(date '+%F %T') ALL_TERMINAL" >> $LOG; exit 0; }
  sleep 300
done
