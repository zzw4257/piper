#!/usr/bin/env bash
# Wait for four genuinely idle GPUs, then run the TP x PP order comparison.
# Claims nothing, preempts nobody: only cards reporting near-zero memory and util.
set -uo pipefail
cd /raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH RAY_DEDUP_LOGS=0
OUT=/var/tmp/ziweizho-scripts/logs/tp_pp_order.txt
POLL=${POLL:-180}; MAX_WAIT=${MAX_WAIT:-43200}; NEED=4
waited=0
while : ; do
  free=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F', ' '{gsub(/ MiB/,"",$2); gsub(/ %/,"",$3); if ($2+0 < 30000 && $3+0 < 5) print $1}' | paste -sd,)
  n=$(echo "$free" | tr ',' '\n' | grep -c . || true)
  if [ "$n" -ge "$NEED" ]; then
    DEVS=$(echo "$free" | cut -d, -f1-4)
    echo "$(date -Is) got $DEVS after ${waited}s" | tee -a "$OUT"
    for rep in 1 2 3; do
      for sched in pp2_tp2_mb4_1f1b pp2_tp2_mb4_gpipe; do
        CUDA_VISIBLE_DEVICES=$DEVS timeout 1800 python examples/test_harness.py \
          --test-file examples/test_tp_mlp.py \
          --base-schedule examples/base-schedules/$sched.json --schedule custom \
          --dim 4096 --hidden 16384 --batch-size 2048 --stages 2 --tp 2 \
          --dtype bf16 --init random --warmup 5 --iters 12 \
          > /var/tmp/ziweizho-scripts/logs/o_${sched}_$rep.log 2>&1
        R=$(grep -o "out/[0-9_]*" /var/tmp/ziweizho-scripts/logs/o_${sched}_$rep.log | tail -1)
        if [ -z "$R" ]; then echo "  rep$rep $sched FAILED" | tee -a "$OUT"; continue; fi
        printf "  rep%d %-20s " "$rep" "$sched" | tee -a "$OUT"
        python3 -c "
import csv,sys,statistics
rows=list(csv.DictReader(open(sys.argv[1]+'/results.csv')))
t=[float(r['iter_time_mean_s'])*1e3 for r in rows]
print(f'iter {statistics.fmean(t):7.2f} ms  ranks={len(rows)}')
" "$R" | tee -a "$OUT"
      done
    done
    echo "$(date -Is) done" | tee -a "$OUT"
    exit 0
  fi
  waited=$((waited+POLL))
  [ "$waited" -ge "$MAX_WAIT" ] && { echo "$(date -Is) gave up after ${waited}s (saw $n idle)" | tee -a "$OUT"; exit 1; }
  sleep "$POLL"
done
