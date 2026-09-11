#!/usr/bin/env bash
# Do the shipped examples still run now that they get real inputs?
# The zero-input fix changed every example's behaviour; nothing has run since.
set -uo pipefail
cd /raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH RAY_DEDUP_LOGS=0
OUT=/var/tmp/ziweizho-scripts/logs/upstream_examples.txt
POLL=${POLL:-180}; MAX_WAIT=${MAX_WAIT:-43200}; NEED=4
waited=0
while : ; do
  free=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F', ' '{gsub(/ MiB/,"",$2); gsub(/ %/,"",$3); if ($2+0 < 30000 && $3+0 < 5) print $1}' | paste -sd,)
  n=$(echo "$free" | tr ',' '\n' | grep -c . || true)
  if [ "$n" -ge "$NEED" ]; then
    DEVS=$(echo "$free" | cut -d, -f1-4)
    echo "$(date -Is) got $DEVS" | tee -a "$OUT"
    run() {  # run <label> <test-file> <base-schedule> [extra...]
      local label=$1 tf=$2 bs=$3; shift 3
      CUDA_VISIBLE_DEVICES=$DEVS timeout 2400 python examples/test_harness.py \
        --test-file "$tf" --base-schedule "examples/base-schedules/$bs.json" \
        --schedule 1f1b --ranks 2 --mbs 4 "$@" \
        > "/var/tmp/ziweizho-scripts/logs/ux_$label.log" 2>&1
      local rc=$?
      local R
      R=$(grep -o "out/[0-9_]*" "/var/tmp/ziweizho-scripts/logs/ux_$label.log" | tail -1)
      if [ $rc -ne 0 ] || [ -z "$R" ]; then
        echo "  $label FAILED rc=$rc" | tee -a "$OUT"
        grep -E "Error|error:|Traceback|assert" "/var/tmp/ziweizho-scripts/logs/ux_$label.log" \
          | head -4 | sed 's/^/      /' | tee -a "$OUT"
        return
      fi
      printf "  %-18s OK  " "$label" | tee -a "$OUT"
      python3 -c "
import csv,sys,statistics,glob,json
rows=list(csv.DictReader(open(sys.argv[1]+'/results.csv')))
t=[float(r['iter_time_mean_s'])*1e3 for r in rows]
print(f\"iter {statistics.fmean(t):7.2f} ms  ranks={len(rows)}\", end='')
" "$R" | tee -a "$OUT"
      # losses now exist because of the loss-plumbing fix; NaN would mean the
      # zero-input fix exposed something the zeros were hiding.
      grep -ho "losses=\[[^]]*\]" "/var/tmp/ziweizho-scripts/logs/ux_$label.log" | head -1 \
        | sed 's/^/  /' | tee -a "$OUT" || echo "" | tee -a "$OUT"
    }
    run qwen_moe_ep examples/test_qwen.py pp2_dp2_ep2
    run llama_pp_dp examples/test_llama.py llama_pp2_dp2
    echo "$(date -Is) done" | tee -a "$OUT"
    exit 0
  fi
  waited=$((waited+POLL))
  [ "$waited" -ge "$MAX_WAIT" ] && { echo "$(date -Is) gave up (saw $n idle)" | tee -a "$OUT"; exit 1; }
  sleep "$POLL"
done
