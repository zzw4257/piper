#!/usr/bin/env bash
# Pair Piper's iteration time with the same arithmetic run bare, across a size
# sweep, to test whether the runtime's share is a constant or scales.
set -u
R=$1; V=$2; TAG=$3
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/hostbound_${TAG}.txt; T=${4:-$HOME/ray_tmp}; mkdir -p $T
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '($4-$3)/1024>30 {print $1}' | head -4 | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 4 ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick) && break
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C" >> $OUT
PM=""
for SEQ in 1024 2048 4096 8192; do
  LOG=logs/hb_${TAG}_${SEQ}.log
  CUDA_VISIBLE_DEVICES=$C timeout 1800 python examples/test_harness.py \
    --test-file examples/test_ring_attn.py --base-schedule examples/base-schedules/cp4_ring_dp.json \
    --schedule custom --steps 4 --seq $SEQ --dim 512 --heads 8 --batch-size 8 \
    --warmup 2 --iters 10 --temp-dir $T > $LOG 2>&1
  RC=$?; RUN=$(grep -oE "out/[0-9]{8}_[0-9]{6}" $LOG | tail -1)
  MS=$(python - "$RUN" <<'PY' 2>/dev/null
import glob, json, sys
ts=[]
for f in sorted(glob.glob(f"{sys.argv[1]}/cp_metrics_dp*.json")):
    ts.append(min(json.load(open(f))["iter_times_s"]))
print(f"{min(ts)*1000:.2f}" if ts else "")
PY
)
  echo "  seq=$SEQ rc=$RC piper_ms=${MS:-NA}" >> $OUT
  PM="${PM}${PM:+,}${MS:-0}"
done
echo "piper_ms=$PM" >> $OUT
CUDA_VISIBLE_DEVICES=$(echo $C | cut -d, -f1) python experiments/probe_host_bound.py \
  --piper-ms "$PM" >> $OUT 2>&1
echo "done $(date -Is)" >> $OUT
