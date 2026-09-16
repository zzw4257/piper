#!/usr/bin/env bash
# The order claim reversed sign on a quiet machine. Three things changed at
# once: the machine, the drained clock, and this project's own default sync
# mode. Rerun the same comparison under the upstream sync mode to rule the
# last one out.
set -u
R=$1; V=$2; T=$3; CARDS=$4; REPS=${5:-5}
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/isolate_order.txt
python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count()>0 else 1)' 2>/dev/null || { echo "CUDA down"; exit 1; }
echo "=== start $(date -Is) cards=$CARDS load=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits|paste -sd,) ===" >> $OUT
run() {
  local S=$1 M=$2
  local RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$CARDS timeout 1800 python examples/test_harness.py \
    --test-file examples/test_tp_mlp.py --base-schedule examples/base-schedules/$S.json \
    --schedule custom --init random --warmup 2 --iters 10 --stages 2 --tp 2 \
    --temp-dir $T 2>&1 | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  python - "$RUN" <<'PY' 2>/dev/null
import glob, json, sys
rows=[json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json"))]
if not rows: print("NA/NA"); raise SystemExit
d=max(r.get('total_timed_s',0)/max(r.get('timed_iters',1),1)*1000 for r in rows)
m=max(min(r["iter_times_s"])*1000 for r in rows)
print(f"{d:.2f}/{m:.2f}")
PY
}
for M in device narrow; do
  echo "-- sync=$M  (drained/per-iter-min)" >> $OUT
  for i in $(seq 1 $REPS); do
    a=$(run pp2_tp2_mb4_1f1b $M); b=$(run pp2_tp2_mb4_gpipe $M)
    echo "   rep$i 1f1b=$a  gpipe=$b" >> $OUT
  done
done
echo "done $(date -Is)" >> $OUT
