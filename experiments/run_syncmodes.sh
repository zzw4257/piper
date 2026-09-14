#!/usr/bin/env bash
# Correctness and timing for the three host-sync modes (log F60).
#   device : upstream, torch.cuda.synchronize() at the end of the update node
#   narrow : wait only on the loss events
#   defer  : no host wait at all; losses are converted at the top of the next step
set -u
R=$1; V=$2; T=$3; TAG=$4; REPS=${5:-4}
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/syncmodes_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local n=$1 mu=$2 ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' -v mu=$mu '$2<mu && ($4-$3)/1024>30 {print $1}' | head -$n | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C4=$(pick 4 101) && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
C2=$(echo $C4 | cut -d, -f1,2)
echo "cards4=$C4 cards2=$C2" >> $OUT
echo "load at start: $(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "%s:%s%% ", $1,$2}')" >> $OUT

echo "--- correctness, every mode, every check, with negative controls" >> $OUT
for M in device narrow defer; do
  echo "== mode=$M" >> $OUT
  PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C2 timeout 2400 python experiments/check_tp_equivalence.py \
    > logs/sm_tp_$M.log 2>&1; echo "  tp rc=$?" >> $OUT
  grep -E "^PASS|^control|assert|Error" logs/sm_tp_$M.log | tail -2 | sed 's/^/    /' >> $OUT
  PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C2 timeout 2400 python experiments/check_cp_equivalence.py --cp 2 \
    -- --temp-dir $T > logs/sm_cp2_$M.log 2>&1; echo "  cp2 rc=$?" >> $OUT
  grep -E "ring ==|negative|assert|Error" logs/sm_cp2_$M.log | tail -2 | sed 's/^/    /' >> $OUT
  PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C4 timeout 2400 python experiments/check_cp_equivalence.py --cp 4 \
    --cp2-schedule cp4_ring_dp --skip-negative-control -- --temp-dir $T --seq 8192 --dim 512 \
    --heads 8 --batch-size 2 --iters 5 > logs/sm_cp4_$M.log 2>&1; echo "  cp4 rc=$?" >> $OUT
  grep -E "ring ==|assert|Error" logs/sm_cp4_$M.log | tail -1 | sed 's/^/    /' >> $OUT
done

echo "--- timing, interleaved over modes, $REPS reps" >> $OUT
for REP in $(seq 1 $REPS); do
  for M in device narrow defer; do
    D=/tmp/sm_${M}_$REP; rm -rf $D; mkdir -p $D
    RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C4 PIPER_TIME_TRACE=$D timeout 1200 \
      python examples/test_harness.py --test-file examples/test_ring_attn.py \
      --base-schedule examples/base-schedules/cp4_ring_dp.json --schedule custom \
      --steps 4 --seq 1024 --dim 512 --heads 8 --batch-size 8 --warmup 2 --iters 12 \
      --temp-dir $T 2>&1 | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
    MS=$(python - "$RUN" <<'PY' 2>/dev/null
import glob, json, sys
ts=[min(json.load(open(f))["iter_times_s"]) for f in sorted(glob.glob(f"{sys.argv[1]}/cp_metrics_dp*.json"))]
print(f"{max(ts)*1000:.2f}" if ts else "NA")
PY
)
    SY=$(grep -h "__upd_final_sync__" $D/trace_rank0_step*.tsv 2>/dev/null | awk -F'\t' '{s+=$6; n++} END{if(n) printf "%.0f", s/n; else print "NA"}')
    echo "  rep$REP $M driver_min=${MS}ms mean_final_sync=${SY}us" >> $OUT
  done
done
echo "done $(date -Is)" >> $OUT
