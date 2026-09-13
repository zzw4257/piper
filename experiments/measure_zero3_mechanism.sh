#!/usr/bin/env bash
# F42 mechanism test. (a) per-node memory trace for shipped ZeRO-3 and
# prefetch_distance=1 at 4 stages: do full-param buffers accumulate during the
# first layer's dispatch in both? (b) prefetch_distance=1 with the host blocked
# on the budget predecessor before alloc: slope predicted ~2.0 if allocation-at-
# dispatch is the cause, ~4 if not.
set -u
R=/raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH
cd $R
OUT=logs/zero3_mech.txt; MAX_WAIT=$((24*3600)); t0=$(date +%s)
echo "start $(date -Is)" >> $OUT
pick_two() { local ok=""; for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$3<100000 {print $1}' | head -2 | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 2 ] && return 1; sleep 30; done; echo "$ok"; }
while :; do CARDS=$(pick_two) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up" >> $OUT; exit 1; }; sleep 120; done
echo "cards=$CARDS $(date -Is)" >> $OUT
mk() { local n=$1 pf=${2:-}
  local f=examples/base-schedules/_zero_probe_s${n}_true${pf:+_pf$pf}.json
  { echo "["; for ((i=0;i<n;i++)); do
      echo "  {\"op\":\"place\",\"filter\":{\"PP\":$i},\"devices\":[0,1],\"stream\":\"default_stream\"},"; done
    echo "  {\"op\":\"replicate\",\"filter\":{\"PP\":\"*\"},\"devices\":[0,1],\"reduce_stream\":\"dp_stream\",\"shard_grads\":true,\"shard_params\":true${pf:+,\"prefetch_distance\":$pf}},"
    echo "  {\"op\":\"split\",\"filter\":{},\"dim_name\":\"MB\",\"num_microbatches\":1}"; echo "]"; } > $f
  echo $f; }
DIM=4096; HID=16384; PSB=$(( (2*DIM*HID + 2*DIM*DIM)*4 ))
run() { # run <label> <stages> <pf-or-empty> <extra env...>
  local label=$1 n=$2 pf=$3; shift 3
  local S=$(mk $n $pf) LOG=logs/zero3_mech_${label}_s${n}.log
  echo "== $label stages=$n $(date -Is)" >> $OUT
  env "$@" CUDA_VISIBLE_DEVICES=$CARDS timeout 1200 \
    python examples/test_harness.py --test-file examples/test_tp_mlp.py --base-schedule $S --schedule custom \
      --stages $n --tp 1 --init random --dim $DIM --hidden $HID --warmup 1 --iters 2 \
      --temp-dir /var/tmp/ziweizho-ray > $LOG 2>&1
  local RC=$? RUN=$(grep -oE "out/[0-9]{8}_[0-9]{6}" $LOG | tail -1)
  echo "rc=$RC run=$RUN per_stage_bytes=$PSB" >> $OUT
  [ -n "$RUN" ] && python - "$RUN" <<'PY' >> $OUT 2>&1
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json")):
    m = json.load(open(f)); print(f"  dp_rank={m.get('dp_rank')} peak_memory_by_rank={m.get('peak_memory_by_rank')}")
PY
}
# (a) traces
run zero3_trace 4 ""  PIPER_MEM_TRACE=$R/logs/memtrace/zero3_s4
run pf1_trace   4 1   PIPER_MEM_TRACE=$R/logs/memtrace/pf1_s4
# (b) host-sync slope
for N in 2 4 8; do run pf1_hostsync $N 1 PIPER_AG_HOST_SYNC=1; done
echo "done $(date -Is)" >> $OUT
