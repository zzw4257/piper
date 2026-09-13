#!/usr/bin/env bash
# F37 verification: peak memory vs depth under ZeRO-3 and plain DP.
# Slope prediction per stage-bytes B: ideal ZeRO-3 = 2.0B, F37 (gather all) = 3.0B, plain DP = 4.0B.
# Memory measurement does not need a quiet interconnect -- only two free cards.
set -u
R=/raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH
cd $R
OUT=logs/zero3_peak.txt
MAX_WAIT=$((24*3600)); t0=$(date +%s)
echo "start $(date -Is)" >> $OUT

pick_two() {  # two cards with <100GB used (peak memory is per-process; util is irrelevant), 3 samples 30s apart
  local ok=""; for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$3<100000 {print $1}' | head -2 | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 2 ] && return 1
    sleep 30
  done
  echo "$ok"
}
while :; do
  CARDS=$(pick_two) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up $(date -Is)" >> $OUT; exit 1; }
  sleep 120
done
echo "cards=$CARDS $(date -Is)" >> $OUT

mk() {  # mk <stages> <shard_params> <shard_grads> [prefetch_distance] -> schedule path
  local n=$1 sp=$2 sg=$3 pf=${4:-}
  # separate line: bash expands a whole `local` word list before assigning, so
  # ${n} on the same line is the (unset) outer n and set -u aborts.
  local f=examples/base-schedules/_zero_probe_s${n}_${sp}${pf:+_pf$pf}.json
  { echo "["; for ((i=0;i<n;i++)); do
      echo "  {\"op\":\"place\",\"filter\":{\"PP\":$i},\"devices\":[0,1],\"stream\":\"default_stream\"},"; done
    echo "  {\"op\":\"replicate\",\"filter\":{\"PP\":\"*\"},\"devices\":[0,1],\"reduce_stream\":\"dp_stream\",\"shard_grads\":$sg,\"shard_params\":$sp${pf:+,\"prefetch_distance\":$pf}},"
    echo "  {\"op\":\"split\",\"filter\":{},\"dim_name\":\"MB\",\"num_microbatches\":1}"; echo "]"; } > $f
  echo $f
}
DIM=4096; HID=16384
# Slopes per stage-bytes: zero3 as shipped 3.0 (F37), zero3_pf1 2.0 (G-4 budget), dp 4.0.
for MODE in zero3 zero3_pf1 dp; do
  PF=""; case $MODE in zero3) SP=true SG=true;; zero3_pf1) SP=true SG=true PF=1;; dp) SP=false SG=false;; esac
  for N in 2 4 8; do
    S=$(mk $N $SP $SG $PF)
    LOG=logs/zero3_peak_${MODE}_s${N}.log
    echo "== $MODE stages=$N $(date -Is)" >> $OUT
    CUDA_VISIBLE_DEVICES=$CARDS timeout 1200 \
      python examples/test_harness.py --test-file examples/test_tp_mlp.py \
        --base-schedule $S --schedule custom \
        --stages $N --tp 1 --init random --dim $DIM --hidden $HID --warmup 1 --iters 2 \
        --temp-dir /var/tmp/ziweizho-ray > $LOG 2>&1
    RC=$?
    RUN=$(grep -oE "out/[0-9]{8}_[0-9]{6}" $LOG | tail -1)
    echo "rc=$RC run=$RUN per_stage_bytes=$(( 2*DIM*HID*4 ))" >> $OUT
    [ -n "$RUN" ] && python - "$RUN" <<'PY' >> $OUT 2>&1
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json")):
    m = json.load(open(f)); print(f"  dp_rank={m.get('dp_rank')} peak_memory_by_rank={m.get('peak_memory_by_rank')}")
PY
  done
done
echo "done $(date -Is)" >> $OUT
