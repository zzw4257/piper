#!/usr/bin/env bash
# F43 fix candidate: bounded buffer pool (PIPER_BUFFER_POOL=1), no host wait.
# Predictions: prefetch_distance=1 + pool  -> slope ~2.0 (like host-sync, without the stall);
#              shipped ZeRO-3 + pool       -> ~3.0 (grad half fixed; forward gather-all remains).
set -u
R=/raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH
cd $R
OUT=logs/zero3_pool.txt; MAX_WAIT=$((24*3600)); t0=$(date +%s)
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
run() { local label=$1 n=$2 pf=$3; shift 3
  local S=$(mk $n $pf) LOG=logs/zero3_pool_${label}_s${n}.log
  echo "== $label stages=$n $(date -Is)" >> $OUT
  env "$@" CUDA_VISIBLE_DEVICES=$CARDS timeout 1200 \
    python examples/test_harness.py --test-file examples/test_tp_mlp.py --base-schedule $S --schedule custom \
      --stages $n --tp 1 --init random --dim $DIM --hidden $HID --warmup 1 --iters 3 \
      --temp-dir /var/tmp/ziweizho-ray > $LOG 2>&1
  local RC=$? RUN=$(grep -oE "out/[0-9]{8}_[0-9]{6}" $LOG | tail -1)
  echo "rc=$RC run=$RUN per_stage_bytes=$PSB" >> $OUT
  [ $RC -ne 0 ] && grep -E "Error|error" $LOG | tail -2 | cut -c1-160 >> $OUT
  [ -n "$RUN" ] && python - "$RUN" <<'PY' >> $OUT 2>&1
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json")):
    m = json.load(open(f)); print(f"  dp_rank={m.get('dp_rank')} peak_memory_by_rank={m.get('peak_memory_by_rank')} min_iter_s={min(m['iter_times_s']):.4f} losses={[round(x,4) for x in m['losses']]}")
PY
}
for N in 2 4 8; do run pf1_pool $N 1 PIPER_BUFFER_POOL=1; done
for N in 2 4 8; do run shipped_pool $N "" PIPER_BUFFER_POOL=1; done
# same-window A/B at 8 stages, interleaved, for a timing hint (cards are shared; hint only)
for i in 1 2; do run pf1_pool_ab 8 1 PIPER_BUFFER_POOL=1; run pf1_hostsync_ab 8 1 PIPER_AG_HOST_SYNC=1; run pf1_plain_ab 8 1 PIPER_NOOP=1; done
echo "done $(date -Is)" >> $OUT
