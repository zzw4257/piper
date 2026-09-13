#!/usr/bin/env bash
# Stage G on the ZJU H200 host: the runs that never got four quiet cards on
# catalyst-fleet1, plus an independent-hardware replication of F42/F44.
# Ordered cheap-first so a card window that closes early still yields something.
set -u
B=$HOME; R=$B/piper-tp
export PATH=$B/piper-venv/bin:$HOME/.local/bin:$PATH
cd $R
OUT=logs/h200.txt; mkdir -p logs
T=$B/ray_tmp; mkdir -p $T
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT

# pick <n> <max_util> <min_free_gib>: n cards, three samples 20s apart
pick() { local n=$1 mu=$2 mf=$3 ok=""
  for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
         | awk -F', ' -v mu=$mu -v mf=$mf '$2<mu && ($4-$3)/1024>mf {print $1}' | head -$n | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1
    sleep 20
  done; echo "$ok"; }

wait_for() { local n=$1 mu=$2 mf=$3 max=$4 t0=$(date +%s) c
  while :; do c=$(pick $n $mu $mf) && { echo "$c"; return 0; }
    [ $(( $(date +%s) - t0 )) -gt $max ] && return 1; sleep 60; done; }

SKIP1=${SKIP1:-0}
# ---- 1. out-of-band CP numerics on this hardware (2 cards, ~1 min) ----------
if [ "$SKIP1" = 0 ]; then
C=$(wait_for 2 90 20 1800) || { echo "no 2 cards" >> $OUT; exit 1; }
echo "== [1] torchrun CP gate  cards=$C $(date -Is)" >> $OUT
CUDA_VISIBLE_DEVICES=$C timeout 600 python -m torch.distributed.run --nproc_per_node=2 \
  test/test_cp_equivalence.py > logs/h200_cp_gate.log 2>&1; echo "rc=$?" >> $OUT
grep -E "CP=|Error|assert" logs/h200_cp_gate.log | tail -3 >> $OUT
fi

# ---- 2. P3: skew topology, memory-light, needs 4 quiet cards ----------------
C4=$(wait_for 4 15 4 7200) || { echo "[2] no 4 quiet cards" >> $OUT; C4=""; }
if [ -n "$C4" ]; then
  echo "== [2] P3 skew topology  cards=$C4 $(date -Is)" >> $OUT
  for PL in 4 16 64; do
    echo "-- payload ${PL}MiB" >> $OUT
    CUDA_VISIBLE_DEVICES=$C4 PAYLOAD_MIB=$PL ITERS=40 timeout 900 \
      python -m torch.distributed.run --nproc_per_node=4 experiments/probe_skew_topology.py \
      > logs/h200_p3_${PL}.log 2>&1; echo "rc=$?" >> $OUT
    grep -vE "Warning|^$|torchrun|WARNING" logs/h200_p3_${PL}.log | tail -40 >> $OUT
  done
fi

# ---- 3. four-step ring: spliced / hoist d=1 / hoist d=0 ---------------------
C4=$(wait_for 4 20 60 7200) || { echo "[3] no 4 cards with memory" >> $OUT; C4=""; }
if [ -n "$C4" ]; then
  echo "== [3] CP=4 ring  cards=$C4 $(date -Is)" >> $OUT
  for S in cp4_ring_dp cp4_ring_dp_hoist cp4_ring_dp_hoist_unbounded; do
    echo "-- $S" >> $OUT
    CUDA_VISIBLE_DEVICES=$C4 timeout 2400 python experiments/check_cp_equivalence.py --cp 4 \
      --cp2-schedule $S --skip-negative-control \
      -- --temp-dir $T --seq 8192 --dim 512 --heads 8 --batch-size 2 --iters 5 \
      > logs/h200_cp4_$S.log 2>&1; echo "rc=$?" >> $OUT
    grep -E "^CP=|peak_memory|worst|Error|failed|assert" logs/h200_cp4_$S.log | tail -12 >> $OUT
  done
fi

# ---- 4. F42/F44 memory slopes replicated on different hardware (2 cards) ----
C=$(wait_for 2 90 60 7200) || { echo "[4] no 2 cards with memory" >> $OUT; echo "done $(date -Is)" >> $OUT; exit 0; }
echo "== [4] memory slopes  cards=$C $(date -Is)" >> $OUT
mk() { local n=$1 pf=${2:-} sp=${3:-true} sg=${4:-true}
  local f=examples/base-schedules/_h200_s${n}_${sp}${pf:+_pf$pf}.json
  { echo "["; for ((i=0;i<n;i++)); do
      echo "  {\"op\":\"place\",\"filter\":{\"PP\":$i},\"devices\":[0,1],\"stream\":\"default_stream\"},"; done
    echo "  {\"op\":\"replicate\",\"filter\":{\"PP\":\"*\"},\"devices\":[0,1],\"reduce_stream\":\"dp_stream\",\"shard_grads\":$sg,\"shard_params\":$sp${pf:+,\"prefetch_distance\":$pf}},"
    echo "  {\"op\":\"split\",\"filter\":{},\"dim_name\":\"MB\",\"num_microbatches\":1}"; echo "]"; } > $f
  echo $f; }
DIM=4096; HID=16384; PSB=$(( (2*DIM*HID + 2*DIM*DIM)*4 ))
run() { local label=$1 n=$2 pf=$3 sp=$4 sg=$5; shift 5
  local S=$(mk $n "$pf" $sp $sg) LOG=logs/h200_mem_${label}_s${n}.log
  echo "== $label stages=$n $(date -Is)" >> $OUT
  env "$@" CUDA_VISIBLE_DEVICES=$C timeout 1200 \
    python examples/test_harness.py --test-file examples/test_tp_mlp.py --base-schedule $S --schedule custom \
      --stages $n --tp 1 --init random --dim $DIM --hidden $HID --warmup 1 --iters 3 \
      --temp-dir $T > $LOG 2>&1
  local RC=$? RUN=$(grep -oE "out/[0-9]{8}_[0-9]{6}" $LOG | tail -1)
  echo "rc=$RC run=$RUN per_stage_bytes=$PSB" >> $OUT
  [ $RC -ne 0 ] && grep -E "Error|error" $LOG | tail -2 | cut -c1-160 >> $OUT
  [ -n "$RUN" ] && python - "$RUN" <<'PY' >> $OUT 2>&1
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json")):
    m = json.load(open(f)); print(f"  dp_rank={m.get('dp_rank')} peak_memory_by_rank={m.get('peak_memory_by_rank')} min_iter_s={min(m['iter_times_s']):.4f}")
PY
}
for N in 2 4 8; do
  run dp          $N ""  false false
  run zero3       $N ""  true  true
  run pf1         $N 1   true  true
  run pf1_pool    $N 1   true  true  PIPER_BUFFER_POOL=1
  run shipped_pool $N "" true  true  PIPER_BUFFER_POOL=1
done
echo "done $(date -Is)" >> $OUT
