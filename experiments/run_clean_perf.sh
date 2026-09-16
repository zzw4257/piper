#!/usr/bin/env bash
# Re-validate the paper's headline directive numbers on a quiet machine.
# All of them (order 16%, fusion 12-19%, the microbatch degradation) were
# measured while other tenants held the interconnect; F36 recorded that a strict
# quiet window never arrived. Per-schedule arguments matter: s2_* puts two
# stages on one pair of cards, tp2_* is a single stage, pp2_tp2_* is two stages
# across four. Drained clock (F61), arms interleaved (F11).
set -u
R=$1; V=$2; T=$3; TAG=$4; CARDS=$5; REPS=${6:-5}
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/cleanperf_${TAG}.txt
python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count()>0 else 1)' 2>/dev/null \
  || { echo "CUDA runtime unavailable (F62); refusing" | tee -a $OUT; exit 1; }
echo "=== start $(date -Is) host=$(hostname) cards=$CARDS ===" >> $OUT
echo "load: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd,)" >> $OUT

run() {  # run <schedule> <n_cards> <stages> <tp>
  local S=$1 N=$2 ST=$3 TP=$4
  local C=$(echo $CARDS | cut -d, -f1-$N)
  local RUN=$(CUDA_VISIBLE_DEVICES=$C timeout 1800 python examples/test_harness.py \
    --test-file examples/test_tp_mlp.py --base-schedule examples/base-schedules/$S.json \
    --schedule custom --init random --warmup 2 --iters 10 --stages $ST --tp $TP \
    --temp-dir $T 2>&1 | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  python - "$RUN" <<'PY' 2>/dev/null
import glob, json, sys
rows=[json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json"))]
print(f"{max(r.get('total_timed_s',0)/max(r.get('timed_iters',1),1)*1000 for r in rows):.2f}" if rows else "NA")
PY
}

pair() {  # pair <label> <A> <B> <n> <stages> <tp>
  local L=$1 A=$2 B=$3 N=$4 ST=$5 TP=$6
  echo "-- $L : $A vs $B (${N} cards, stages=$ST tp=$TP)" >> $OUT
  for i in $(seq 1 $REPS); do
    a=$(run $A $N $ST $TP); b=$(run $B $N $ST $TP)
    echo "   rep$i  $A=${a}ms  $B=${b}ms" >> $OUT
  done
}

pair "order 1f1b vs gpipe"  pp2_tp2_mb4_1f1b pp2_tp2_mb4_gpipe       4 2 2
pair "fusion under 1f1b"    pp2_tp2_mb4_1f1b pp2_tp2_mb4_1f1b_fused  4 2 2
pair "fusion single-stage"  tp2_mb4          tp2_mb4_fused           2 1 2

echo "-- microbatch sweep at fixed total work (2 cards, stages=2 tp=2)" >> $OUT
for i in $(seq 1 $REPS); do
  line="   rep$i"
  for S in s2_tp2_mb1 s2_tp2_mb2 s2_tp2_mb4 s2_tp2_mb8; do
    line="$line  ${S##s2_tp2_}=$(run $S 2 2 2)ms"
  done
  echo "$line" >> $OUT
done
echo "done $(date -Is)" >> $OUT
