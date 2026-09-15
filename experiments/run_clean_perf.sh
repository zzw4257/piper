#!/usr/bin/env bash
# Re-validate the paper's headline directive numbers on a quiet machine.
# Every one of them (order 16%, fusion 12-19%, stream 1.06x) was measured while
# other tenants held the interconnect; F36 recorded that a strict quiet window
# never arrived. This runs when one does, with the drained clock from F61 and
# arms interleaved per F11.
set -u
R=$1; V=$2; T=$3; TAG=$4; CARDS=$5; REPS=${6:-5}
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/cleanperf_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) cards=$CARDS ===" >> $OUT
echo "load: $(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd,)" >> $OUT

run() {  # run <schedule> <n_cards> <extra...>
  local S=$1 N=$2; shift 2
  local C=$(echo $CARDS | cut -d, -f1-$N)
  local RUN=$(CUDA_VISIBLE_DEVICES=$C timeout 1800 python examples/test_harness.py \
    --test-file examples/test_tp_mlp.py --base-schedule examples/base-schedules/$S.json \
    --schedule custom --init random --warmup 2 --iters 10 --temp-dir $T "$@" 2>&1 \
    | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  python - "$RUN" <<'PY' 2>/dev/null
import glob, json, sys
rows=[json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json"))]
if not rows: print("NA"); raise SystemExit
print(f"{max(r.get('total_timed_s',0)/max(r.get('timed_iters',1),1)*1000 for r in rows):.2f}")
PY
}

pair() {  # pair <label> <A> <B> <n_cards> <extra...>
  local L=$1 A=$2 B=$3 N=$4; shift 4
  echo "-- $L : $A  vs  $B  (${N}卡)" >> $OUT
  for i in $(seq 1 $REPS); do
    a=$(run $A $N "$@"); b=$(run $B $N "$@")
    echo "   rep$i  $A=${a}ms  $B=${b}ms" >> $OUT
  done
}

# 1. order: the paper's 16% claim, PP=2 x TP=2 on four cards
pair "order (1f1b vs gpipe)" pp2_tp2_mb4_1f1b pp2_tp2_mb4_gpipe 4 --tp 2 --stages 2
# 2. fusion under 1f1b: the paper says indistinguishable
pair "fusion under 1f1b"     pp2_tp2_mb4_1f1b pp2_tp2_mb4_1f1b_fused 4 --tp 2 --stages 2
# 3. fusion on a single stage: the paper's 12-19% claim
pair "fusion single-stage"   s2_tp2_mb4 tp2_mb4_fused 2 --tp 2 --stages 1
# 4. microbatch count at fixed total work: the paper's 2.6x degradation
echo "-- microbatch sweep (固定总工作量)" >> $OUT
for i in $(seq 1 $REPS); do
  line="   rep$i"
  for S in s2_tp2_mb1 s2_tp2_mb2 s2_tp2_mb4 s2_tp2_mb8; do
    line="$line  $S=$(run $S 2 --tp 2 --stages 1)ms"
  done
  echo "$line" >> $OUT
done
echo "done $(date -Is)" >> $OUT
