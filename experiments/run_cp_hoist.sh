#!/usr/bin/env bash
# G-3 runs, chained behind run_cp_gate.sh. 2 cards: hoisted numerics must still
# match dense. 4 cards: spliced vs hoisted(d=1) vs hoisted(d=0) -- P1 revised
# predicts d=0 peak memory grows with steps (every chunk resident), d=1 does not.
set -u
R=/raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH
cd $R
OUT=logs/cp_hoist.txt; MAX_WAIT=$((48*3600)); t0=$(date +%s)
echo "start $(date -Is)" >> $OUT
until grep -qE "^(done|gave up)" logs/cp_gate.txt 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up waiting for gate" >> $OUT; exit 1; }
  sleep 180
done
pick() { local n=$1 ok=""; for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2<10 && $3<100000 {print $1}' | head -$n | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 30; done; echo "$ok"; }
while :; do CARDS=$(pick 2) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up (2)" >> $OUT; exit 1; }; sleep 120; done
echo "cards=$CARDS $(date -Is)" >> $OUT
echo "== hoisted CP=2 vs dense" >> $OUT
CUDA_VISIBLE_DEVICES=$CARDS timeout 2400 python experiments/check_cp_equivalence.py --cp 2 \
  --cp2-schedule cp2_ring_dp_hoist --skip-negative-control --extra --temp-dir /var/tmp/ziweizho-ray \
  > logs/cp_hoist_2.log 2>&1; echo "rc=$?" >> $OUT
grep -E "^CP=|peak_memory|Error|failed|assert" logs/cp_hoist_2.log | tail -10 >> $OUT

echo "== 4 cards: spliced / hoisted d=1 / hoisted d=0" >> $OUT
while :; do CARDS4=$(pick 4) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up (4) $(date -Is)" >> $OUT; exit 0; }; sleep 300; done
echo "cards=$CARDS4 $(date -Is)" >> $OUT
for S in cp4_ring_dp cp4_ring_dp_hoist cp4_ring_dp_hoist_unbounded; do
  echo "-- $S" >> $OUT
  CUDA_VISIBLE_DEVICES=$CARDS4 timeout 2400 python experiments/check_cp_equivalence.py --cp 4 \
    --cp2-schedule $S --skip-negative-control \
    --extra --temp-dir /var/tmp/ziweizho-ray --seq 8192 --dim 512 --heads 8 --batch-size 2 --iters 5 \
    > logs/cp_hoist_4_$S.log 2>&1; echo "rc=$?" >> $OUT
  grep -E "^CP=|peak_memory|worst|Error|failed|assert" logs/cp_hoist_4_$S.log | tail -12 >> $OUT
done
echo "done $(date -Is)" >> $OUT
