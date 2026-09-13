#!/usr/bin/env bash
# Stage G-2 gate runs, chained behind the F37 memory job so the two never grab
# the same cards. Then P3 on four cards, which may wait much longer.
set -u
R=/raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH
cd $R
OUT=logs/cp_gate.txt; MAX_WAIT=$((36*3600)); t0=$(date +%s)
echo "start $(date -Is)" >> $OUT
until grep -qE "^(done|gave up)" logs/zero3_peak.txt 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up waiting for peak job" >> $OUT; exit 1; }
  sleep 120
done
pick() {  # pick <n>: n cards with util<10% and mem<8GB, 3 samples 30s apart
  local n=$1 ok=""; for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2<10 && $3<8000 {print $1}' | head -$n | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1
    sleep 30
  done; echo "$ok"
}
while :; do CARDS=$(pick 2) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up (2 cards)" >> $OUT; exit 1; }; sleep 120; done
echo "cards=$CARDS $(date -Is)" >> $OUT
echo "== torchrun CP gate" >> $OUT
CUDA_VISIBLE_DEVICES=$CARDS timeout 600 python -m torch.distributed.run --nproc_per_node=2 \
  test/test_cp_equivalence.py > logs/cp_gate_torchrun.log 2>&1; echo "rc=$?" >> $OUT
grep -E "CP=|Error|error|assert" logs/cp_gate_torchrun.log | tail -4 >> $OUT
echo "== in-Piper CP equivalence" >> $OUT
CUDA_VISIBLE_DEVICES=$CARDS timeout 2400 python experiments/check_cp_equivalence.py \
  --extra --temp-dir /var/tmp/ziweizho-ray > logs/cp_gate_piper.log 2>&1; echo "rc=$?" >> $OUT
grep -E "^CP=|negative control|Error|failed|assert" logs/cp_gate_piper.log | tail -8 >> $OUT
if ! grep -q "ring == CP=1 dense" logs/cp_gate_piper.log; then
  # Insurance for a scarce window: if the replicate-composed run failed (F38's
  # empty reductions are untested at runtime), validate the ring alone on one
  # iteration -- no replicate, so no weight sync, so only iteration 0 is valid.
  echo "== fallback: cp2_ring (no replicate), one iteration" >> $OUT
  CUDA_VISIBLE_DEVICES=$CARDS timeout 1800 python experiments/check_cp_equivalence.py \
    --cp2-schedule cp2_ring --extra --temp-dir /var/tmp/ziweizho-ray --warmup 0 --iters 1 \
    > logs/cp_gate_piper_fallback.log 2>&1; echo "rc=$?" >> $OUT
  grep -E "^CP=|negative control|Error|failed|assert" logs/cp_gate_piper_fallback.log | tail -8 >> $OUT
fi
echo "== P3 skew topology (4 cards)" >> $OUT
while :; do CARDS4=$(pick 4) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up (4 cards) $(date -Is)" >> $OUT; exit 0; }; sleep 300; done
echo "cards=$CARDS4 $(date -Is)" >> $OUT
CUDA_VISIBLE_DEVICES=$CARDS4 timeout 900 python -m torch.distributed.run --nproc_per_node=4 \
  experiments/probe_skew_topology.py > logs/p3_skew.log 2>&1; echo "rc=$?" >> $OUT
cat logs/p3_skew.log | grep -vE "Warning|^$" | head -40 >> $OUT
echo "done $(date -Is)" >> $OUT
