#!/usr/bin/env bash
# Interleaved before/after for the forwarded-detach skip. Swapping a source file
# between harness launches, min over reps, spread reported (F54's rule).
set -u
R=$1; V=$2; T=$3; TAG=$4; REPS=${5:-5}
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/ab_fwdskip_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '$2<25 && ($4-$3)/1024>30 {print $1}' | head -4 | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 4 ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick) && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no quiet cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C" >> $OUT
for REP in $(seq 1 $REPS); do
  for ARM in before after; do
    cp $R/snap/executors.${ARM}.py src/executors.py
    D=/tmp/ab_${ARM}_$REP; rm -rf $D; mkdir -p $D
    CUDA_VISIBLE_DEVICES=$C PIPER_TIME_TRACE=$D timeout 1200 \
      python examples/test_harness.py --test-file examples/test_ring_attn.py \
      --base-schedule examples/base-schedules/cp4_ring_dp.json --schedule custom \
      --steps 4 --seq 1024 --dim 512 --heads 8 --batch-size 8 --warmup 2 --iters 12 \
      --temp-dir $T > /tmp/ab_${ARM}_$REP.log 2>&1
    echo "rep$REP $ARM rc=$? $(python experiments/summarize_inner.py $D 2>/dev/null | grep 'per iteration')" >> $OUT
    python experiments/summarize_inner.py $D 2>/dev/null | grep -E "^forward  |^backward " | sed "s/^/  $ARM /" >> $OUT
  done
done
cp $R/snap/executors.after.py src/executors.py
echo "done $(date -Is)" >> $OUT
