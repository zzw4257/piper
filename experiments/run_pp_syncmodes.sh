#!/usr/bin/env bash
# The case `narrow` would have been unsafe in: a PP rank that computes no loss,
# so its loss-event wait is empty. Needs four cards (PP=2 x TP=2).
set -u
R=$1; V=$2; T=$3; TAG=$4
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/ppsync_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '($3-$2)/1024>30 {print $1}' | head -4 | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 4 ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick) && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C load: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd,)" >> $OUT
for M in device narrow defer; do
  echo "== mode=$M  TP x PP on four cards, 3 optimizer steps" >> $OUT
  PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C timeout 3000 \
    python experiments/check_tp_equivalence.py --pp > logs/ppsync_${TAG}_$M.log 2>&1
  echo "  rc=$?" >> $OUT
  grep -E "PASS|FAIL|assert|Error|worst|TP x PP|pp" logs/ppsync_${TAG}_$M.log | tail -4 | sed 's/^/    /' >> $OUT
done
echo "done $(date -Is)" >> $OUT
