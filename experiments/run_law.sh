#!/usr/bin/env bash
# The collective-cost law sweep. Takes repo root and venv bin as arguments so
# the same script runs on catalyst-fleet1 and the H200 host.
set -u
R=$1; V=$2; TAG=$3; NEED=${4:-4}
export PATH=$V:$PATH
cd $R; mkdir -p logs
OUT=logs/law_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
cuda_ok() { python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count()>0 else 1)' >/dev/null 2>&1; }
pick() { cuda_ok || return 1; local n=$1 ok=""
  for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
         | awk -F', ' '$2<20 && ($4-$3)/1024>6 {print $1}' | head -$n | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 20; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick $NEED) && break
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "no $NEED quiet cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C $(date -Is)" >> $OUT
CUDA_VISIBLE_DEVICES=$C LAW_OUT=$R/logs/law_${TAG}.json timeout 3600 \
  python -m torch.distributed.run --nproc_per_node=$NEED experiments/probe_collective_law.py \
  >> $OUT 2>&1
echo "rc=$? done $(date -Is)" >> $OUT
