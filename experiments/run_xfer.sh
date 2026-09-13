#!/usr/bin/env bash
set -u
R=$1; V=$2; TAG=$3
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/xfer_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '$2<20 && ($4-$3)/1024>6 {print $1}' | head -4 | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 4 ] && return 1; sleep 20; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick) && break
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C" >> $OUT
CUDA_VISIBLE_DEVICES=$C XFER_OUT=$R/logs/xfer_${TAG}.json timeout 2400 \
  python -m torch.distributed.run --nproc_per_node=4 experiments/probe_skew_transfer.py >> $OUT 2>&1
echo "rc=$?" >> $OUT
# fine sweep for the ring localization knee, same cards
CUDA_VISIBLE_DEVICES=$C PAYLOADS=16,24,32,40,48,64,80,96 ITERS=25 \
  LAW_OUT=$R/logs/knee_${TAG}.json timeout 2400 \
  python -m torch.distributed.run --nproc_per_node=4 experiments/probe_collective_law.py \
  >> logs/knee_${TAG}.txt 2>&1
echo "knee rc=$? done $(date -Is)" >> $OUT
