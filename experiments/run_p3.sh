#!/usr/bin/env bash
# P3: is the per-collective skew tax global synchronization or NCCL? Four quiet
# cards (this is a timing), independent of the CP chains.
set -u
R=/raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH
cd $R
OUT=logs/p3.txt; MAX_WAIT=$((72*3600)); t0=$(date +%s)
echo "start $(date -Is)" >> $OUT
pick() { local n=$1 mu=${2:-10} ok=""; for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v mu=$mu '$2<mu && $3<100000 {print $1}' | head -$n | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 30; done; echo "$ok"; }
echo "== P3 skew topology (4 cards)" >> $OUT
while :; do CARDS4=$(pick 4) && break
  [ $(( $(date +%s) - t0 )) -gt $MAX_WAIT ] && { echo "gave up (4 cards) $(date -Is)" >> $OUT; exit 0; }; sleep 300; done
echo "cards=$CARDS4 $(date -Is)" >> $OUT
CUDA_VISIBLE_DEVICES=$CARDS4 timeout 900 python -m torch.distributed.run --nproc_per_node=4 \
  experiments/probe_skew_topology.py > logs/p3_skew.log 2>&1; echo "rc=$?" >> $OUT
cat logs/p3_skew.log | grep -vE "Warning|^$" | head -40 >> $OUT
echo "done $(date -Is)" >> $OUT
