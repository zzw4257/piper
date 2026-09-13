#!/usr/bin/env bash
# Full numerical verification of the forwarded-output detach skip, plus an
# interleaved before/after measurement of the node times it is meant to cut.
set -u
R=$1; V=$2; T=$3; TAG=$4
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/fwdskip_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local n=$1; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F', ' '($3-$2)/1024>30 {print $1}' | head -$n | paste -sd,; }
C2=$(pick 2); C4=$(pick 4)
echo "cards2=$C2 cards4=$C4" >> $OUT

step() { echo "== $1 $(date -Is)" >> $OUT; }

step "torchrun TP math (no Piper)"
CUDA_VISIBLE_DEVICES=$C2 timeout 600 python -m torch.distributed.run --nproc_per_node=2 \
  test/test_tp_equivalence.py > logs/fs_tp_gate.log 2>&1; echo "rc=$?" >> $OUT

step "torchrun CP math (no Piper)"
CUDA_VISIBLE_DEVICES=$C2 timeout 600 python -m torch.distributed.run --nproc_per_node=2 \
  test/test_cp_equivalence.py > logs/fs_cp_gate.log 2>&1; echo "rc=$?" >> $OUT
grep -E "CP=|controls" logs/fs_cp_gate.log | tail -1 >> $OUT

step "in-Piper TP=2 vs TP=1 (shipped path, 3 optimizer steps, with negative control)"
CUDA_VISIBLE_DEVICES=$C2 timeout 2400 python experiments/check_tp_equivalence.py \
  > logs/fs_tp_piper.log 2>&1; echo "rc=$?" >> $OUT
grep -E "TP=|negative|iteration|assert|Error" logs/fs_tp_piper.log | tail -6 >> $OUT

step "in-Piper CP=2 vs dense (with negative control)"
CUDA_VISIBLE_DEVICES=$C2 timeout 2400 python experiments/check_cp_equivalence.py --cp 2 \
  -- --temp-dir $T > logs/fs_cp2.log 2>&1; echo "rc=$?" >> $OUT
grep -E "^CP=|negative|worst|Error|assert" logs/fs_cp2.log | tail -5 >> $OUT

step "in-Piper CP=4, spliced and hoisted"
for S in cp4_ring_dp cp4_ring_dp_hoist; do
  CUDA_VISIBLE_DEVICES=$C4 timeout 2400 python experiments/check_cp_equivalence.py --cp 4 \
    --cp2-schedule $S --skip-negative-control -- --temp-dir $T --seq 8192 --dim 512 --heads 8 \
    --batch-size 2 --iters 5 > logs/fs_cp4_$S.log 2>&1
  echo "  $S rc=$?" >> $OUT
  grep -E "^CP=4 ring ==|Error|assert" logs/fs_cp4_$S.log | tail -2 >> $OUT
done

step "node timings, interleaved A/B against the saved pre-change executor"
for REP in 1 2 3; do
  for ARM in after before; do
    [ $ARM = before ] && cp /var/tmp/ziweizho-scripts/executors.before_fwdskip.py src/executors.py 2>/dev/null \
                      || cp /var/tmp/ziweizho-scripts/executors.after_fwdskip.py src/executors.py 2>/dev/null
    D=/var/tmp/fs_${ARM}_$REP; rm -rf $D; mkdir -p $D
    CUDA_VISIBLE_DEVICES=$C4 PIPER_TIME_TRACE=$D timeout 1200 \
      python examples/test_harness.py --test-file examples/test_ring_attn.py \
      --base-schedule examples/base-schedules/cp4_ring_dp.json --schedule custom \
      --steps 4 --seq 1024 --dim 512 --heads 8 --batch-size 8 --warmup 2 --iters 12 \
      --temp-dir $T > /tmp/fs_${ARM}_$REP.log 2>&1
    echo "  rep$REP $ARM rc=$?" >> $OUT
    python experiments/summarize_inner.py $D 2>/dev/null | grep -E "^forward|^backward |per iteration" \
      | sed "s/^/    $ARM /" >> $OUT
  done
done
cp /var/tmp/ziweizho-scripts/executors.after_fwdskip.py src/executors.py
echo "done $(date -Is)" >> $OUT
