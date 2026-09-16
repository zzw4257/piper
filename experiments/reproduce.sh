#!/usr/bin/env bash
# One entry point for everything this project claims. Each block prints PASS or
# the number it measured, so a reader can run it and see the claims stand.
#
#   experiments/reproduce.sh cpu          # no GPU, ~15 s
#   experiments/reproduce.sh gpu2 <a,b>   # two cards
#   experiments/reproduce.sh gpu4 <a,b,c,d>
#   experiments/reproduce.sh all <a,b,c,d>
set -u
MODE=${1:-cpu}; CARDS=${2:-}
R=$(cd "$(dirname "$0")/.." && pwd); cd $R
V=${PIPER_VENV:-/var/tmp/ziweizho-piper-venv/bin}; export PATH=$V:$PATH
T=${PIPER_RAY_TMP:-/var/tmp/ziweizho-ray}
hr() { printf '%s\n' "------------------------------------------------------------"; }
say() { printf '\n== %s\n' "$*"; }

cuda_ok() { python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count()>0 else 1)' 2>/dev/null; }

do_cpu() {
  say "CPU: the whole test suite (82 tests, no GPU)"
  python -m pytest -m "not gpu" -q 2>&1 | tail -2
  say "CPU: what the lowered DAG looks like for TP, for CP, and for ZeRO-3 prefetch"
  PYTHONPATH=examples:. python experiments/dump_dag.py \
    --schedule examples/base-schedules/tp2.json 2>/dev/null | tail -6
  PYTHONPATH=examples:. python experiments/dump_dag.py --model ring \
    --schedule examples/base-schedules/cp2_ring_dp_hoist.json --steps 3 2>/dev/null \
    | grep -E "RING_COMM|COMPUTE" | grep PASS=F | cut -c1-96
  say "CPU: K/V survive segmentation as bare placeholders (the CP design's premise)"
  PYTHONPATH=examples:. python experiments/probe_cp_segments.py --steps 4 2>/dev/null \
    | sed -n '/=== verdicts/,$p'
}

do_gpu2() {
  cuda_ok || { echo "CUDA unavailable on this host (F62)"; return 1; }
  say "2 GPU: tensor-parallel math, outside Piper (torchrun, no Ray, no DAG)"
  CUDA_VISIBLE_DEVICES=$CARDS python -m torch.distributed.run --nproc_per_node=2 \
    test/test_tp_equivalence.py 2>&1 | grep -E "^PASS|^FAIL|Error" | tail -1
  say "2 GPU: ring-attention math, outside Piper, with two negative controls"
  CUDA_VISIBLE_DEVICES=$CARDS python -m torch.distributed.run --nproc_per_node=2 \
    test/test_cp_equivalence.py 2>&1 | grep -E "CP=2 ring|Error" | tail -1
  say "2 GPU: TP=2 equals TP=1 inside Piper across optimizer steps, and dropping the directive breaks it"
  CUDA_VISIBLE_DEVICES=$CARDS python experiments/check_tp_equivalence.py 2>&1 \
    | grep -E "^PASS|^control" | tail -2
  say "2 GPU: CP=2 ring equals dense attention inside Piper, with a no-ring control"
  CUDA_VISIBLE_DEVICES=$CARDS python experiments/check_cp_equivalence.py --cp 2 \
    -- --temp-dir $T 2>&1 | grep -E "ring ==|negative" | tail -2
}

do_gpu4() {
  cuda_ok || { echo "CUDA unavailable on this host (F62)"; return 1; }
  say "4 GPU: TP x PP against a one-GPU baseline"
  CUDA_VISIBLE_DEVICES=$CARDS python experiments/check_tp_equivalence.py --pp 2>&1 \
    | grep -E "PASS  TP x PP|FAIL" | tail -1
  say "4 GPU: CP=4, spliced and hoisted, both against dense"
  for S in cp4_ring_dp cp4_ring_dp_hoist; do
    printf '   %-22s ' "$S"
    CUDA_VISIBLE_DEVICES=$CARDS python experiments/check_cp_equivalence.py --cp 4 \
      --cp2-schedule $S --skip-negative-control -- --temp-dir $T --seq 8192 --dim 512 \
      --heads 8 --batch-size 2 --iters 5 2>&1 | grep -oE "ring == CP=1 dense.*" | tail -1
  done
  say "4 GPU: the three host-sync modes compute identical parameters (fp64 checksum)"
  bash experiments/run_checksum_matrix.sh $R $V $T repro >/dev/null 2>&1
  sed -n '/CP=4 ring/,$p' logs/checksum_repro.txt 2>/dev/null | grep -E "mode=|rank0" | head -6
  say "4 GPU: collective cost is affine in bytes when the ranks are synchronized (F48)"
  CUDA_VISIBLE_DEVICES=$CARDS PAYLOADS=4,16,64,256 ITERS=20 \
    python -m torch.distributed.run --nproc_per_node=4 experiments/probe_collective_law.py 2>&1 \
    | grep -E "^ *[0-9]+\.0 all_reduce (synced|skewed)" | head -8
}

case "$MODE" in
  cpu)  do_cpu ;;
  gpu2) do_gpu2 ;;
  gpu4) do_gpu4 ;;
  all)  do_cpu; do_gpu2; do_gpu4 ;;
  *) echo "usage: $0 {cpu|gpu2|gpu4|all} [cards]"; exit 2 ;;
esac
hr
echo "findings: notes/log.md   evidence: results/   papers: paper/*.pdf (local)"
