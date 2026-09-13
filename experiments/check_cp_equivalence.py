"""End-to-end CP equivalence inside Piper: CP=2 ring must match CP=1 dense.

    CUDA_VISIBLE_DEVICES=0,4 python experiments/check_cp_equivalence.py

Both runs draw the same full-sequence data and start from the same projection
weights. CP=2 reports one loss per rank, each a mean over its half of the
sequence; with equal chunks their mean is the dense loss. The comparison spans
optimizer steps so the backward ring is constrained: without it each rank's dK/dV
would be missing the other rank's contribution and the weights would diverge
from iteration 2. A third run with the ring dropped proves the comparison can
fail.

In-Piper counterpart of test/test_cp_equivalence.py (torchrun, no Ray, no DAG).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

RTOL, ATOL = 1e-4, 1e-4


def _run(schedule: str, extra: list[str], repo: str) -> dict:
    cmd = [
        sys.executable, "examples/test_harness.py",
        "--test-file", "examples/test_ring_attn.py",
        "--base-schedule", f"examples/base-schedules/{schedule}.json",
        "--schedule", "custom", *extra,
    ]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        sys.exit(f"{schedule} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    dirs = re.findall(r"out/\d{8}_\d{6}", proc.stdout)
    if not dirs:
        sys.exit(f"{schedule}: could not find the run directory in harness output")
    run_dir = os.path.join(repo, dirs[-1])
    metrics = [json.load(open(f)) for f in sorted(glob.glob(f"{run_dir}/cp_metrics_dp*.json"))]
    if not metrics:
        sys.exit(f"{schedule}: no cp_metrics_dp*.json under {run_dir}")
    return {"dir": dirs[-1], "metrics": metrics}


def _mean_over_ranks(metrics):
    per_rank = [m["losses"] for m in sorted(metrics, key=lambda m: m["dp_rank"])]
    assert len({len(l) for l in per_rank}) == 1, "ranks report different iteration counts"
    return [sum(step) / len(step) for step in zip(*per_rank)], per_rank


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--skip-negative-control", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after a literal --: harness/test args passed to every run")
    ap.add_argument("--cp", type=int, default=2, help="CP degree / ring steps / ranks")
    ap.add_argument("--negative-schedule", default=None,
                    help="default: cp{N}_no_ring_dp")
    ap.add_argument("--cp2-schedule", default=None,
                    help="cp2_ring (no replicate) is valid only for a single "
                         "iteration: projection grads then diverge across ranks.")
    args = ap.parse_args()
    if args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]

    n = args.cp
    sched = args.cp2_schedule or f"cp{n}_ring_dp"
    cp2 = _run(sched, ["--steps", str(n), *args.extra], args.repo)
    cp1 = _run("cp1", ["--steps", "1", *args.extra], args.repo)
    mean2, ranks2 = _mean_over_ranks(cp2["metrics"])
    l1 = cp1["metrics"][0]["losses"]
    print(f"CP=1 dense ({cp1['dir']}):        {[round(x, 6) for x in l1]}")
    for r, l in enumerate(ranks2):
        print(f"CP={n} rank{r} [{sched}] ({cp2['dir']}): {[round(x, 6) for x in l]}")
    print(f"CP={n} mean over ranks:             {[round(x, 6) for x in mean2]}")
    for m in cp2["metrics"]:
        print(f"  rank{m['dp_rank']} peak_memory_by_rank={m['peak_memory_by_rank']} "
              f"iter_times_s={[round(t, 4) for t in m['iter_times_s']]}")
    assert len(ranks2) == n, f"expected {n} CP ranks, got {len(ranks2)}"
    assert len(l1) == len(mean2), "iteration counts differ"
    worst = 0.0
    for i, (a, b) in enumerate(zip(mean2, l1)):
        worst = max(worst, abs(a - b))
        assert abs(a - b) <= ATOL + RTOL * abs(b), (
            f"iteration {i}: CP=2 mean loss {a} != dense {b}. Iteration 0 alone "
            f"blames the forward ring; a later iteration blames the backward ring "
            f"or the projection-weight replicate."
        )
    print(f"CP={n} ring == CP=1 dense across {len(l1)} iterations, worst |diff| {worst:.3e}")

    if not args.skip_negative_control:
        bad = _run(args.negative_schedule or f"cp{n}_no_ring_dp",
                   ["--steps", str(n), *args.extra], args.repo)
        mean_bad, _ = _mean_over_ranks(bad["metrics"])
        diff = max(abs(a - b) for a, b in zip(mean_bad, l1))
        assert diff > ATOL + RTOL * max(abs(x) for x in l1), (
            f"dropping ring_exchange left the loss within tolerance ({diff:.3e}); "
            f"the comparison has no power at this size"
        )
        print(f"negative control (no ring) differs by {diff:.3e}: the comparison can fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
