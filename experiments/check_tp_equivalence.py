"""End-to-end TP equivalence inside Piper: TP=2 sharded must match TP=1 unsharded.

    CUDA_VISIBLE_DEVICES=0,4 python experiments/check_tp_equivalence.py

Runs both schedules through the harness with --init fixed, so both start from the
same global weights (sliced per rank for TP=2), and compares the per-iteration
losses. The comparison spans optimizer steps on purpose: a missing *backward*
all-reduce leaves iteration 1 correct and diverges from iteration 2, because each
rank would step on only its own partial input gradient.

This is the in-Piper counterpart to test/test_tp_equivalence.py, which checks the
same claim out of band under torchrun with no Ray or DAG involved.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

# Measured on 2xB200, dim 512, hidden 2048, fp32: iterations 1 and 3 agree
# exactly at 6 dp, iteration 2 differs by 1e-6. That is fp32 reduction-order
# noise across a different number of NCCL contributions, far below anything a
# misplaced collective produces (dropping either collective moves the loss in
# the third decimal; see tp2_no_collective).
RTOL, ATOL = 1e-4, 1e-4


def _run(schedule: str, extra: list[str], repo: str) -> dict:
    cmd = [
        sys.executable, "examples/test_harness.py",
        "--test-file", "examples/test_tp_mlp.py",
        "--base-schedule", f"examples/base-schedules/{schedule}.json",
        "--schedule", "custom", "--init", "fixed", *extra,
    ]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        sys.exit(f"{schedule} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    dirs = re.findall(r"out/\d{8}_\d{6}", proc.stdout)
    if not dirs:
        sys.exit(f"{schedule}: could not find the run directory in harness output")
    run_dir = os.path.join(repo, dirs[-1])
    metrics = [json.load(open(f)) for f in sorted(glob.glob(f"{run_dir}/tp_metrics_dp*.json"))]
    if not metrics:
        sys.exit(f"{schedule}: no tp_metrics_dp*.json under {run_dir}")
    return {"dir": dirs[-1], "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--skip-negative-control", action="store_true",
                    help="Skip the third run that proves the comparison can fail.")
    args = ap.parse_args()

    tp2 = _run("tp2", [], args.repo)
    tp1 = _run("tp1", ["--tp", "1"], args.repo)

    l2 = [m["losses"] for m in tp2["metrics"]]
    l1 = tp1["metrics"][0]["losses"]
    print(f"TP=1 ({tp1['dir']}): {[round(x, 6) for x in l1]}")
    for m, losses in zip(tp2["metrics"], l2):
        print(f"TP=2 rank{m['dp_rank']} ({tp2['dir']}): {[round(x, 6) for x in losses]}")

    assert len(l2) == 2, f"expected two TP ranks, got {len(l2)}"

    # The forward all-reduce replicates the region output, so both TP ranks must
    # compute the same loss. Different values here mean rank-local output leaked
    # through.
    for i, (a, b) in enumerate(zip(*l2)):
        assert abs(a - b) <= ATOL + RTOL * abs(b), (
            f"iteration {i}: TP ranks disagree ({a} vs {b}); the forward "
            f"all-reduce did not replicate the region output"
        )

    assert len(l1) == len(l2[0]), "iteration counts differ"
    worst = 0.0
    for i, (sharded, ref) in enumerate(zip(l2[0], l1)):
        err = abs(sharded - ref)
        worst = max(worst, err)
        assert err <= ATOL + RTOL * abs(ref), (
            f"iteration {i}: TP=2 loss {sharded} differs from unsharded TP=1 "
            f"{ref} by {err:.3e}. From iteration 2 onward this also implicates "
            f"the backward all-reduce, since the optimizer has already stepped."
        )

    # Guard against a vacuous pass: with the directive removed and everything
    # else identical, the same comparison must fail loudly. Measured gap is ~1.0,
    # six orders of magnitude above the tolerance.
    if not args.skip_negative_control:
        ctl = _run("tp2_no_collective", [], args.repo)
        ctl_losses = [m["losses"] for m in ctl["metrics"]]
        ctl_err = max(
            abs(a - b) for losses in ctl_losses for a, b in zip(losses, l1)
        )
        print(f"control (no directive, {ctl['dir']}): "
              f"{[round(x, 6) for x in ctl_losses[0]]}  max|diff| vs TP=1 {ctl_err:.3e}")
        assert ctl_err > 100 * (ATOL + RTOL), (
            f"removing the shard_tensor directive changed the loss by only "
            f"{ctl_err:.3e}, so this comparison would pass with no communication "
            f"at all and proves nothing"
        )

    print(f"\nPASS  TP=2 matches TP=1 across {len(l1)} iterations "
          f"(worst |diff| {worst:.2e}), both TP ranks agree, and removing the "
          f"directive breaks it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
