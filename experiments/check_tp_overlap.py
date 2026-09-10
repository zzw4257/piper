"""Is TP communication hidden behind compute? Only with more than one microbatch.

    CUDA_VISIBLE_DEVICES=<two idle gpus> python experiments/check_tp_overlap.py

Runs the same model at one and four microbatches and compares, per iteration,
the summed GPU kernel duration against the wall span of the GPU timeline.
Summed durations do not shrink when work overlaps, so sum/span > 1 is direct
evidence that two streams ran concurrently.

The prediction being tested comes from the lowered DAG, not from a measurement:
with one microbatch the serial order is
    s0.seg1 -> tp_all_reduce.0 -> s0.seg2
so the collective has nothing to overlap with and `shard_tensor`'s `stream`
field is inert. With four microbatches, tp_all_reduce.0 (MB 0) and
s0.seg1.splitMB1 (MB 1) are independent, so the collective can hide.

Needs the batch large enough that GPU work dominates Piper's per-node Python
dispatch; at batch 1024, dim 8192 the timeline is ~20% idle and the effect is
muddier (still visible, but noisier).
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_tp import _node_uid, _split_iterations, _GPU_CATS  # noqa: E402


def _run(schedule: str, args) -> str:
    cmd = [
        sys.executable, "examples/test_harness.py",
        "--test-file", "examples/test_tp_mlp.py",
        "--base-schedule", f"examples/base-schedules/{schedule}.json",
        "--schedule", "custom",
        "--dim", str(args.dim), "--hidden", str(args.hidden),
        "--batch-size", str(args.batch_size), "--dtype", "bf16", "--init", "random",
        "--warmup", "5", "--iters", "3",
        "--pytorch-profiler", "--pytorch-profiler-iters", str(args.iters),
    ]
    proc = subprocess.run(cmd, cwd=args.repo, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        sys.exit(f"{schedule} failed:\n{proc.stdout[-2500:]}\n{proc.stderr[-2500:]}")
    dirs = re.findall(r"out/\d{8}_\d{6}", proc.stdout)
    if not dirs:
        sys.exit(f"{schedule}: no run directory in harness output")
    return os.path.join(args.repo, dirs[-1])


def _concurrency(run_dir: str, n_iters: int) -> list[float]:
    traces = glob.glob(os.path.join(run_dir, "pytorch_profile_*dprank1.json"))
    if not traces:
        sys.exit(f"no rank-1 trace under {run_dir}")
    with open(traces[0], encoding="utf-8") as f:
        raw = json.load(f).get("traceEvents", [])
    events = [
        e for e in raw
        if e.get("cat") in _GPU_CATS and e.get("dur") and e.get("name")
        and _node_uid(e["name"]) is not None
    ]
    events.sort(key=lambda e: e["ts"])
    which = _split_iterations(events, n_iters, 500.0)

    sums: dict[int, float] = {}
    spans: dict[int, tuple[float, float]] = {}
    for e, it in zip(events, which):
        sums[it] = sums.get(it, 0.0) + e["dur"]
        lo, hi = spans.get(it, (e["ts"], e["ts"] + e["dur"]))
        spans[it] = (min(lo, e["ts"]), max(hi, e["ts"] + e["dur"]))
    return [
        sums[i] / (spans[i][1] - spans[i][0])
        for i in sorted(sums) if spans[i][1] > spans[i][0]
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--dim", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args(argv)

    out = {}
    for schedule in ("tp2_mb1", "tp2_mb4"):
        c = _concurrency(_run(schedule, args), args.iters)
        out[schedule] = c
        print(f"{schedule:<10s} concurrency (sum/span) over {len(c)} iters: "
              f"min {min(c):.2f}x  median {statistics.median(c):.2f}x  max {max(c):.2f}x")

    mb1, mb4 = out["tp2_mb1"], out["tp2_mb4"]

    assert max(mb1) <= 1.02, (
        f"one microbatch reached concurrency {max(mb1):.2f}x, but the lowered DAG "
        f"puts the collective strictly between two compute nodes, so nothing "
        f"should overlap. Either the DAG changed or the metric is wrong."
    )
    assert statistics.median(mb4) > 1.02, (
        f"four microbatches gave median concurrency {statistics.median(mb4):.2f}x. "
        f"TP communication is not being hidden. If the GPU timeline is idle "
        f"(concurrency well below 1 even at mb4), the batch is too small for GPU "
        f"work to dominate Piper's per-node dispatch -- raise --batch-size."
    )
    print(f"\nPASS  one microbatch cannot hide the collective (max {max(mb1):.2f}x); "
          f"four microbatches do (median {statistics.median(mb4):.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
