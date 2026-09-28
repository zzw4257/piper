"""Ring vs hoisted ring vs one all-gather for CP=4 (log F86): iteration time and peak memory.

    python experiments/sweep_cp_decompose.py --seqs 8192 32768 -- --temp-dir /tmp/ray

Interleaved arms, repeated; reports the minimum iteration time of the slowest rank.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

ARMS = ["cp4_ring_dp", "cp4_ring_dp_hoist", "cp4_gather_dp"]


def run(repo, sched, seq, a, extra):
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", "examples/test_ring_attn.py",
           "--base-schedule", f"examples/base-schedules/{sched}.json", "--schedule", "custom",
           "--steps", "4", "--seq", str(seq), "--dim", str(a.dim), "--heads", str(a.heads),
           "--batch-size", str(a.batch), "--warmup", "2", "--iters", str(a.iters), *extra]
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=600)
    d = re.findall(r"out/\d{8}_\d{6}", p.stdout)
    if p.returncode or not d:
        return None
    ms = [json.load(open(f)) for f in glob.glob(f"{repo}/{d[-1]}/cp_metrics_dp*.json")]
    return (max(min(m["iter_times_s"]) for m in ms) * 1e3,
            max(max(m["peak_memory_by_rank"].values()) for m in ms) / 2**20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for seq in a.seqs:
        best = {}
        for _ in range(a.reps):
            for s in ARMS:
                r = run(repo, s, seq, a, extra)
                if r and (s not in best or r[0] < best[s][0]):
                    best[s] = r
        for s in ARMS:
            t, m = best.get(s, (float("nan"), float("nan")))
            print(f"seq={seq} {s:20s} min_iter={t:8.2f} ms peak={m:8.0f} MiB", flush=True)
    print("PASS")


if __name__ == "__main__":
    main()
