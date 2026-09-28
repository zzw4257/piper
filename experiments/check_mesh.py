"""GPU check for F82: named mesh axes let TP and DP compose.

    CUDA_VISIBLE_DEVICES=a,b,c,d python experiments/check_mesh.py

Three runs of the TP MLP from the same global weights and data: one GPU on the whole
batch; DP=2 with the batch split between two replicas; and TP=2 x DP=2 on a
[["dp", 2], ["tp", 2]] mesh, the same split, each replica TP-sharded. Piper's DP sums
gradients, so TP x DP must train exactly as DP=2 does (to fp32 rounding), replica by
replica; before any update, the replicas' mean loss must equal one GPU's. Then open PR
#24's startup weight sync: it must corrupt TP without a mesh and change nothing on one.
"""
import glob
import json
import os
import re
import subprocess
import sys

TOL = 1e-5


def run(schedule, extra, repo):
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", "examples/test_tp_mlp.py",
           "--base-schedule", f"examples/base-schedules/{schedule}.json", "--schedule", "custom", *extra]
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        sys.exit(f"{schedule} failed:\n{p.stdout[-2500:]}\n{p.stderr[-2500:]}")
    d = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", p.stdout)[-1])
    return {int(json.load(open(f))["dp_rank"]): json.load(open(f))["losses"]
            for f in glob.glob(f"{d}/tp_metrics_dp*.json")}, os.path.relpath(d, repo)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra = sys.argv[1:] + ["--warmup", "0", "--iters", "5", "--init", "fixed"]
    one, d1 = run("tp1", extra + ["--tp", "1"], repo)
    dp, d2 = run("dp2", extra + ["--tp", "1", "--dp-split", "2"], repo)
    tpdp, d3 = run("tp2dp2", extra + ["--tp", "2", "--dp-split", "2"], repo)
    print(f"one GPU  {d1}: {[round(x, 6) for x in one[0]]}")
    ok = True
    for rep, place in ((0, 0), (1, 2)):   # replica r is place 2r on the [dp, tp] mesh
        a, b = dp[rep], tpdp[place]
        worst = max(abs(x - y) for x, y in zip(a, b))
        ok &= worst <= TOL
        print(f"replica {rep}: DP=2 {d2} {[round(x, 6) for x in a]}")
        print(f"           TPxDP {d3} {[round(x, 6) for x in b]}  max diff {worst:.1e}")
    first = (dp[0][0] + dp[1][0]) / 2
    print(f"step 1: mean of the two replicas {first:.6f}, one GPU {one[0][0]:.6f}, diff {abs(first - one[0][0]):.1e}")
    ok &= abs(first - one[0][0]) <= TOL
    tp_pairs = max(abs(x - y) for x, y in zip(tpdp[0], tpdp[1]))
    print(f"the two TP ranks of replica 0 report the same loss: max diff {tp_pairs:.1e}")

    # PR #24's startup sync broadcasts every parameter over dp_group. Without a mesh that
    # group is the TP group and rank 1's shard is overwritten; on the mesh it is the dp axis.
    tp, _ = run("tp2", extra + ["--tp", "2"], repo)
    tp24, d4 = run("tp2", extra + ["--tp", "2", "--pr24-sync"], repo)
    tpdp24, d5 = run("tp2dp2", extra + ["--tp", "2", "--dp-split", "2", "--pr24-sync"], repo)
    bad = abs(tp24[0][0] - tp[0][0])
    good = max(abs(x - y) for r in (0, 2) for x, y in zip(tpdp24[r], tpdp[r]))
    print(f"PR #24 sync, TP=2 without a mesh {d4}: step 1 {tp24[0][0]:.6f} vs {tp[0][0]:.6f} unsynced (off by {bad:.2f})")
    print(f"PR #24 sync, TP x DP on the mesh {d5}: max diff vs unsynced {good:.1e}")
    ok &= bad > 0.1 and good <= TOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
