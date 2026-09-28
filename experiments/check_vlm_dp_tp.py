"""GPU check for F87: TP composes with DP on SmolVLM. Mesh dp=2 x tp=3 must match dp=2 alone.

    CUDA_VISIBLE_DEVICES=a,b,c,d,e,f python experiments/check_vlm_dp_tp.py --data DIR

Both runs give replica d rows d, d+2, ... of one batch of 32, so the per-replica losses
must agree replica for replica over 5 Adam steps.
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

TOL = 1e-4


def run(schedule, extra, repo):
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", "examples/test_smolvlm.py",
           "--base-schedule", f"examples/base-schedules/{schedule}.json", "--schedule", "custom", *extra]
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=800)
    if p.returncode:
        sys.exit(f"{schedule} failed\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    d = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", p.stdout)[-1])
    ms = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/branches_metrics_dp*.json"))]
    return ms


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra = sys.argv[1:] + ["--batch-size", "32", "--dp-split", "2"]
    a = run("vlm_dp2", extra, repo)
    b = run("vlm_dp2_tp3", extra + ["--tp", "3"], repo)
    # one metrics file per driver; on the mesh, drivers 0-2 are replica 0 and 3-5 replica 1
    la = {m["dp_rank"]: m["losses"] for m in a}
    lb = {m["dp_rank"]: m["losses"] for m in b}
    worst = 0.0
    for place, losses in sorted(lb.items()):
        ref = la[place // 3]
        worst = max(worst, max(abs(x - y) for x, y in zip(losses, ref)))
        print(f"dp2_tp3 place {place} (replica {place // 3}) {[round(x, 6) for x in losses]}")
    for r, losses in sorted(la.items()):
        print(f"dp2     replica {r}           {[round(x, 6) for x in losses]}")
    ta = max(statistics.median(m["iter_times"]) for m in a) * 1e3
    tb = max(statistics.median(m["iter_times"]) for m in b) * 1e3
    print(f"median step dp2 {ta:.1f} ms, dp2_tp3 {tb:.1f} ms; worst |diff| {worst:.1e}")
    print("PASS" if worst <= TOL else "FAIL")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    sys.exit(main())
