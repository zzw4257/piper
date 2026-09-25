"""GPU check for F71: routed branches train like threaded ones and run concurrently.

    CUDA_VISIBLE_DEVICES=a,b,c python experiments/check_branches.py

Four runs of the two-branch model from the same weights and data: one GPU
(threaded, the validated path), one GPU routed, three GPUs threaded, three GPUs
routed. Losses must agree to fp32 noise. On three GPUs, routing lets the two
encoders run at once, so its step should be close to max(encoders) + decoder
instead of their sum.
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

TOL = 1e-4  # relative (absolute below 1)


TEST = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--test=")), "examples/test_branches.py")


def run(schedule, extra, repo):
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", TEST,
           "--base-schedule", f"examples/base-schedules/{schedule}.json", "--schedule", "custom", *extra]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        sys.exit(f"{schedule} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    run_dir = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", proc.stdout)[-1])
    m = [json.load(open(f)) for f in glob.glob(f"{run_dir}/branches_metrics_dp*.json")]
    return {"dir": os.path.relpath(run_dir, repo), "losses": m[0]["losses"], "times": m[0]["iter_times"]}


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra = [a for a in sys.argv[1:] if not a.startswith(("--runs=", "--test="))]
    names = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--runs=")),
                 ["br_single", "br_single_routed", "br_threaded", "br_routed"])
    runs = {name: run(name, extra, repo) for name in names}
    ref = runs[names[0]]["losses"]
    ok = True
    for name, r in runs.items():
        worst = max(abs(a - b) / max(abs(b), 1.0) for a, b in zip(r["losses"], ref))
        ok &= worst <= TOL
        print(f"{name:18s} {r['dir']}  losses {[round(x, 6) for x in r['losses']]}  "
              f"vs one GPU {worst:.1e}  median step {statistics.median(r['times']) * 1e3:.1f} ms")
    if "br_threaded" in runs and "br_routed" in runs:
        t_thr = statistics.median(runs["br_threaded"]["times"])
        t_rt = statistics.median(runs["br_routed"]["times"])
        print(f"three GPUs: threaded {t_thr * 1e3:.1f} ms, routed {t_rt * 1e3:.1f} ms, ratio {t_thr / t_rt:.2f}x")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
