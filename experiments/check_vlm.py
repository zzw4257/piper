"""GPU check for F77: SmolVLM placements at fixed work (batch 32 over m microbatches).

    CUDA_VISIBLE_DEVICES=a,b,c,d python experiments/check_vlm.py --mb=4 --data DIR

For one m: one GPU, then 2 stages (vision | embedding+decoder) and 4 stages (vision in
three | embedding+decoder), threaded and routed, GPipe and (m > 1) 1F1B. Losses must
match the one-GPU run; prints median step and per-rank peak memory.
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
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        return None, proc.stdout[-1500:] + proc.stderr[-1500:]
    d = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", proc.stdout)[-1])
    m = json.load(open(sorted(glob.glob(f"{d}/branches_metrics_dp*.json"))[0]))
    return {"dir": os.path.relpath(d, repo), "losses": m["losses"], "t": statistics.median(m["iter_times"]),
            "mem": [round(v, 1) for _, v in sorted(m["peak_mem_gb"].items(), key=lambda kv: int(kv[0]))]}, None


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--mb=")), "1"))
    extra = [a for a in sys.argv[1:] if not a.startswith("--mb=")] + ["--batch-size", str(32 // m)]
    ngpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    names = [(f"vlm_single_v3_mb{m}", 3)]
    for lay, v, need in (("2st", 1, 2), ("4st", 3, 4)):
        if need > ngpu:
            continue
        for kind in ("threaded", "routed"):
            names += [(f"vlm_{lay}_{kind}_mb{m}", v)] + ([(f"vlm_{lay}_{kind}_mb{m}_1f1b", v)] if m > 1 else [])
    ref, ok = None, True
    for name, v in names:
        r, err = run(name, extra + ["--vis-stages", str(v)], repo)
        if r is None:
            ok = False
            print(f"{name:28s} FAILED\n{err}")
            continue
        ref = ref or r
        worst = max(abs(a - b) for a, b in zip(r["losses"], ref["losses"]))
        ok &= worst <= TOL
        print(f"{name:28s} {r['dir']}  vs one GPU {worst:.1e}  median step {r['t'] * 1e3:7.1f} ms  "
              f"x{ref['t'] / r['t']:.2f}  peak GB {r['mem']}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
