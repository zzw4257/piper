"""GPU check for F84: SmolVLM with the vision tower data-parallel on three GPUs, the decoder on a fourth.

    CUDA_VISIBLE_DEVICES=a,b,c,d python experiments/check_vlm_het.py --data DIR

48 pairs per step, fixed work over m microbatches. For each m: one GPU; the vision
tower in three chunks on GPUs 0-2 sharing its weights (F83) with embedding and decoder on
GPU 3; and, for comparison, the vision tower split into three pipeline stages. Losses
must match one GPU.
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
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        return None, p.stdout[-1500:] + p.stderr[-1500:]
    d = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", p.stdout)[-1])
    m = json.load(open(sorted(glob.glob(f"{d}/branches_metrics_dp*.json"))[0]))
    return {"dir": os.path.relpath(d, repo), "losses": m["losses"], "t": statistics.median(m["iter_times"]),
            "mem": [round(v, 1) for _, v in sorted(m["peak_mem_gb"].items(), key=lambda kv: int(kv[0]))]}, None


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ms = [int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--mb=")] or [1, 2, 4]
    extra = [a for a in sys.argv[1:] if not a.startswith("--mb=")]
    ok = True
    for m in ms:
        b = ["--batch-size", str(48 // m)]
        runs = [(f"vlm_single_v3_mb{m}", ["--vis-stages", "1", "--vis-chunks", "3"])]
        runs += [(f"vlm_het3_mb{m}", ["--vis-stages", "1", "--vis-chunks", "3"])]
        if m > 1:
            runs += [(f"vlm_het3_mb{m}_1f1b", ["--vis-stages", "1", "--vis-chunks", "3"]),
                     (f"vlm_4st_routed_mb{m}_1f1b", ["--vis-stages", "3", "--vis-chunks", "1"])]
        else:
            runs += [(f"vlm_4st_routed_mb{m}", ["--vis-stages", "3", "--vis-chunks", "1"])]
        ref = None
        for name, a in runs:
            r, err = run(name, extra + b + a, repo)
            if r is None:
                ok = False
                print(f"{name:26s} FAILED\n{err}", flush=True)
                continue
            ref = ref or r
            worst = max(abs(x - y) for x, y in zip(r["losses"], ref["losses"]))
            ok &= worst <= TOL
            print(f"m={m} {name:26s} {r['dir']}  vs one GPU {worst:.1e}  step {r['t'] * 1e3:7.1f} ms  "
                  f"x{ref['t'] / r['t']:.2f}  peak GB {r['mem']}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
