"""F88: SmolVLM fine-tuning curves through Piper equal one GPU's over many steps.

    CUDA_VISIBLE_DEVICES=... python experiments/check_vlm_converge.py --run vlm_het3_mb1 --data DIR --iters 200

A new batch of 48 COCO pairs every step (the data cycles), Adam from the released weights.
One placement per call, so each fits a time-boxed job: one GPU (vlm_single_v3_mb1), vision
data-parallel on three GPUs + decoder (vlm_het3_mb1, F84), TP=3 on every layer (vlm_tp3, F87).
Prints the curve as one JSON line; compare the lines of several calls. The control,
vlm_single, is one GPU with the vision tower unchunked: the same math in a different
floating-point order. Arguments after the known ones go to the driver last, so they override.
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys

ARGS = {"vlm_single": ["--vis-chunks", "1"], "vlm_single_v3_mb1": ["--vis-chunks", "3"],
        "vlm_het3_mb1": ["--vis-chunks", "3"], "vlm_tp3": ["--tp", "3"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(ARGS), required=True)
    a, extra = ap.parse_known_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", "examples/test_smolvlm.py",
           "--base-schedule", f"examples/base-schedules/{a.run}.json", "--schedule", "custom",
           "--batch-size", "48", "--stream", "--warmup", "0", *ARGS[a.run], *extra]
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=840)
    if p.returncode:
        sys.exit(f"{a.run} failed\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    d = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", p.stdout)[-1])
    m = json.load(open(sorted(glob.glob(f"{d}/branches_metrics_dp*.json"))[0]))
    print(f"{a.run}: {len(m['losses'])} steps, first {m['losses'][0]:.6f}, last {m['losses'][-1]:.6f}, "
          f"median step {statistics.median(m['iter_times']) * 1e3:.1f} ms")
    print("CURVE " + json.dumps({"run": a.run, "losses": m["losses"], "iter_times": m["iter_times"]}))
    print("PASS")


if __name__ == "__main__":
    main()
