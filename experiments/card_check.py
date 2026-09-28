"""F94: the same SmolVLM vision step (batch 48, fwd + bwd, fp32) on every claimed card, one at a time.

    python experiments/card_check.py --data DIR

Prints each card's median step, SM clock, power, and active throttle reasons while it runs.
"""
import argparse, os, statistics, subprocess, sys, time
sys.argv = [a for a in sys.argv if not a.startswith("--temp-dir") and not a.startswith("/var/tmp/ziweizho-ray")]
import torch
sys.path[:0] = ["examples", ".", "experiments"]
from profile_stages import smolvlm_stages


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=12)
    a, _ = ap.parse_known_args()
    phys = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    for i, pid in enumerate(phys):
        torch.cuda.set_device(i)
        make, _ = smolvlm_stages(a.data)
        fn, ins = make(48)["vision"]
        ins = [x.detach().requires_grad_(x.is_floating_point()) for x in ins]
        def fb():
            out = fn(*ins); out.backward(torch.ones_like(out))
        for _ in range(5): fb()
        q = subprocess.Popen(["nvidia-smi", "-i", pid, "--query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active",
                              "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
        ts = []
        for _ in range(a.n):
            torch.cuda.synchronize(); s = time.perf_counter(); fb(); torch.cuda.synchronize(); ts.append((time.perf_counter() - s) * 1e3)
        q.terminate(); rows = [l.split(", ") for l in q.communicate()[0].strip().splitlines() if l.count(",") == 5]
        clk = [float(r[0]) for r in rows]; pw = [float(r[2]) for r in rows]
        reasons = sorted({r[5] for r in rows})
        print(f"GPU {pid}: step median {statistics.median(ts):7.1f} ms (min {min(ts):7.1f}, max {max(ts):7.1f}) | SM clock {statistics.median(clk):5.0f}/{rows[0][1]} MHz"
              f" | power {statistics.median(pw):4.0f}/{rows[0][3]} W | temp {rows[-1][4]} C | throttle reasons {reasons}", flush=True)
        del make, fn, ins; torch.cuda.empty_cache()
    print("PASS")


if __name__ == "__main__":
    main()
