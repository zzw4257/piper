"""F94: does load elsewhere on the host slow one GPU's eager SmolVLM vision tower? (2 claimed cards)

    python experiments/interference_probe.py --data DIR

Card A times the vision tower (batch 48, forward + backward) in phases: the other card idle; the
other card running bf16 matmuls; idle again; host CPUs busy (16 spinning processes). Card A's SM
clock and power are sampled during each phase.
"""
import argparse, multiprocessing as mp, os, statistics, subprocess, sys, time
sys.argv = [a for a in sys.argv if not a.startswith("--temp-dir") and not a.startswith("/var/tmp/ziweizho-ray")]
import torch
sys.path[:0] = ["examples", ".", "experiments"]
from profile_stages import smolvlm_stages


def burner(stop_at):
    torch.cuda.set_device(1)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    while time.time() < stop_at:
        for _ in range(50):
            a = (a @ a).clamp_(-1, 1)
        torch.cuda.synchronize()


def spinner(stop_at):
    x = 0
    while time.time() < stop_at:
        x += 1


def sample(phys, dur):
    out = subprocess.run(["nvidia-smi", "-i", phys, "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits",
                          "-lms", "250"], capture_output=True, text=True, timeout=dur + 5) if False else None
    return out


def phase(name, fn, n, phys):
    q = subprocess.Popen(["nvidia-smi", "-i", phys, "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                          "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - s) * 1e3)
    q.terminate(); lines = [l.split(",") for l in q.communicate()[0].strip().splitlines() if l.count(",") == 2]
    clk = [float(a) for a, _, _ in lines]; pw = [float(b) for _, b, _ in lines]; tp = [float(c) for _, _, c in lines]
    print(f"{name:28s} step median {statistics.median(ts):7.1f} ms  min {min(ts):7.1f}  | card A SM clock median {statistics.median(clk):6.0f} MHz"
          f" (min {min(clk):5.0f})  power {statistics.median(pw):5.0f} W  temp {max(tp):3.0f} C", flush=True)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=15)
    a, _ = ap.parse_known_args()
    phys = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    torch.cuda.set_device(0)
    make, _ = smolvlm_stages(a.data)
    fn, ins = make(48)["vision"]
    ins = [x.detach().requires_grad_(x.is_floating_point()) for x in ins]
    def fb():
        out = fn(*ins); out.backward(torch.ones_like(out))
    for _ in range(5): fb()
    ctx = mp.get_context("spawn")
    r = {}
    r["idle"] = phase("other card idle", fb, a.n, phys[0])
    p = ctx.Process(target=burner, args=(time.time() + 60,)); p.start(); time.sleep(8)
    r["gpu"] = phase("other card: bf16 matmuls", fb, a.n, phys[0]); p.terminate(); p.join()
    time.sleep(5)
    r["idle2"] = phase("other card idle again", fb, a.n, phys[0])
    ps = [ctx.Process(target=spinner, args=(time.time() + 60,)) for _ in range(16)]
    [x.start() for x in ps]; time.sleep(2)
    r["cpu"] = phase("host: 16 busy CPU processes", fb, a.n, phys[0]); [x.terminate() for x in ps]
    print(f"slowdown: other GPU busy {r['gpu'] / r['idle']:.2f}x, host CPUs busy {r['cpu'] / r['idle']:.2f}x; cpu count {os.cpu_count()}")
    print("PASS")


if __name__ == "__main__":
    main()
