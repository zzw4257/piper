"""What does calling torch.autograd.backward cost, before it does any work?

F55/F56 traced Piper's per-iteration cost down to `compute.backward`, which is
`torch.autograd.backward(outputs, grads)` and nothing else, at ~822 us per
segment. Piper calls it once per segment where an unsegmented model calls it
once per step, so if the engine has a large per-call fixed cost that is the
whole segmentation tax (F54) with no further explanation needed.

Measures the intercept and slope of `backward` against graph length, then the
thing Piper actually does: one backward over a chain of N ops versus K
backwards over N/K-op chains, same total work.

Interleaved with the spread reported, per F54.
"""
import argparse
import statistics
import sys
import time

import torch


def _interleaved(variants, rounds=30, warmup=5):
    for _ in range(warmup):
        for _, fn in variants:
            fn()
    torch.cuda.synchronize()
    out = {n: [] for n, _ in variants}
    for _ in range(rounds):
        for name, fn in variants:
            torch.cuda.synchronize()
            t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
            torch.cuda.synchronize()
            out[name].append((t1 - t0) * 1e6)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numel", type=int, default=1 << 20, help="tensor size (small: launch-bound)")
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args(argv)
    dev = torch.device("cuda", 0)
    print(f"{torch.cuda.get_device_name(0)}  numel={a.numel}")

    def chain(n_ops):
        """Build an n_ops-long chain and return a closure that backwards it."""
        def run():
            x = torch.randn(a.numel, device=dev, requires_grad=True)
            y = x
            for _ in range(n_ops):
                y = y * 1.0001
            torch.autograd.backward([y], [torch.ones_like(y)])
        return run

    lens = [1, 2, 4, 8, 16, 32]
    res = _interleaved([(f"chain{n}", chain(n)) for n in lens], rounds=a.rounds)
    print(f"\n{'ops':>5}{'min us':>9}{'median':>9}{'spread':>9}")
    pts = []
    for n in lens:
        v = sorted(res[f"chain{n}"])
        print(f"{n:>5}{v[0]:>9.0f}{statistics.median(v):>9.0f}{statistics.median(v)-v[0]:>9.0f}")
        pts.append((n, v[0]))
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    a0 = my - b * mx
    print(f"\nbackward(n ops) = {a0:.0f} us + {b:.1f} us/op   "
          f"-> the engine's per-call fixed cost is {a0:.0f} us")

    # The segmentation question, with no model math in the way: the same 32 ops
    # backwarded in one call versus in K calls of 32/K.
    def split(k):
        n = 32 // k
        def run():
            for _ in range(k):
                x = torch.randn(a.numel, device=dev, requires_grad=True)
                y = x
                for _ in range(n):
                    y = y * 1.0001
                torch.autograd.backward([y], [torch.ones_like(y)])
        return run

    ks = [1, 2, 4, 8]
    res2 = _interleaved([(f"k{k}", split(k)) for k in ks], rounds=a.rounds)
    print(f"\n{'segments':>9}{'ops each':>10}{'min us':>9}{'vs k=1':>9}{'predicted':>11}")
    base = min(res2["k1"])
    for k in ks:
        v = min(res2[f"k{k}"])
        pred = a0 * k + b * 32
        print(f"{k:>9}{32//k:>10}{v:>9.0f}{v/base:>8.2f}x{pred:>11.0f}")
    print("\npredicted = fixed_cost * segments + slope * total_ops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
