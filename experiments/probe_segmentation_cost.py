"""What does Piper's segmented execution cost, in plain PyTorch, with no Piper?

F52 measured the same arithmetic at 2.4 ms of host launch time written straight
and 11.05 ms inside Piper, and put 82% of the difference in the compute nodes.
The obvious check -- turn on Inductor -- fails on both hosts for unrelated
toolchain reasons, so this decomposes the gap directly instead: the same ring
arithmetic, three ways, no Ray and no DAG.

  A straight      one function, one backward over the whole thing
  B detached      a detach/clone boundary between ring steps, autograd called
                  once per segment with explicit grad_outputs -- what Piper does
  C detached+gm   B, with each segment run as a codegen'd fx.GraphModule --
                  which is what Piper actually does (actor.py recompile()s the
                  module and calls it; it does NOT use fx.Interpreter)
  D +marshalling  C, plus the name-keyed argument marshalling Piper's
                  _bucket_forward_runner does on every call

Host launch time is the quantity of interest: at these sizes the GPU is never
the bottleneck (F50/F52).
"""
import argparse
import math
import sys
import time

import torch
import torch.fx as fx


def _step_fn(scale):
    def f(q, k, v, o, m, l):
        s = torch.matmul(q, k.transpose(-1, -2)) * scale
        m_new = torch.maximum(m, s.amax(dim=-1))
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new.unsqueeze(-1))
        l2 = l * alpha + p.sum(dim=-1)
        o2 = o * alpha.unsqueeze(-1) + torch.matmul(p, v)
        return o2, m_new, l2
    return f


def _interleaved(variants, rounds=25, warmup=3):
    """Round-robin the variants instead of running each in a block.

    Host timing on a shared box drifts by more than the differences being
    measured (a block-structured first attempt put B above C, which is
    impossible by construction). Interleaving is the F11/F36 rule applied to
    host time: every variant sees the same drift, and the minimum of each is
    comparable.
    """
    for _ in range(warmup):
        for _, fn in variants:
            fn()
    torch.cuda.synchronize()
    host = {name: [] for name, _ in variants}
    gpu = {name: [] for name, _ in variants}
    for _ in range(rounds):
        for name, fn in variants:
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter(); a.record(); fn(); b.record(); t1 = time.perf_counter()
            torch.cuda.synchronize()
            host[name].append((t1 - t0) * 1000); gpu[name].append(a.elapsed_time(b))
    return host, gpu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--s-local", type=int, default=256)
    a = ap.parse_args(argv)
    dev = torch.device("cuda", 0)
    B, H, S, D, N = a.batch_size, a.heads, a.s_local, a.head_dim, a.steps
    scale = 1.0 / math.sqrt(D)
    step = _step_fn(scale)
    mk = lambda: torch.randn(B, H, S, D, device=dev)
    q0, k0, v0 = mk(), mk(), mk()
    print(f"{torch.cuda.get_device_name(0)}  steps={N} b={B} h={H} s_local={S} d={D}")

    def straight():
        q = q0.clone().requires_grad_(True)
        o = torch.zeros_like(q)
        m = torch.full(q.shape[:-1], float("-inf"), device=dev)
        l = torch.zeros(q.shape[:-1], device=dev)
        for _ in range(N):
            o, m, l = step(q, k0, v0, o, m, l)
        (o / l.unsqueeze(-1)).pow(2).sum().backward()

    def detached(runner=None):
        def run():
            q = q0.clone().requires_grad_(True)
            o = torch.zeros_like(q)
            m = torch.full(q.shape[:-1], float("-inf"), device=dev)
            l = torch.zeros(q.shape[:-1], device=dev)
            saved = []
            for _ in range(N):
                ins = [t.detach().clone().requires_grad_(True) for t in (q, o, m, l)]
                qi, oi, mi, li = ins
                if runner is None:
                    outs = step(qi, k0, v0, oi, mi, li)
                else:
                    outs = runner(qi, k0, v0, oi, mi, li)
                saved.append((ins, outs))
                q, (o, m, l) = qi, outs
            o, m, l = saved[-1][1]
            loss = (o / l.unsqueeze(-1)).pow(2).sum()
            g = torch.autograd.grad(loss, list(saved[-1][1]),
                                    allow_unused=True, retain_graph=True)
            for ins, outs in reversed(saved):
                live = [(x, gi) for x, gi in zip(outs, g) if gi is not None]
                if not live:
                    break
                g = torch.autograd.grad([x for x, _ in live], ins,
                                        grad_outputs=[gi for _, gi in live],
                                        allow_unused=True, retain_graph=True)
                g = g[1:]   # q's grad is a leaf of this segment; carry o, m, l
        return run

    gm = fx.symbolic_trace(step)
    gm.to(dev)
    gm.recompile()          # exactly as actor.py does

    names = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
    def marshalled(*vals):
        # _bucket_forward_runner rebuilds the call list by placeholder name on
        # every call; mirrored here to price it.
        dynamic = {n: v for n, v in zip(names, vals)}
        call_args = [dynamic[n] if n in dynamic else None for n in names]
        return gm(*call_args)

    rows = [("A straight", straight),
            ("B detached", detached()),
            ("C detached+gm", detached(gm)),
            ("D +marshalling", detached(marshalled))]
    host, gpu = _interleaved(rows)
    base = min(host["A straight"])
    print(f"{'variant':<17}{'host min':>10}{'p25':>8}{'median':>8}{'gpu min':>9}{'vs A':>8}{'sep?':>7}")
    prev = None
    for name, _ in rows:
        h = sorted(host[name]); g = min(gpu[name])
        p25 = h[len(h) // 4]; med = h[len(h) // 2]
        sep = "" if prev is None else ("yes" if h[0] > prev[-1] * 1.0 and h[0] - prev[0] > (h[len(h)//2] - h[0]) else "no")
        print(f"{name:<17}{h[0]:>10.2f}{p25:>8.2f}{med:>8.2f}{g:>9.2f}{h[0]/base:>7.2f}x{sep:>7}")
        prev = h
    spread = {n: (sorted(host[n])[len(host[n])//2] - min(host[n])) for n, _ in rows}
    print("\nmin-to-median spread (ms): " + ", ".join(f"{n.split()[0]}={v:.2f}" for n, v in spread.items()))
    print("A difference is only readable if it exceeds that spread.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
