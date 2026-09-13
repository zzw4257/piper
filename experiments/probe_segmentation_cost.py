"""What does Piper's segmented execution cost, in plain PyTorch, with no Piper?

F52 measured the same arithmetic at 2.4 ms of host launch time written straight
and 11.05 ms inside Piper, and put 82% of the difference in the compute nodes.
The obvious check -- turn on Inductor -- fails on both hosts for unrelated
toolchain reasons, so this decomposes the gap directly instead: the same ring
arithmetic, three ways, no Ray and no DAG.

  A straight      one function, one backward over the whole thing
  B detached      a detach/clone boundary between ring steps, autograd called
                  once per segment with explicit grad_outputs -- what Piper does
  C detached+fx   B, with each segment run through torch.fx.Interpreter, which
                  is how Piper executes a segment's GraphModule

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


def _time(fn, reps=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    host, gpu = [], []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); a.record(); fn(); b.record(); t1 = time.perf_counter()
        torch.cuda.synchronize()
        host.append((t1 - t0) * 1000); gpu.append(a.elapsed_time(b))
    return min(host), min(gpu)


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

    def detached(interpreter=None):
        def run():
            q = q0.clone().requires_grad_(True)
            o = torch.zeros_like(q)
            m = torch.full(q.shape[:-1], float("-inf"), device=dev)
            l = torch.zeros(q.shape[:-1], device=dev)
            saved = []
            for _ in range(N):
                ins = [t.detach().clone().requires_grad_(True) for t in (q, o, m, l)]
                qi, oi, mi, li = ins
                if interpreter is None:
                    outs = step(qi, k0, v0, oi, mi, li)
                else:
                    outs = interpreter.run(qi, k0, v0, oi, mi, li)
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
    interp = fx.Interpreter(gm)

    rows = [("A straight", straight),
            ("B detached", detached()),
            ("C detached+fx", detached(interp))]
    print(f"{'variant':<16}{'host ms':>10}{'gpu ms':>9}{'host vs A':>11}")
    base = None
    for name, fn in rows:
        try:
            h, g = _time(fn)
        except Exception as e:
            print(f"{name:<16}{'FAILED':>10}  {type(e).__name__}: {str(e)[:70]}")
            continue
        base = base or h
        print(f"{name:<16}{h:>10.2f}{g:>9.2f}{h/base:>10.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
