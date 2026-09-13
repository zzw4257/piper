"""How much of a Piper iteration is the GPU doing the model's arithmetic?

The CP overlap sweep found iteration time flat at ~15 ms while the attention
work grew 4x (s_local 256 -> 512), which is the signature of a step that is not
compute-bound. This measures the same arithmetic with no Ray, no DAG and no
dispatch loop -- one process, one GPU, the tensors a single CP rank owns -- so
the ratio to Piper's reported iteration time is the runtime's share.

It is deliberately generous to Piper: the reference runs the *whole* per-rank
forward and backward including the ring-step math, and omits only the
collectives (which F48/F49 price separately at 37-350 us each).

    python experiments/probe_host_bound.py --steps 4 --heads 8 --dim 512 \
        --batch-size 8 --seqs 1024,2048,4096,8192
"""
import argparse
import math
import sys
import time

import torch


def one_rank_step(dim, heads, steps, batch, s_local, dev, dtype=torch.float32):
    """Forward + backward of one CP rank's share, collectives removed."""
    hd = dim // heads
    scale = 1.0 / math.sqrt(hd)
    wq = torch.randn(dim, dim, device=dev, dtype=dtype, requires_grad=True)
    wo = torch.randn(dim, dim, device=dev, dtype=dtype, requires_grad=True)
    x = torch.randn(batch, s_local, dim, device=dev, dtype=dtype)
    k0 = torch.randn(batch, heads, s_local, hd, device=dev, dtype=dtype)
    v0 = torch.randn(batch, heads, s_local, hd, device=dev, dtype=dtype)
    y = torch.randn(batch, s_local, dim, device=dev, dtype=dtype)

    def step():
        q = (x @ wq.T).view(batch, s_local, heads, hd).transpose(1, 2)
        o = torch.zeros_like(q)
        m = torch.full(q.shape[:-1], float("-inf"), dtype=dtype, device=dev)
        l = torch.zeros(q.shape[:-1], dtype=dtype, device=dev)
        k, v = k0, v0
        for _ in range(steps):
            s = torch.matmul(q, k.transpose(-1, -2)) * scale
            m_new = torch.maximum(m, s.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new.unsqueeze(-1))
            l = l * alpha + p.sum(dim=-1)
            o = o * alpha.unsqueeze(-1) + torch.matmul(p, v)
            m = m_new
        o = o / l.unsqueeze(-1)
        out = o.transpose(1, 2).reshape(batch, s_local, dim) @ wo.T
        (out - y).float().pow(2).mean().backward()
        wq.grad = None; wo.grad = None

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    gpu, host = [], []
    for _ in range(10):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a.record(); step(); b.record()
        t1 = time.perf_counter()          # host returns once everything is enqueued
        torch.cuda.synchronize()
        gpu.append(a.elapsed_time(b)); host.append((t1 - t0) * 1000)
    return min(gpu), min(host)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seqs", default="1024,2048,4096,8192")
    ap.add_argument("--piper-ms", default="",
                    help="comma-separated Piper iteration times (ms) in the same order")
    args = ap.parse_args(argv)
    dev = torch.device("cuda", 0)
    print(f"{torch.cuda.get_device_name(0)}  dim={args.dim} heads={args.heads} "
          f"steps={args.steps} batch={args.batch_size}")
    piper = [float(x) for x in args.piper_ms.split(",")] if args.piper_ms else []
    seqs = [int(s) for s in args.seqs.split(",")]
    print(f"{'seq':>7}{'s_local':>9}{'GPU math ms':>13}{'host launch':>13}{'rel work':>10}"
          + (f"{'Piper ms':>11}{'GPU frac':>10}" if piper else ""))
    base = None
    for i, seq in enumerate(seqs):
        s_local = seq // args.steps
        try:
            ms, host_ms = one_rank_step(args.dim, args.heads, args.steps, args.batch_size, s_local, dev)
        except torch.cuda.OutOfMemoryError:
            print(f"{seq:>7}{s_local:>9}{'OOM':>13}")
            torch.cuda.empty_cache(); continue
        base = base or ms
        row = f"{seq:>7}{s_local:>9}{ms:>13.2f}{host_ms:>13.2f}{ms/base:>10.2f}"
        if i < len(piper):
            row += f"{piper[i]:>11.2f}{ms/piper[i]*100:>9.1f}%"
        print(row)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
