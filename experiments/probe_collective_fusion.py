"""Would fusing K TP all-reduces into one actually save (K-1) x fixed cost?

    torchrun --nproc_per_node=2 experiments/probe_collective_fusion.py

F32 measured that a TP all-reduce costs ~220-320us almost regardless of payload
over a 32x range, which implies K separate collectives cost about K times that
while one collective of K times the payload costs about the same as one. If so,
fusing across microbatches is worth (K-1) x ~250us.

Tested outside Piper on purpose: no Ray, no DAG, no dispatch loop, so a null
result cannot be blamed on Piper's bookkeeping, exactly as in F19's CUDA-graph
probe. Also measures the cat/split the real pass would need, since a saving that
the copies eat is not a saving.
"""
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

DIM = 4096
DTYPE = torch.bfloat16
WARMUP, ITERS = 5, 15
BW = 360e9  # bytes/s, measured in F11


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    out = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e6)
    return min(out), statistics.median(out)


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group("nccl")

    if rank == 0:
        print(f"{'K':>3}{'batch':>7}{'per-coll':>10}{'separate':>11}"
              f"{'fused':>10}{'fused+copy':>12}{'speedup':>9}{'bound':>9}")

    for k, batch in ((4, 2048), (8, 2048), (16, 1024), (8, 512)):
        bufs = [torch.randn(batch, DIM, device=dev, dtype=DTYPE) for _ in range(k)]
        flat = torch.empty(k * batch, DIM, device=dev, dtype=DTYPE)

        def separate():
            for b in bufs:
                dist.all_reduce(b)

        def fused_only():
            dist.all_reduce(flat)

        def fused_with_copies():
            torch.cat(bufs, dim=0, out=flat)
            dist.all_reduce(flat)
            for i, b in enumerate(bufs):
                b.copy_(flat[i * batch:(i + 1) * batch])

        sep_min, _ = timeit(separate)
        fus_min, _ = timeit(fused_only)
        cpy_min, _ = timeit(fused_with_copies)
        per_coll = batch * DIM * 2
        bound = k * per_coll / BW * 1e6

        if rank == 0:
            print(f"{k:>3}{batch:>7}{per_coll/2**20:>9.1f}M{sep_min:>11.0f}"
                  f"{fus_min:>10.0f}{cpy_min:>12.0f}"
                  f"{sep_min/cpy_min:>8.2f}x{bound:>9.0f}")

    if rank == 0:
        print("\nseparate = K all_reduces of one payload each")
        print("fused    = one all_reduce of K payloads, no data movement")
        print("fused+copy = cat into a flat buffer, reduce, copy back "
              "(what a real fusion pass must do)")
        print("bound    = total bytes / 360 GB/s")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
