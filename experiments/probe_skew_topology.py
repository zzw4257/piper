"""P3 (notes/cp-design.md §7): is the per-collective arrival-skew tax a property
of *global* synchronization or of NCCL itself?

    torchrun --nproc_per_node=4 experiments/probe_skew_topology.py

Inject a known delay on one rank before each collective and read, on every other
rank, how long the collective took. An all-reduce must wait for the slowest rank,
so every rank's duration should track the injected skew. A ring exchange couples
a rank only to its two neighbours: the skewed rank's neighbours should pay, the
rank opposite it should not -- on that step. With 2 ranks the two topologies are
the same graph, so this needs >= 3; the skewed rank is chosen non-adjacent to
rank 0 when the world allows.

Output is per-(collective, skew) median and min of rank-local durations in us,
plus median/min as the contamination check from log F36.
"""
import os
import statistics
import sys

import torch
import torch.distributed as dist

PAYLOAD_MIB = float(os.environ.get("PAYLOAD_MIB", "16"))
ITERS = int(os.environ.get("ITERS", "40"))
SKEWS_US = (0, 100, 500, 2000)


def _sleep_us(us: float) -> None:
    # cycles at ~1.9 GHz; a busy-wait kernel so the skew sits on the GPU stream.
    torch.cuda._sleep(int(us * 1900))


def _ring(t, buf):
    n, me = dist.get_world_size(), dist.get_rank()
    ops = [dist.P2POp(dist.isend, t, (me + 1) % n), dist.P2POp(dist.irecv, buf, (me - 1) % n)]
    for w in dist.batch_isend_irecv(ops):
        w.wait()


def _time(fn, skew_us, skewed_rank):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dist.barrier()
    torch.cuda.synchronize()
    if dist.get_rank() == skewed_rank and skew_us:
        _sleep_us(skew_us)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    n = dist.get_world_size()
    skewed = n // 2  # non-adjacent to rank 0 for n >= 4
    numel = int(PAYLOAD_MIB * 2**20 / 4)
    t = torch.ones(numel, device="cuda")
    buf = torch.empty_like(t)
    arms = {"all_reduce": lambda: dist.all_reduce(t), "ring": lambda: _ring(t, buf)}

    for _ in range(5):
        for fn in arms.values():
            _time(fn, 0, skewed)

    rows = []
    for name, fn in arms.items():
        for skew in SKEWS_US:
            d = sorted(_time(fn, skew, skewed) for _ in range(ITERS))
            med, mn = statistics.median(d), d[0]
            rows.append((name, skew, rank, med, mn))
    gathered = [None] * n
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        print(f"world={n} skewed_rank={skewed} payload={PAYLOAD_MIB}MiB iters={ITERS}")
        print(f"{'coll':<10}{'skew_us':>8}{'rank':>5}{'median_us':>11}{'min_us':>9}{'med/min':>8}  role")
        for rows_ in gathered:
            for name, skew, r, med, mn in rows_:
                role = "SKEWED" if r == skewed else ("neighbour" if (r - skewed) % n in (1, n - 1) else "opposite")
                print(f"{name:<10}{skew:>8}{r:>5}{med:>11.1f}{mn:>9.1f}{med / mn:>8.2f}  {role}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
