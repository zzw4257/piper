"""Can a Piper-shaped TP step be captured in a CUDA graph, and does it help?

    torchrun --nproc_per_node=2 experiments/probe_cudagraph.py

F18 left one question: the cost of a TP step tracks the *number* of collectives,
not their bytes (payload fell 8x while communication rose 26x). The leading
hypothesis is that each rank runs its own Python dispatch loop, so every
collective re-synchronizes two independently-drifting streams of kernel launches.
A CUDA graph replaces the whole step with one launch, which would remove that
drift entirely -- so if the hypothesis is right, capture should collapse
per-collective cost toward payload/bandwidth.

Deliberately independent of Piper: it reproduces the *shape* of a Piper TP step
(compute on the default stream, all-reduce on a second stream, joined by events)
without Ray, the DAG executor, or any of Piper's per-step Python work. If capture
does not help even here, the hypothesis is wrong and changing Piper would be
wasted. Staged so a failure localizes:

  1 compute + autograd          -- is autograd capturable at all
  2 + all-reduce, one stream    -- is NCCL capturable
  3 + second stream and events  -- is Piper's shape capturable
  4 timing, eager vs replay     -- does it actually help
"""
import datetime
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

DIM = 4096
HIDDEN_LOCAL = 8192      # per rank, i.e. hidden 16384 at tp=2
BATCH = 512
DTYPE = torch.bfloat16
N_COLLECTIVES = 16       # matches F18's mb=4 x 2 stages x 2 passes
WARMUP, ITERS = 5, 12


def _log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def _payload_us(bytes_per_coll: int, gbps: float = 360.0) -> float:
    return bytes_per_coll / (gbps * 1e9) * 1e6


def stage1_compute_only(rank, dev):
    """Is autograd capturable?"""
    w = torch.randn(HIDDEN_LOCAL, DIM, device=dev, dtype=DTYPE, requires_grad=True)
    x = torch.randn(BATCH, DIM, device=dev, dtype=DTYPE)

    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            w.grad = None
            F.linear(x, w).float().pow(2).sum().backward()
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    w.grad = None
    with torch.cuda.graph(g):
        F.linear(x, w).float().pow(2).sum().backward()
    g.replay()
    torch.cuda.synchronize()
    return "captured compute + autograd"


def stage2_single_stream(rank, dev, pool):
    """Is NCCL capturable?"""
    t = torch.randn(BATCH, DIM, device=dev, dtype=DTYPE)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            dist.all_reduce(t)
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        for _ in range(2):
            dist.all_reduce(t)
    g.replay()
    torch.cuda.synchronize()
    return "captured NCCL all-reduce on one stream"


def _step(x, w_up, w_down, comm_stream, n_coll):
    """One Piper-shaped step: compute on the current stream, collectives on another."""
    cur = torch.cuda.current_stream()
    out = None
    for _ in range(n_coll):
        h = F.linear(F.gelu(F.linear(x, w_up)), w_down)
        # hand the boundary tensor to the comm stream, reduce, hand it back --
        # this is what TP_COMM does in Piper's executor
        comm_stream.wait_stream(cur)
        with torch.cuda.stream(comm_stream):
            h = h.contiguous()
            dist.all_reduce(h)
        cur.wait_stream(comm_stream)
        out = h if out is None else out + h
    return out


def stage3_two_streams(rank, dev, pool, n_coll):
    """Is Piper's two-stream shape capturable?"""
    w_up = torch.randn(HIDDEN_LOCAL, DIM, device=dev, dtype=DTYPE)
    w_down = torch.randn(DIM, HIDDEN_LOCAL, device=dev, dtype=DTYPE)
    x = torch.randn(BATCH, DIM, device=dev, dtype=DTYPE)
    comm = torch.cuda.Stream(device=dev)

    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _step(x, w_up, w_down, comm, n_coll)
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        _step(x, w_up, w_down, comm, n_coll)
    g.replay()
    torch.cuda.synchronize()
    return g, (x, w_up, w_down, comm)


def stage4_timing(rank, dev, g, args, n_coll):
    """Eager dispatch against graph replay, same work."""
    x, w_up, w_down, comm = args

    def timeit(fn, label):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        return min(samples), statistics.median(samples)

    eager_min, eager_med = timeit(
        lambda: _step(x, w_up, w_down, comm, n_coll), "eager")
    graph_min, graph_med = timeit(g.replay, "graph")
    return eager_min, eager_med, graph_min, graph_med


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", rank % torch.cuda.device_count())
    print(f"[rank {rank}] cuda ready on {dev}, joining process group", flush=True)
    # No device_id: eager NCCL init hung for 10 minutes on this shared host.
    # A bounded timeout so a capture deadlock reports instead of hanging.
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=180))
    print(f"[rank {rank}] process group joined", flush=True)

    _log(rank, f"world={world} device={torch.cuda.get_device_name(dev)} "
               f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}")
    _log(rank, f"batch={BATCH} dim={DIM} hidden_local={HIDDEN_LOCAL} "
               f"collectives/step={N_COLLECTIVES} dtype={DTYPE}\n")

    try:
        _log(rank, "stage 1:", stage1_compute_only(rank, dev))
    except Exception as e:
        _log(rank, f"stage 1 FAILED: {type(e).__name__}: {str(e)[:200]}")
        return 1

    pool = torch.cuda.graph_pool_handle()
    try:
        _log(rank, "stage 2:", stage2_single_stream(rank, dev, pool))
    except Exception as e:
        _log(rank, f"stage 2 FAILED: {type(e).__name__}: {str(e)[:300]}")
        _log(rank, "  -> NCCL is not capturable here; a full-step graph is out, "
                   "but capturing the compute segments alone is still possible")
        return 1

    try:
        g, args = stage3_two_streams(rank, dev, pool, N_COLLECTIVES)
        _log(rank, "stage 3: captured two-stream step with events")
    except Exception as e:
        _log(rank, f"stage 3 FAILED: {type(e).__name__}: {str(e)[:300]}")
        return 1

    e_min, e_med, g_min, g_med = stage4_timing(rank, dev, g, args, N_COLLECTIVES)
    payload = BATCH * DIM * 2  # bf16
    ideal = _payload_us(payload) * N_COLLECTIVES
    _log(rank, "\nstage 4: eager dispatch vs graph replay, same work")
    _log(rank, f"  eager  min {e_min:9.1f} us   median {e_med:9.1f} us   "
               f"per collective {e_min / N_COLLECTIVES:7.1f} us")
    _log(rank, f"  graph  min {g_min:9.1f} us   median {g_med:9.1f} us   "
               f"per collective {g_min / N_COLLECTIVES:7.1f} us")
    _log(rank, f"  payload-only bound {ideal:9.1f} us   "
               f"per collective {ideal / N_COLLECTIVES:7.1f} us")
    if e_min > 0:
        _log(rank, f"\n  graph replay is {e_min / g_min:.2f}x faster; "
                   f"eager sits {e_min / ideal:.1f}x above the payload bound, "
                   f"graph sits {g_min / ideal:.1f}x")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
