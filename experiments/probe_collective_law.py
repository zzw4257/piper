"""Is a collective's cost a function of its bytes, and under what condition?

Reconciles F19/F31/F32/F33/F45 into one testable law:

    cost(collective) ~ max( transport(bytes), arrival_skew )

F31/F32 measured "cost is nearly independent of payload" on a contended host
with computation between collectives -- i.e. in the regime where the skew term
dominates. If the law holds, the same sweep with the ranks synchronized should
be linear in bytes, and the crossover payload is where a bandwidth model starts
to be valid.

Three regimes per payload, same collective, same iteration count:

  synced  : barrier + device sync immediately before each call, skew ~ 0
  natural : a matmul between calls and no barrier, so ranks drift as they do
            inside a real Piper step
  skewed  : one rank delayed by SKEW_US before each call

Two collectives: all_reduce (global) and a one-hop ring (neighbour P2P), so the
same sweep also locates the eager threshold from F45 -- the payload above which
the ring stops localizing a straggler to one hop.

    torchrun --nproc_per_node=4 experiments/probe_collective_law.py
    PAYLOADS=1,4,16,64,256 ITERS=30 MATMUL_MS=2 SKEW_US=2000 ...
"""
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

PAYLOADS = [float(x) for x in os.environ.get("PAYLOADS", "0.5,1,2,4,8,16,32,64,128,256").split(",")]
ITERS = int(os.environ.get("ITERS", "30"))
SKEW_US = float(os.environ.get("SKEW_US", "2000"))
MATMUL_MS = float(os.environ.get("MATMUL_MS", "2"))
OUT = os.environ.get("LAW_OUT", "")


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    n, me = dist.get_world_size(), dist.get_rank()
    dev = torch.device("cuda", rank % torch.cuda.device_count())

    # A matmul sized to roughly MATMUL_MS on this card, calibrated once so the
    # "natural" regime drifts the ranks by a realistic amount on both hosts.
    m = 2048
    a = torch.randn(m, m, device=dev)
    for _ in range(5):
        a @ a
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); a @ a; e.record(); torch.cuda.synchronize()
    one = s.elapsed_time(e)
    reps = max(1, int(MATMUL_MS / max(one, 1e-3)))

    def busy():
        for _ in range(reps):
            a.mul_(1.0)

    rows = []
    for mib in PAYLOADS:
        numel = max(1, int(mib * 2**20 / 4))
        t = torch.ones(numel, device=dev)
        buf = torch.empty_like(t)

        def all_reduce():
            dist.all_reduce(t)

        def ring():
            ops = [dist.P2POp(dist.isend, t, (me + 1) % n),
                   dist.P2POp(dist.irecv, buf, (me - 1) % n)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()

        for cname, coll in (("all_reduce", all_reduce), ("ring", ring)):
            for regime in ("synced", "natural", "skewed"):
                for _ in range(3):
                    coll()
                torch.cuda.synchronize(); dist.barrier()
                d = []
                for _ in range(ITERS):
                    if regime == "synced":
                        dist.barrier(); torch.cuda.synchronize()
                    elif regime == "natural":
                        busy()
                    elif regime == "skewed":
                        dist.barrier(); torch.cuda.synchronize()
                        if me == n // 2:
                            torch.cuda._sleep(int(SKEW_US * 1900))
                    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    ev0.record(); coll(); ev1.record()
                    torch.cuda.synchronize()
                    d.append(ev0.elapsed_time(ev1) * 1000.0)
                d.sort()
                rows.append({"mib": mib, "coll": cname, "regime": regime, "rank": me,
                             "min": d[0], "median": statistics.median(d),
                             "p90": d[int(0.9 * (len(d) - 1))]})
        del t, buf
        torch.cuda.empty_cache()

    gathered = [None] * n
    dist.all_gather_object(gathered, rows)
    if me == 0:
        flat = [r for rs in gathered for r in rs]
        meta = {"world": n, "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "iters": ITERS, "skew_us": SKEW_US,
                "matmul_ms": MATMUL_MS, "matmul_reps": reps,
                "skewed_rank": n // 2}
        if OUT:
            with open(OUT, "w") as f:
                json.dump({"meta": meta, "rows": flat}, f)
        print(json.dumps(meta))
        print(f"{'MiB':>7} {'coll':<11}{'regime':<9}{'min_us':>9}{'med_us':>9}{'GB/s_min':>10}")
        for mib in PAYLOADS:
            for c in ("all_reduce", "ring"):
                for rg in ("synced", "natural", "skewed"):
                    sel = [r for r in flat if r["mib"] == mib and r["coll"] == c and r["regime"] == rg]
                    if not sel:
                        continue
                    mn = max(r["min"] for r in sel)      # group cost = slowest rank
                    md = max(r["median"] for r in sel)
                    bw = (mib * 2**20) / (mn * 1e-6) / 1e9
                    print(f"{mib:>7} {c:<11}{rg:<9}{mn:>9.1f}{md:>9.1f}{bw:>10.1f}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
