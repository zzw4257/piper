"""The transfer function from injected skew to collective cost (log F48/F49).

F48 first wrote the law as cost ~ max(transport, skew) and its own data says
otherwise: skewed minus synced is constant at 1870-1926 us across a 64x payload
range, i.e. the two terms *add*. Physically they must: the group cannot start
until the last rank arrives, so a waiting rank measures skew + transport while
the late rank measures transport alone.

This sweeps skew as well as payload so the form is tested rather than inferred
from two endpoints -- the F17/F19 lesson about quadratic fits through two points.

    torchrun --nproc_per_node=4 experiments/probe_skew_transfer.py
    PAYLOADS=4,32,128 SKEWS=0,25,50,100,200,400,800,1600,3200 ITERS=25
"""
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

PAYLOADS = [float(x) for x in os.environ.get("PAYLOADS", "4,32,128").split(",")]
SKEWS = [float(x) for x in os.environ.get("SKEWS", "0,25,50,100,200,400,800,1600,3200").split(",")]
ITERS = int(os.environ.get("ITERS", "25"))
OUT = os.environ.get("XFER_OUT", "")


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    n, me = dist.get_world_size(), dist.get_rank()
    dev = torch.device("cuda", rank % torch.cuda.device_count())
    late = n // 2

    # Calibrate torch.cuda._sleep cycles per microsecond on this card rather
    # than assuming 1.9 GHz: the injected skew has to be the axis, not a guess.
    probe = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    probe[0].record(); torch.cuda._sleep(int(2e7)); probe[1].record()
    torch.cuda.synchronize()
    cycles_per_us = 2e7 / (probe[0].elapsed_time(probe[1]) * 1000.0)

    rows = []
    for mib in PAYLOADS:
        numel = max(1, int(mib * 2**20 / 4))
        t = torch.ones(numel, device=dev)
        for skew in SKEWS:
            for _ in range(3):
                dist.all_reduce(t)
            torch.cuda.synchronize(); dist.barrier()
            d = []
            for _ in range(ITERS):
                dist.barrier(); torch.cuda.synchronize()
                if me == late and skew > 0:
                    torch.cuda._sleep(int(skew * cycles_per_us))
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); dist.all_reduce(t); e1.record()
                torch.cuda.synchronize()
                d.append(e0.elapsed_time(e1) * 1000.0)
            d.sort()
            rows.append({"mib": mib, "skew": skew, "rank": me, "late": me == late,
                         "min": d[0], "median": statistics.median(d)})
        del t
        torch.cuda.empty_cache()

    gathered = [None] * n
    dist.all_gather_object(gathered, rows)
    if me == 0:
        flat = [r for rs in gathered for r in rs]
        meta = {"world": n, "device": torch.cuda.get_device_name(0),
                "cycles_per_us": cycles_per_us, "iters": ITERS, "late_rank": late}
        if OUT:
            json.dump({"meta": meta, "rows": flat}, open(OUT, "w"))
        print(json.dumps(meta))
        print(f"\n{'MiB':>6}{'skew':>7}{'waiting':>10}{'late':>9}{'wait-base':>11}{'additive?':>11}")
        for mib in PAYLOADS:
            base = min(r["min"] for r in flat if r["mib"] == mib and r["skew"] == 0)
            for s in SKEWS:
                sel = [r for r in flat if r["mib"] == mib and r["skew"] == s]
                w = max(r["min"] for r in sel if not r["late"])
                l = min(r["min"] for r in sel if r["late"])
                print(f"{mib:>6}{s:>7.0f}{w:>10.1f}{l:>9.1f}{w-base:>11.1f}"
                      f"{(w - base) / s if s else float('nan'):>11.2f}")
        print("\nadditive law predicts (waiting - base)/skew -> 1.0 at every payload;")
        print("a max() law predicts waiting = max(base, skew), i.e. the ratio falling")
        print("below 1 whenever base > skew.")
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
