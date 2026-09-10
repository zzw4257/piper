"""Measure rank arrival skew directly, instead of inferring it from NCCL kernels.

    python experiments/measure_skew.py out/<ts>

Joins the per-dp_rank metrics artifacts on the wall clock. TP peers live in
different driver processes (Piper runs one `run_dp_rank` per dp_rank), so a
single process cannot see the skew; only the join can.

Reports, per step: the spread in entry times across ranks, and the spread in exit
times. Entry spread is what the first collective of the step absorbs by spinning
inside the NCCL kernel (notes/log.md F11), and what F16 had to model as a
constant to rank configurations correctly.
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load(run_dir: str) -> dict[int, dict[int, tuple[int, int]]]:
    """rank -> iter -> (enter_ns, exit_ns)."""
    out: dict[int, dict[int, tuple[int, int]]] = {}
    files = sorted(glob.glob(os.path.join(run_dir, "tp_metrics_dp*.json")))
    if not files:
        sys.exit(f"no tp_metrics_dp*.json under {run_dir}")
    for f in files:
        m = json.load(open(f))
        for rank, rows in (m.get("step_timestamps") or {}).items():
            out[int(rank)] = {int(i): (int(a), int(b)) for i, a, b in rows}
    if not out:
        sys.exit("artifacts carry no step_timestamps; rerun with the current driver")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)

    per_rank = load(args.run_dir)
    ranks = sorted(per_rank)
    print(f"ranks: {ranks}")
    if len(ranks) < 2:
        print("only one rank recorded; skew is not defined for a single rank")
        return 1

    iters = sorted(set.intersection(*(set(per_rank[r]) for r in ranks)))
    print(f"steps with all ranks present: {len(iters)}\n")
    print(f"{'step':<6s}{'entry spread (us)':>19s}{'exit spread (us)':>18s}"
          f"{'slowest rank in':>17s}")
    entry, exit_ = [], []
    for i in iters:
        e = [per_rank[r][i][0] for r in ranks]
        x = [per_rank[r][i][1] for r in ranks]
        es, xs = (max(e) - min(e)) / 1e3, (max(x) - min(x)) / 1e3
        entry.append(es); exit_.append(xs)
        late = ranks[e.index(max(e))]
        print(f"{i:<6d}{es:>19.1f}{xs:>18.1f}{late:>17d}")

    print(f"\nentry spread: min {min(entry):8.1f} us   median "
          f"{statistics.median(entry):8.1f} us   max {max(entry):8.1f} us")
    print(f"exit  spread: min {min(exit_):8.1f} us   median "
          f"{statistics.median(exit_):8.1f} us   max {max(exit_):8.1f} us")
    print(f"\nF16 had to assume SKEW_US = 12000 to rank configurations correctly.")
    print(f"Measured median entry spread: {statistics.median(entry):.0f} us.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
