"""Attribute GPU time to TrainingDAG nodes from a harness PyTorch-profiler trace.

    python experiments/profile_tp.py out/<ts>/pytorch_profile_*_dprank0.json --iters 2

Piper labels every GPU event with the DAG node that issued it, so the trace can
be aggregated per node uid rather than per kernel name.

Iterations are split on gaps in the GPU timeline, not by counting occurrences of
each node: a node emits however many kernels its work needs (`upd.0` emits six
for a foreach Adam step), so occurrence counting smears one iteration across
several. With --iters N the N-1 largest gaps are taken as the boundaries, which
needs no threshold tuning.

Per-iteration variance is reported first on purpose. On a shared host it is the
only way to see that a communication share is contention rather than a
measurement.
"""
import argparse
import collections
import json
import sys

_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _node_uid(name: str) -> str | None:
    """'forward:{...}:uids0.seg1::kernel_name' -> 's0.seg1'."""
    head = name.split("::", 1)[0]
    i = head.find(":uid")
    return head[i + 4:] if i >= 0 else None


def _task_of(name: str) -> str:
    return name.split(":", 1)[0]


def _split_iterations(events: list[dict], n_iters: int | None, gap_us: float) -> list[int]:
    """Return the iteration index of each event (events must be ts-sorted).

    Anchored on the UPD node, which runs exactly once per iteration and always
    last: a new iteration starts at the first non-UPD event after a run of UPD
    events. Gap heuristics are wrong here -- a TP rank waiting on its peer inside
    the NCCL kernel (notes/log.md F11) produces gaps *within* an iteration that
    are larger than the gaps between them, which split one iteration into pieces
    and yields absurd spans like 100us.
    """
    upd = [_node_uid(e["name"]).startswith("upd") for e in events]
    if any(upd):
        out, cur, prev_was_upd = [], 0, False
        for i in range(len(events)):
            if prev_was_upd and not upd[i]:
                cur += 1
            out.append(cur)
            prev_was_upd = upd[i]
        return out

    # No UPD in the trace: fall back to gaps.
    gaps = [
        (events[i + 1]["ts"] - (events[i]["ts"] + events[i]["dur"]), i + 1)
        for i in range(len(events) - 1)
    ]
    if n_iters and n_iters > 1:
        boundaries = {i for _, i in sorted(gaps, reverse=True)[: n_iters - 1]}
    else:
        boundaries = {i for g, i in gaps if g > gap_us}

    out, cur = [], 0
    for i in range(len(events)):
        if i in boundaries:
            cur += 1
        out.append(cur)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--iters", type=int, default=None,
                    help="Number of profiled iterations; splits on the N-1 largest gaps.")
    ap.add_argument("--gap-us", type=float, default=500.0,
                    help="Gap treated as an iteration boundary when --iters is absent.")
    args = ap.parse_args(argv)

    with open(args.trace, encoding="utf-8") as f:
        raw = json.load(f).get("traceEvents", [])
    events = [
        e for e in raw
        if e.get("cat") in _GPU_CATS and e.get("dur") and e.get("name")
        and _node_uid(e["name"]) is not None
    ]
    if not events:
        print("no labelled GPU events with a duration in this trace", file=sys.stderr)
        return 1
    events.sort(key=lambda e: e["ts"])

    which = _split_iterations(events, args.iters, args.gap_us)
    spans: dict[int, tuple[float, float]] = {}
    for e, it in zip(events, which):
        lo, hi = spans.get(it, (e["ts"], e["ts"] + e["dur"]))
        spans[it] = (min(lo, e["ts"]), max(hi, e["ts"] + e["dur"]))
    per_iter: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    kernels: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    tasks: dict[str, str] = {}
    for e, it in zip(events, which):
        uid = _node_uid(e["name"])
        per_iter[it][uid] += e["dur"]
        kernels[it][uid] += 1
        tasks.setdefault(uid, _task_of(e["name"]))

    iters = sorted(per_iter)
    print(f"{len(iters)} iteration(s), {len(tasks)} DAG nodes, {len(events)} GPU events\n")

    totals = [sum(per_iter[i].values()) for i in iters]
    spread = (max(totals) / min(totals)) if min(totals) else float("inf")
    for i, tot in zip(iters, totals):
        tp = sum(v for u, v in per_iter[i].items() if u.startswith("tp_all_reduce"))
        share = 100.0 * tp / tot if tot else float("nan")
        print(f"iter{i}: GPU {tot:9.1f} us   TP all-reduce {tp:8.1f} us  {share:5.1f}%   "
              f"({sum(kernels[i].values())} events)")
    # Summed kernel durations do not shrink when work overlaps, so the sum alone
    # cannot answer "was the collective hidden". Compare it against the wall span
    # of the iteration's GPU timeline: sum > span means streams ran concurrently.
    print()
    for i, tot in zip(iters, totals):
        lo, hi = spans[i]
        span = hi - lo
        print(f"iter{i}: sum {tot:9.1f} us   span {span:9.1f} us   "
              f"concurrency {tot / span if span else float('nan'):5.2f}x")

    print(f"\nper-iteration total spread: {spread:.2f}x")
    if spread > 1.3:
        print("WARNING: iterations differ by more than 30%. On a shared host that is\n"
              "         contention, and the shares above are not a measurement.")

    print()
    uids = sorted(tasks, key=lambda u: -sum(per_iter[i][u] for i in iters))
    head = "  ".join(f"iter{i} (us)" for i in iters)
    print(f"{'node':<22s} {'task':<24s} {head}     min      max/min")
    for uid in uids:
        cells = "  ".join(f"{per_iter[i][uid]:11.1f}" for i in iters)
        vals = [per_iter[i][uid] for i in iters if per_iter[i][uid]]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
        ratio = (hi / lo) if lo else float("inf")
        print(f"{uid:<22s} {tasks[uid]:<24s} {cells} {lo:9.1f} {ratio:9.2f}x")

    # On a shared host the minimum is the only defensible statistic: nothing here
    # makes a kernel *faster* than its uncontended time, so min converges to it
    # from above while the mean tracks whoever else is using the machine. Compute
    # and communication are reported separately because they contend for
    # different resources -- SMs are per-GPU and can be held exclusively, but
    # NVLink/NVSwitch is machine-wide, so a job can own its GPUs outright and
    # still have its collectives squeezed.
    min_by_uid = {
        u: min([per_iter[i][u] for i in iters if per_iter[i][u]] or [0.0])
        for u in uids
    }
    comm = sum(v for u, v in min_by_uid.items() if u.startswith("tp_all_reduce"))
    comp = sum(v for u, v in min_by_uid.items() if not u.startswith("tp_all_reduce"))
    print(f"\nmin-over-iterations:  compute {comp:9.1f} us   TP all-reduce {comm:8.1f} us"
          f"   share {100.0 * comm / (comp + comm):5.1f}%")
    worst = max(
        (max(per_iter[i][u] for i in iters) / min_by_uid[u], u)
        for u in uids if min_by_uid[u]
    )
    print(f"worst per-node max/min: {worst[0]:.1f}x ({worst[1]})")
    comm_nodes = [u for u in uids if u.startswith("tp_all_reduce") and min_by_uid[u]]
    if len(comm_nodes) > 1:
        lo = min(min_by_uid[u] for u in comm_nodes)
        hi = max(min_by_uid[u] for u in comm_nodes)
        if hi > 2 * lo:
            print(
                f"\nNOTE: collectives with the same payload differ by {hi / lo:.1f}x even at\n"
                f"      their minimum. The first collective of an iteration absorbs rank\n"
                f"      arrival skew: each device is driven by its own Ray actor, so one\n"
                f"      rank enters the NCCL kernel early and spins. Compare the two ranks'\n"
                f"      traces -- if one is large where the other is ~min, that time is skew,\n"
                f"      not communication cost. Use the *later* collective as the estimate."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
