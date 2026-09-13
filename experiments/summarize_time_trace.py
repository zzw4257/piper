"""Split a Piper iteration into Ray/driver, host dispatch, and GPU waiting.

Reads the per-rank TSVs written by PIPER_TIME_TRACE and pairs them with the
driver-side iteration time the harness records. The dispatch loop enqueues
without blocking except where a node waits on an event, so:

  dispatch  = last node's host timestamp - first node's
  drain     = end marker - last-enqueue marker   (host waiting for the GPU)
  ray/driver = driver iteration time - (dispatch + drain)

    python experiments/summarize_time_trace.py <trace_dir> [driver_ms]
"""
import glob
import os
import re
import sys


def main(argv):
    d = argv[0]
    driver_ms = float(argv[1]) if len(argv) > 1 else None
    files = sorted(glob.glob(os.path.join(d, "trace_rank*_step*.tsv")))
    if not files:
        print("no traces in", d); return 1
    per_step = {}
    for f in files:
        m = re.search(r"rank(\d+)_step(\d+)", f)
        rank, step = int(m.group(1)), int(m.group(2))
        rows = [l.rstrip("\n").split("\t") for l in open(f)][1:]
        ts = {r[0]: float(r[4]) for r in rows if r[0].startswith("__")}
        node_ts = [float(r[4]) for r in rows if not r[0].startswith("__")]
        n_nodes = len(node_ts)
        if not node_ts or "__last_enqueue__" not in ts:
            continue
        dispatch = (ts["__last_enqueue__"] - node_ts[0]) * 1000
        drain = (ts["__end__"] - ts["__last_enqueue__"]) * 1000
        per_step.setdefault(step, []).append((rank, dispatch, drain, n_nodes))
    steps = sorted(per_step)
    print(f"{'step':>5}{'nodes':>7}{'dispatch ms':>13}{'drain ms':>11}"
          + (f"{'driver ms':>11}{'ray+driver':>12}" if driver_ms else ""))
    for s in steps:
        rows = per_step[s]
        disp = max(r[1] for r in rows); dr = max(r[2] for r in rows); nn = rows[0][3]
        line = f"{s:>5}{nn:>7}{disp:>13.2f}{dr:>11.2f}"
        if driver_ms:
            line += f"{driver_ms:>11.2f}{driver_ms - disp - dr:>12.2f}"
        print(line)
    last = per_step[steps[-1]]
    disp = max(r[1] for r in last); nn = last[0][3]
    print(f"\nper-node host cost, last step: {disp / max(nn - 1, 1) * 1000:.0f} us "
          f"over {nn} nodes")

    # Where the host time goes, by task type: the gap before a node is the cost
    # of the node dispatched before it.
    import collections
    by_task = collections.defaultdict(list)
    f = sorted(glob.glob(os.path.join(d, "trace_rank0_step*.tsv")),
               key=lambda p: int(re.search(r"step(\d+)", p).group(1)))[-1]
    rows = [l.rstrip("\n").split("\t") for l in open(f)][1:]
    nodes = [(r[0], r[1], float(r[4])) for r in rows if not r[0].startswith("__")]
    end = [float(r[4]) for r in rows if r[0] == "__last_enqueue__"][0]
    for i, (uid, task, ts) in enumerate(nodes):
        nxt = nodes[i + 1][2] if i + 1 < len(nodes) else end
        by_task[task].append((nxt - ts) * 1e6)
    print(f"\n{'task':<24}{'n':>4}{'total ms':>10}{'mean us':>10}{'max us':>10}")
    tot = sum(sum(v) for v in by_task.values())
    for t, v in sorted(by_task.items(), key=lambda kv: -sum(kv[1])):
        print(f"{t:<24}{len(v):>4}{sum(v)/1000:>10.2f}{sum(v)/len(v):>10.0f}{max(v):>10.0f}")
    print(f"{'TOTAL':<24}{sum(len(v) for v in by_task.values()):>4}{tot/1000:>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
