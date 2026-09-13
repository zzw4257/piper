"""Of a compute node's host cost, how much is inside compute.forward/backward?

F54 left ~2x of Piper's 3.8x host-side tax unattributed and said an external
mock could not resolve it. This reads the inner_us column PIPER_TIME_TRACE now
writes: the node's total host cost is the gap to the next node's timestamp, and
`inner` is the compute call alone, so the difference is the executor's own work
around it -- event creation, buffer store traffic, boundary handling, dispatch.
"""
import collections
import glob
import os
import re
import statistics
import sys


def main(argv):
    d = argv[0]
    files = sorted(glob.glob(os.path.join(d, "trace_rank0_step*.tsv")),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    if not files:
        print("no traces"); return 1
    agg = collections.defaultdict(lambda: ([], []))
    for f in files[len(files) // 2:]:            # steady state only
        rows = [l.rstrip("\n").split("\t") for l in open(f)][1:]
        nodes = [(r[0], r[1], float(r[4]), float(r[5])) for r in rows if not r[0].startswith("__")]
        end = [float(r[4]) for r in rows if r[0] == "__last_enqueue__"][0]
        for i, (uid, task, ts, inner) in enumerate(nodes):
            nxt = nodes[i + 1][2] if i + 1 < len(nodes) else end
            total = (nxt - ts) * 1e6
            agg[task][0].append(total)
            if inner >= 0:
                agg[task][1].append(inner)
    print(f"{'task':<24}{'n':>4}{'total us':>10}{'inner us':>10}{'executor us':>13}{'inner %':>9}")
    t_all = t_in = 0.0
    for task, (tot, inn) in sorted(agg.items(), key=lambda kv: -statistics.median(kv[1][0]) * len(kv[1][0])):
        mt = statistics.median(tot)
        mi = statistics.median(inn) if inn else 0.0
        n = len(tot) // max(1, len(files) - len(files) // 2)
        t_all += mt * n; t_in += mi * n
        print(f"{task:<24}{n:>4}{mt:>10.0f}{mi if inn else float('nan'):>10.0f}"
              f"{mt - mi:>13.0f}{(mi / mt * 100) if inn else float('nan'):>8.0f}%")
    print(f"\nper iteration: {t_all/1000:.2f} ms total, {t_in/1000:.2f} ms inside compute.* "
          f"({t_in/t_all*100:.0f}%), {(t_all-t_in)/1000:.2f} ms executor")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
