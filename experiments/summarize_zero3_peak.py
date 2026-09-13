"""Turn logs/zero3_peak.txt into slopes against the F37/F39 predictions.

Per arm, fit peak_bytes = a + slope * (stages * per_stage_bytes) over the
measured depths and print slope in units of stage-bytes. Predictions:
zero3 (shipped) 3.0, zero3_pf1 (prefetch_distance=1) 2.0, dp 4.0. The intercept
absorbs activations, workspace and the CUDA context.

    python experiments/summarize_zero3_peak.py [logs/zero3_peak.txt]
"""
import ast
import re
import sys

PRED = {"zero3": 3.0, "zero3_pf1": 2.0, "dp": 4.0}


def parse(path):
    runs = {}  # (mode, stages) -> {"per_stage": int, "peaks": [int per rank]}
    cur = None
    for line in open(path):
        m = re.match(r"== (\w+) stages=(\d+)", line)
        if m:
            cur = (m.group(1), int(m.group(2)))
            runs[cur] = {"per_stage": None, "peaks": [], "rc": None}
            continue
        if cur is None:
            continue
        m = re.search(r"rc=(\d+).*per_stage_bytes=(\d+)", line)
        if m:
            runs[cur]["rc"] = int(m.group(1)); runs[cur]["per_stage"] = int(m.group(2))
        m = re.search(r"peak_memory_by_rank=(\{.*\})", line)
        if m:
            d = ast.literal_eval(m.group(1))
            runs[cur]["peaks"].append(max(d.values()))
    return runs


def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    return slope, my - slope * mx


def main(path="logs/zero3_peak.txt", per_stage_override=None):
    runs = parse(path)
    if per_stage_override:
        for r in runs.values():
            r["per_stage"] = int(per_stage_override)
    if not runs:
        print("no completed runs in", path); return 1
    by_mode = {}
    for (mode, stages), r in sorted(runs.items()):
        ok = r["rc"] == 0 and r["peaks"]
        peak = max(r["peaks"]) if r["peaks"] else None
        print(f"{mode:<10} stages={stages:<2} rc={r['rc']} peak_max_rank={peak/2**30 if peak else None:.2f} GiB"
              if peak else f"{mode:<10} stages={stages:<2} rc={r['rc']} (no metrics)")
        if ok:
            by_mode.setdefault(mode, []).append((stages * r["per_stage"], peak))
    print()
    print(f"{'arm':<10}{'points':>7}{'slope (stage-bytes)':>22}{'predicted':>11}{'intercept GiB':>15}")
    for mode, pts in by_mode.items():
        if len(pts) < 2:
            print(f"{mode:<10}{len(pts):>7}   (need >= 2 depths)"); continue
        s, a = fit([x for x, _ in pts], [y for _, y in pts])
        print(f"{mode:<10}{len(pts):>7}{s:>22.2f}{PRED.get(mode, float('nan')):>11.1f}{a/2**30:>15.2f}")
    return 0


if __name__ == "__main__":
    # TPMlp stage = pre(dim^2) + up + down (2 dim*hid) + post(dim^2); the runner
    # logged only 2*dim*hid*4. Pass the true value as the second argument.
    sys.exit(main(*sys.argv[1:]))
