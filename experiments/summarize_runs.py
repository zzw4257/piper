"""Summarize harness run directories: one line per run, one row per DP rank.

    python experiments/summarize_runs.py out/20260910_* 

Reads results.csv (written by test_harness.py) and, when present,
tp_metrics_dp*.json (written by test_tp_mlp.py, which carries the losses the
harness CSV drops).
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys


def _summarize(run_dir: str) -> str | None:
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    means = [float(r["iter_time_mean_s"]) for r in rows]
    stds = [float(r["iter_time_std_s"]) for r in rows]
    peaks = [
        float(v) for r in rows for k, v in r.items()
        if k.startswith("peak_memory_pp") and v
    ]
    schedule = rows[0].get("schedule", "?")
    model = rows[0].get("model", "?")
    samples = rows[0].get("samples", "?")

    losses = ""
    metrics_files = sorted(glob.glob(os.path.join(run_dir, "tp_metrics_dp*.json")))
    if metrics_files:
        first = json.load(open(metrics_files[0]))
        if first.get("losses"):
            losses = "  loss[0]=%.6f" % first["losses"][0]

    return (
        "%-26s %-34s n=%-3s iter %8.3f ms (std %6.3f)  peak %6.2f GB%s"
        % (schedule, model, samples, statistics.fmean(means) * 1000,
           max(stds) * 1000, max(peaks) if peaks else float("nan"), losses)
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args(argv)

    lines = []
    for d in args.dirs:
        line = _summarize(d)
        if line:
            lines.append((os.path.basename(d.rstrip("/")), line))
    if not lines:
        print("no summarizable run directories", file=sys.stderr)
        return 1
    for stamp, line in sorted(lines):
        print(f"{stamp}  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
