"""GPU check for F75: on models whose lowering routing does not change, routed runs equal threaded ones.

    CUDA_VISIBLE_DEVICES=a,b,c,d python experiments/check_route_equiv.py [--only=name,...]

Each case runs one schedule twice, as given and with {"op": "route", "mode": "consumers"}
prepended. The DAGs have the same nodes and edges, but every forward and backward then
takes the executor's routed path, including its handling of each boundary collective
(TP all-reduce, EP all-to-all, CP ring, hoisted CP ring). Losses and every rank's
parameter checksum must be equal to the last bit.
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

CASES = [  # name, test file, schedule, extra args, GPUs
    ("tp2", "test_tp_mlp.py", "tp2", [], 2),
    ("tp2_derived", "test_tp_mlp.py", "tp2_derived", [], 2),
    ("ep2", "test_tp_mlp.py", "tp2_as_ep", [], 2),
    ("cp2_ring", "test_ring_attn.py", "cp2_ring", ["--steps", "2"], 2),
    ("pp2_tp2_1f1b", "test_tp_mlp.py", "pp2_tp2_mb4_1f1b", ["--stages", "2"], 4),
    ("cp4_hoist", "test_ring_attn.py", "cp4_ring_dp_hoist", ["--steps", "4"], 4),
]


def run(test_file, schedule_path, extra, repo):
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", f"examples/{test_file}",
           "--base-schedule", schedule_path, "--schedule", "custom", *extra]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return None, (proc.stdout[-1500:] + proc.stderr[-1500:])
    run_dir = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", proc.stdout)[-1])
    ms = [json.load(open(f)) for f in sorted(glob.glob(f"{run_dir}/*_metrics_dp*.json"))]
    losses = [m["losses"] for m in ms]
    ck = sorted((r["rank"], r["sum"], r["sumsq"]) for m in ms for r in m.get("param_checksums") or [])
    return {"dir": os.path.relpath(run_dir, repo), "losses": losses, "ck": ck}, None


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    extra_all = [a for a in sys.argv[1:] if not a.startswith("--only=")]
    ngpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    ok = True
    tmp = tempfile.mkdtemp(prefix="route_equiv_")
    for name, test_file, sched, extra, need in CASES:
        if (only and name not in only) or need > ngpu:
            continue
        base = f"examples/base-schedules/{sched}.json"
        routed = os.path.join(tmp, f"{sched}_routed.json")
        json.dump([{"op": "route", "mode": "consumers"}] + json.load(open(os.path.join(repo, base))),
                  open(routed, "w"), indent=1)
        a, ea = run(test_file, base, extra + extra_all, repo)
        b, eb = run(test_file, routed, extra + extra_all, repo)
        if a is None or b is None:
            ok = False
            print(f"{name:14s} FAILED TO RUN\n{ea or ''}{eb or ''}")
            continue
        same = a["losses"] == b["losses"] and a["ck"] == b["ck"] and bool(a["ck"])
        ok &= same
        print(f"{name:14s} threaded {a['dir']}  routed {b['dir']}  losses {[[round(x, 6) for x in l] for l in a['losses']]}  "
              f"{len(a['ck'])} rank checksums  {'identical' if same else 'DIFFER'}")
        if not same:
            print(f"  threaded {a['losses']} {a['ck']}\n  routed   {b['losses']} {b['ck']}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
