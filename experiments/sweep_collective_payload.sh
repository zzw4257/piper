#!/usr/bin/env bash
# Where does a TP all-reduce stop being latency-bound? Sweep payload at TP=2.
set -uo pipefail
cd /raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=0,3
L=/var/tmp/ziweizho-scripts/logs
DIM=4096; HID=16384
for rep in 1 2 3; do
  for B in 256 512 1024 2048 4096 8192; do
    timeout 1800 python examples/test_harness.py --test-file examples/test_tp_mlp.py \
      --base-schedule examples/base-schedules/tp2_mb1_s1.json --schedule custom \
      --dim $DIM --hidden $HID --batch-size $B --stages 1 --tp 2 \
      --dtype bf16 --init random --warmup 4 --iters 8 \
      --pytorch-profiler --pytorch-profiler-iters 6 > $L/pl_${B}_$rep.log 2>&1 || true
  done
done
python3 - <<'PY'
import csv, glob, json, collections, statistics, os
L = "/var/tmp/ziweizho-scripts/logs"
DIM, BW = 4096, 360e9   # BW measured in F11
_G = {"kernel", "gpu_memcpy", "gpu_memset"}
def uid(n):
    h = n.split("::", 1)[0]; i = h.find(":uid"); return h[i+4:] if i >= 0 else None
def comm_us(run, iters=6):
    tot = 0.0; files = glob.glob(run + "/pytorch_profile_*dprank*.json")
    if not files: return None
    for f in files:
        for e in json.load(open(f)).get("traceEvents", []):
            if e.get("cat") in _G and e.get("dur") and (uid(e.get("name","")) or "").startswith("tp_all_reduce"):
                tot += e["dur"]
    return tot / iters / len(files)
print(f"{'batch':>6}{'payload':>10}{'bound/coll':>12}{'measured/coll':>15}{'ratio':>8}")
for B in (256, 512, 1024, 2048, 4096, 8192):
    vals = []
    for rep in (1, 2, 3):
        log = f"{L}/pl_{B}_{rep}.log"
        if not os.path.exists(log): continue
        runs = [l for l in open(log) if "out/" in l]
        import re
        m = re.findall(r"out/\d{8}_\d{6}", "".join(runs))
        if not m: continue
        c = comm_us(os.path.join("/raid/user_data/ziweizho/piper-tp", m[-1]))
        if c: vals.append(c / 2.0)      # two collectives per iteration
    if not vals: print(f"{B:>6}  no data"); continue
    payload = B * DIM * 2
    bound = payload / BW * 1e6
    best = min(vals)
    print(f"{B:>6}{payload/2**20:>9.1f}M{bound:>12.1f}{best:>15.1f}{best/bound:>7.1f}x")
PY
