#!/usr/bin/env bash
set -uo pipefail
cd /raid/user_data/ziweizho/piper-tp
export PATH=/var/tmp/ziweizho-piper-venv/bin:$PATH RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=0,3,4,6
L=/var/tmp/ziweizho-scripts/logs
# Constant global batch and constant total hidden: TP only changes how much of
# the MLP each card owns. The all-reduce payload is [batch, dim] either way.
DIM=4096; HID=16384; B=8192
for rep in 1 2 3; do
  for tp in 2 4; do
    timeout 2400 python examples/test_harness.py --test-file examples/test_tp_mlp.py \
      --base-schedule examples/base-schedules/tp${tp}_mb1_s1.json --schedule custom \
      --dim $DIM --hidden $HID --batch-size $B --stages 1 --tp $tp \
      --dtype bf16 --init random --warmup 4 --iters 10 \
      --pytorch-profiler --pytorch-profiler-iters 6 > $L/sc_${tp}_$rep.log 2>&1 \
      || { echo "FAILED tp=$tp rep$rep"; tail -4 $L/sc_${tp}_$rep.log; continue; }
  done
done
echo "tp  rep  iter(ms)   compute(us)  comm(us)  coll/iter"
for tp in 2 4; do
  for rep in 1 2 3; do
    R=$(grep -o "out/[0-9_]*" $L/sc_${tp}_$rep.log 2>/dev/null | tail -1)
    [ -z "$R" ] && { printf "%-4s%-5s n/a\n" "$tp" "$rep"; continue; }
    printf "%-4s%-5s" "$tp" "$rep"
    python3 -c "
import csv,sys,statistics,json,glob,collections
run=sys.argv[1]
rows=list(csv.DictReader(open(run+'/results.csv')))
it=statistics.fmean([float(r['iter_time_mean_s'])*1e3 for r in rows])
_G={'kernel','gpu_memcpy','gpu_memset'}
def uid(n):
    h=n.split('::',1)[0]; i=h.find(':uid'); return h[i+4:] if i>=0 else None
tot=collections.Counter(); nev=0
files=glob.glob(run+'/pytorch_profile_*dprank*.json')
for f in files:
    for e in json.load(open(f)).get('traceEvents',[]):
        if e.get('cat') in _G and e.get('dur') and uid(e.get('name','')) is not None:
            u=uid(e['name'])
            k='comm' if u.startswith('tp_all_reduce') else 'compute'
            tot[k]+=e['dur']
            if k=='comm': nev+=1
r=len(files); n=6
print(f' {it:8.2f}   {tot[\"compute\"]/n/r:9.0f}   {tot[\"comm\"]/n/r:8.0f}   {nev/n/r:6.1f}')
" "$R"
  done
done
