#!/usr/bin/env bash
# P2 at a resolution that can see it (log F46): the ring's communication
# fraction goes as 1/s_local, so sweep the local chunk down. Interleaved A/B
# (spliced vs hoist d=1), repeated, minimum of each arm.
set -u
B=$HOME; R=$B/piper-tp
export PATH=$B/piper-venv/bin:$HOME/.local/bin:$PATH
cd $R; OUT=logs/h200_overlap.txt; T=$B/ray_tmp
echo "=== start $(date -Is) ===" >> $OUT
pick() { local n=$1 mu=$2 mf=$3 ok=""
  for s in 1 2 3; do
    ok=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
         | awk -F', ' -v mu=$mu -v mf=$mf '$2<mu && ($4-$3)/1024>mf {print $1}' | head -$n | paste -sd,)
    [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 20; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick 4 20 6) && break
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "no 4 cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C" >> $OUT
run() { local S=$1 SEQ=$2 TAG=$3
  local LOG=logs/h200_ov_${TAG}_${S}_${SEQ}.log
  CUDA_VISIBLE_DEVICES=$C timeout 1800 python examples/test_harness.py \
    --test-file examples/test_ring_attn.py --base-schedule examples/base-schedules/$S.json \
    --schedule custom --steps 4 --seq $SEQ --dim 512 --heads 8 --batch-size 8 \
    --warmup 2 --iters 10 --temp-dir $T > $LOG 2>&1
  local RC=$? RUN=$(grep -oE "out/[0-9]{8}_[0-9]{6}" $LOG | tail -1)
  echo -n "  $S seq=$SEQ rc=$RC " >> $OUT
  [ -n "$RUN" ] && python - "$RUN" <<'PY' >> $OUT 2>&1 || echo "" >> $OUT
import glob, json, sys
ts=[]; pk=[]
for f in sorted(glob.glob(f"{sys.argv[1]}/cp_metrics_dp*.json")):
    m=json.load(open(f)); ts.append(min(m["iter_times_s"])); pk.append(max(m["peak_memory_by_rank"].values()))
print(f"min_iter={max(ts)*1000:.2f}ms peak={max(pk)/2**20:.0f}MiB")
PY
}
for SEQ in 1024 2048 4096 8192; do
  echo "-- s_local=$((SEQ/4))  (seq=$SEQ)" >> $OUT
  for rep in 1 2 3; do
    run cp4_ring_dp       $SEQ r$rep
    run cp4_ring_dp_hoist $SEQ r$rep
  done
done
echo "done $(date -Is)" >> $OUT
