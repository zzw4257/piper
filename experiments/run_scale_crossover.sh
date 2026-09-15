#!/usr/bin/env bash
# Where does the sync-mode win go as the model gets big enough to be GPU-bound?
# The 1.25-1.38x was measured on a model whose iteration is host dispatch (F50).
# Sweep the work per step and find where device / narrow / defer converge --
# that crossover is what the change is actually worth on a real model.
set -u
R=$1; V=$2; T=$3; TAG=$4; REPS=${5:-3}
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/scale_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
cuda_ok() { python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count()>0 else 1)' >/dev/null 2>&1; }
pick() { cuda_ok || return 1; local ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '($3-$2)/1024>25 {print $1}' | head -4 | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt 4 ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C=$(pick) && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
echo "cards=$C load: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd,)" >> $OUT

# seq grows the attention work quadratically; batch linearly. Four points spanning
# host-bound to GPU-bound, per the F50 sweep (GPU share 12% -> 84%).
for CFG in "1024 8" "2048 8" "4096 4" "8192 2"; do
  FREE=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F", " -v c="$C" 'BEGIN{split(c,a,",")} {i=NR-1; for(k in a) if(a[k]==i){f=($2-$1)/1024; if(m==""||f<m) m=f}} END{printf "%d", m}')
  set -- $CFG; SEQ=$1; BS=$2
  NEED=$(( SEQ*BS/1024 + 4 ))
  if [ "${FREE:-0}" -lt "$NEED" ]; then echo "-- seq=$SEQ batch=$BS SKIPPED (need ~${NEED}G, have ${FREE}G)" >> $OUT; continue; fi
  echo "-- seq=$SEQ batch=$BS (s_local=$((SEQ/4)))" >> $OUT
  for REP in $(seq 1 $REPS); do
    for M in device narrow defer; do
      D=/tmp/sc_${M}; rm -rf $D; mkdir -p $D
      RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C PIPER_TIME_TRACE=$D timeout 1800 \
        python examples/test_harness.py --test-file examples/test_ring_attn.py \
        --base-schedule examples/base-schedules/cp4_ring_dp.json --schedule custom \
        --steps 4 --seq $SEQ --dim 512 --heads 8 --batch-size $BS --warmup 2 --iters 10 \
        --temp-dir $T 2>&1 | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
      MS=$(python - "$RUN" <<'PY' 2>/dev/null
import glob, json, sys
import statistics
rows=[json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/cp_metrics_dp*.json"))]
if not rows: print("NA")
else:
    tot=[r.get("total_timed_s",0)/max(r.get("timed_iters",1),1)*1000 for r in rows]
    mn=[min(r["iter_times_s"])*1000 for r in rows]
    print(f"{max(tot):.2f}/{max(mn):.2f}")
PY
)
      SY=$(grep -h "__upd_final_sync__" $D/trace_rank0_step*.tsv 2>/dev/null | awk -F'\t' '{s+=$6;n++} END{if(n)printf "%.0f",s/n; else print "NA"}')
      echo "   rep$REP $M drained=${MS%%/*}ms per-iter-min=${MS##*/}ms sync=${SY}us" >> $OUT
    done
  done
done
echo "done $(date -Is)" >> $OUT
