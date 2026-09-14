#!/usr/bin/env bash
# The two paths the sync modes had not covered: expert parallelism (the shipped
# Qwen example) and ZeRO-3 (which used to return before the sync-mode branch).
# Losses must be identical across modes; that is the whole test.
set -u
R=$1; V=$2; T=$3; TAG=$4
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/epzero_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local n=$1 ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '($3-$2)/1024>40 {print $1}' | head -$n | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C4=$(pick 4) && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
C2=$(echo $C4 | cut -d, -f1,2)
echo "cards4=$C4 cards2=$C2 load: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd,)" >> $OUT

echo "--- EP: shipped Qwen example, pp2_dp2_ep2, four cards" >> $OUT
for M in device narrow defer; do
  RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C4 timeout 2400 python examples/test_harness.py \
    --test-file examples/test_qwen.py --base-schedule examples/base-schedules/pp2_dp2_ep2.json \
    --schedule 1f1b --ranks 2 --mbs 4 --no-use-inductor --temp-dir $T 2>&1 | tee logs/ez_ep_$M.log \
    | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  echo -n "  $M rc=? run=$RUN losses=" >> $OUT
  python - "$RUN" <<'PY' >> $OUT 2>&1
import glob, json, sys
fs = sorted(glob.glob(f"{sys.argv[1]}/qwen_metrics_dp*.json"))
if not fs: print("NO METRICS"); raise SystemExit
for f in fs[:1]:
    d = json.load(open(f)); print([round(x, 6) for x in d.get("losses", [])][:6])
PY
  grep -E "Error|Traceback" logs/ez_ep_$M.log | tail -1 | sed 's/^/    /' >> $OUT
done

echo "--- ZeRO-3: three stages, shard_params, two cards" >> $OUT
for M in device narrow defer; do
  RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C2 timeout 2400 python examples/test_harness.py \
    --test-file examples/test_tp_mlp.py --base-schedule examples/base-schedules/zero3_s3_mb1.json \
    --schedule custom --stages 3 --tp 1 --init random --dim 512 --hidden 2048 \
    --warmup 1 --iters 4 --temp-dir $T 2>&1 | tee logs/ez_z3_$M.log \
    | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  echo -n "  $M run=$RUN losses=" >> $OUT
  python - "$RUN" <<'PY' >> $OUT 2>&1
import glob, json, sys
fs = sorted(glob.glob(f"{sys.argv[1]}/tp_metrics_dp*.json"))
if not fs: print("NO METRICS"); raise SystemExit
print([round(x, 6) for x in json.load(open(fs[0])).get("losses", [])])
PY
  grep -E "Error|Traceback" logs/ez_z3_$M.log | tail -1 | sed 's/^/    /' >> $OUT
done
echo "done $(date -Is)" >> $OUT
