#!/usr/bin/env bash
# Redo F60's correctness matrix with the fp64 parameter checksum instead of the
# loss. The EP cell in particular compared a bf16 loss that is 7.625 whatever
# the model computed; the checksum carries every gradient ever applied.
set -u
R=$1; V=$2; T=$3; TAG=$4
export PATH=$V:$PATH; cd $R; mkdir -p logs
OUT=logs/checksum_${TAG}.txt
echo "=== start $(date -Is) host=$(hostname) ===" >> $OUT
pick() { local n=$1 ok=""; for s in 1 2 3; do
  ok=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '($3-$2)/1024>40 {print $1}' | head -$n | paste -sd,)
  [ "$(echo $ok | tr ',' '\n' | grep -c .)" -lt $n ] && return 1; sleep 15; done; echo "$ok"; }
t0=$(date +%s)
while :; do C4=$(pick 4) && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no cards" >> $OUT; exit 1; }; sleep 60; done
C2=$(echo $C4 | cut -d, -f1,2)
echo "cards4=$C4 cards2=$C2" >> $OUT

dump() {  # dump <run dir> <glob>
  python - "$1" "$2" <<'PY' 2>/dev/null
import glob, json, sys
fs = sorted(glob.glob(f"{sys.argv[1]}/{sys.argv[2]}"))
if not fs: print("NO METRICS"); raise SystemExit
cs = []
for f in fs:
    for c in json.load(open(f)).get("param_checksums", []) or []:
        cs.append((c["rank"], c["n_params"], c["sum"], c["sumsq"]))
if not cs: print("NO CHECKSUM"); raise SystemExit
for r, n, s, q in sorted(set(cs)):
    print(f"      rank{r} n={n} sum={s:.12e} sumsq={q:.12e}")
PY
}

echo "--- EP: shipped Qwen, pp2_dp2_ep2, four cards" >> $OUT
for M in device narrow defer; do
  echo "  mode=$M" >> $OUT
  RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C4 timeout 2400 python examples/test_harness.py \
    --test-file examples/test_qwen.py --base-schedule examples/base-schedules/pp2_dp2_ep2.json \
    --schedule 1f1b --ranks 2 --mbs 4 --no-use-inductor --temp-dir $T 2>&1 \
    | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  dump "$RUN" "qwen_metrics_dp*.json" >> $OUT
done

echo "--- ZeRO-3: three stages, shard_params, two cards" >> $OUT
for M in device narrow defer; do
  echo "  mode=$M" >> $OUT
  RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C2 timeout 2400 python examples/test_harness.py \
    --test-file examples/test_tp_mlp.py --base-schedule examples/base-schedules/zero3_s3_mb1.json \
    --schedule custom --stages 3 --tp 1 --init random --dim 512 --hidden 2048 \
    --warmup 1 --iters 4 --temp-dir $T 2>&1 | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  dump "$RUN" "tp_metrics_dp*.json" >> $OUT
done

echo "--- CP=4 ring, four cards" >> $OUT
for M in device narrow defer; do
  echo "  mode=$M" >> $OUT
  RUN=$(PIPER_SYNC_MODE=$M CUDA_VISIBLE_DEVICES=$C4 timeout 2400 python examples/test_harness.py \
    --test-file examples/test_ring_attn.py --base-schedule examples/base-schedules/cp4_ring_dp.json \
    --schedule custom --steps 4 --seq 2048 --dim 512 --heads 8 --batch-size 4 \
    --warmup 1 --iters 4 --temp-dir $T 2>&1 | grep -oE "out/[0-9]{8}_[0-9]{6}" | tail -1)
  dump "$RUN" "cp_metrics_dp*.json" >> $OUT
done
echo "done $(date -Is)" >> $OUT
