#!/usr/bin/env bash
# H200 counterparts of the B200 sweeps, run strictly in sequence in one process
# so no cross-waiting is possible (the chained version deadlocked on mutually
# matching pgrep patterns).
set -u
R=/data/home/Licheng/piper-tp; V=/data/home/Licheng/piper-venv/bin; T=/data/home/Licheng/ray_tmp
export PATH=$V:$PATH; cd $R; mkdir -p logs $T
echo "=== sequential H200 chain start $(date -Is) ===" >> logs/h200_all.txt
experiments/run_law.sh      $R $V h200                >> logs/h200_all.txt 2>&1
echo "-- law rc=$? $(date -Is)" >> logs/h200_all.txt
experiments/run_xfer.sh     $R $V h200                >> logs/h200_all.txt 2>&1
echo "-- xfer rc=$? $(date -Is)" >> logs/h200_all.txt
experiments/run_hostbound.sh $R $V h200 $T            >> logs/h200_all.txt 2>&1
echo "-- hostbound rc=$? $(date -Is)" >> logs/h200_all.txt
echo "=== done $(date -Is) ===" >> logs/h200_all.txt
