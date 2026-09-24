"""GPU check for F69: derived TP collectives and regather=false both train identically.

    CUDA_VISIBLE_DEVICES=0,2 python experiments/check_derived_and_lifetime.py

TP: shard_tensor with params (collectives derived from placements) against the
hand-ruled shard_tensor and the unsharded TP=1 run. ZeRO-3: regather=false
(parameters held from forward to backward) against shipped ZeRO-3. Same global
weights in every run (--init fixed), so losses must agree to fp32 noise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_tp_equivalence import _run  # noqa: E402

TOL = 1e-4


def losses(r):
    return [[round(x, 6) for x in m["losses"]] for m in r["metrics"]]


def worst(a, b):
    return max(abs(x - y) for la, lb in zip(a, b) for x, y in zip(la, lb))


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tp1 = _run("tp1", ["--tp", "1"], repo)
    rule = _run("tp2", [], repo)
    derived = _run("tp2_derived", [], repo)
    print("TP=1                 ", tp1["dir"], losses(tp1))
    print("TP=2 rule            ", rule["dir"], losses(rule))
    print("TP=2 derived         ", derived["dir"], losses(derived))
    d_vs_rule = worst(losses(derived), losses(rule))
    d_vs_tp1 = worst(losses(derived), losses(tp1) * 2)
    print(f"derived vs rule {d_vs_rule:.2e}   derived vs TP=1 {d_vs_tp1:.2e}")

    z = ["--tp", "1", "--stages", "3", "--init", "random"]  # fixed init slices by dp_rank as if it were a TP rank
    shipped = _run("zero3_s3_mb1", z, repo)
    keep = _run("zero3_s3_mb1_keep", z, repo)
    print("ZeRO-3 shipped       ", shipped["dir"], losses(shipped))
    print("ZeRO-3 regather=false", keep["dir"], losses(keep))
    k_vs_s = worst(losses(keep), losses(shipped))
    peaks = [(m.get("peak_memory_by_rank")) for r in (shipped, keep) for m in r["metrics"][:1]]
    print(f"regather=false vs shipped {k_vs_s:.2e}   peak memory shipped/keep: {peaks}")
    ok = d_vs_rule <= TOL and d_vs_tp1 <= TOL and k_vs_s <= TOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
