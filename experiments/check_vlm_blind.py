"""F95: blind test of the pipeline model. Placements the model priced before they were ever run.

    CUDA_VISIBLE_DEVICES=a,b,c,d python experiments/check_vlm_blind.py --data DIR

Predictions from experiments/select_vlm.py with the clean GPU-0 profile (notes/profile_smolvlm_b200.json),
written here before measuring. 48 pairs per step; losses must match one GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_vlm import TOL, run  # noqa: E402

# (schedule, vis stages, microbatches, predicted ms)
BLIND = [("vlm_2st_routed_mb1", 1, 1, 734.2), ("vlm_2st_routed_mb2", 1, 2, 688.8), ("vlm_2st_routed_mb2_1f1b", 1, 2, 622.9),
         ("vlm_2st_routed_mb4", 1, 4, 671.8), ("vlm_2st_routed_mb4_1f1b", 1, 4, 630.4),
         ("vlm_4st_routed_mb2", 3, 2, 482.7), ("vlm_4st_routed_mb4", 3, 4, 358.6)]


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra = sys.argv[1:]
    ref = {}
    ok = True
    for name, v, m, pred in BLIND:
        if m not in ref:
            r0, err = run(f"vlm_single_v3_mb{m}", extra + ["--batch-size", str(48 // m), "--vis-chunks", "3"], repo)
            if r0 is None:
                print(f"one GPU m={m} FAILED\n{err}"); return 1
            ref[m] = r0
        r, err = run(name, extra + ["--batch-size", str(48 // m), "--vis-stages", str(v)], repo)
        if r is None:
            print(f"{name} FAILED\n{err}"); ok = False; continue
        worst = max(abs(a - b) for a, b in zip(r["losses"], ref[m]["losses"]))
        ok &= worst <= TOL
        print(f"{name:26s} predicted {pred:6.1f} ms  measured {r['t'] * 1e3:6.1f} ms  error {pred / (r['t'] * 1e3) - 1:+.1%}  "
              f"vs one GPU {worst:.1e}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
