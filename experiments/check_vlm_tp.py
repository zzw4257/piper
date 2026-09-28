"""GPU check for F87: SmolVLM with Megatron TP on every vision and decoder layer.

    CUDA_VISIBLE_DEVICES=a,b,c python experiments/check_vlm_tp.py --data DIR

One GPU, then TP=3 (every attention and MLP split three ways, heads 12/9/3 divide),
then TP=3 with the collectives removed, which must differ. Losses over 5 Adam steps
from the released weights must match the one-GPU run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_vlm import TOL, run  # noqa: E402


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra = sys.argv[1:] + ([] if "--batch-size" in sys.argv else ["--batch-size", "16"])
    ref, err = run("vlm_single", extra, repo)
    if ref is None:
        print(f"vlm_single FAILED\n{err}")
        return 1
    print(f"{'vlm_single':16s} {ref['dir']}  losses {[round(x, 6) for x in ref['losses']]}  "
          f"median step {ref['t'] * 1e3:7.1f} ms  peak GB {ref['mem']}", flush=True)
    ok = True
    for name, must_match in (("vlm_tp3", True), ("vlm_tp3_nocomm", False)):
        r, err = run(name, extra + ["--tp", "3"], repo)
        if r is None:
            print(f"{name} FAILED\n{err}")
            ok = False
            continue
        worst = max(abs(a - b) for a, b in zip(r["losses"], ref["losses"]))
        ok &= (worst <= TOL) == must_match
        print(f"{name:16s} {r['dir']}  vs one GPU {worst:.1e} ({'must match' if must_match else 'must differ'})  "
              f"median step {r['t'] * 1e3:7.1f} ms  peak GB {r['mem']}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
