"""F98: SmolVLM2-2.2B placements on four GPUs, measured against the model's prices (DocVQA, 17 tiles).

    CUDA_VISIBLE_DEVICES=a,b,c,d python experiments/check_vlm2.py --data DIR/T17 NAME [NAME ...]

6 examples per step at m microbatches (each microbatch sees the same 6/m examples: `split` does not slice, F74); losses must match one GPU at the same m. Names come from
notes/vlm2_candidates.json (experiments/select_vlm2.py), which also holds each prediction.
"""
import glob, json, os, re, statistics, subprocess, sys

TOL = 1e-4


def run(schedule, extra, repo):
    cmd = [sys.executable, "examples/test_harness.py", "--test-file", "examples/test_smolvlm.py",
           "--base-schedule", f"examples/base-schedules/{schedule}.json", "--schedule", "custom", *extra]
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=800)
    if p.returncode:
        return None, p.stdout[-1500:] + p.stderr[-1500:]
    d = os.path.join(repo, re.findall(r"out/\d{8}_\d{6}", p.stdout)[-1])
    m = json.load(open(sorted(glob.glob(f"{d}/branches_metrics_dp*.json"))[0]))
    return m, None


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = sys.argv[1:]
    data = args[args.index("--data") + 1]
    names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--data", "--cands"))]
    names = [n for n in names if not n.startswith("/var/tmp")]
    cands = args[args.index("--cands") + 1] if "--cands" in args else "notes/vlm2_candidates.json"
    cand = {c[0]: c for c in json.load(open(os.path.join(repo, cands)))}
    common = ["--data", data, "--iters", "4", "--warmup", "1"]
    refs = {}
    for n in names:
        if not n.startswith("single"):
            continue
        m = int(n[len("single"):])
        # one microbatch's forward then its backward (gradient accumulation): all-forwards-first
        # would hold every microbatch's activations and does not fit (log F98)
        seq = []
        for i in range(m):
            seq.append([{"PP": q, "PASS": "F", "MB": i} for q in range(3)])
            seq.append([{"PP": q, "PASS": "B", "MB": i} for q in reversed(range(3))])
        d = [{"op": "route", "mode": "consumers"}] + [{"op": "place", "filter": {"PP": i}, "devices": [0]} for i in range(3)] + \
            [{"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": m}, {"op": "order", "filters": seq}]
        json.dump(d, open(os.path.join(repo, f"examples/base-schedules/vlm2_single_mb{m}.json"), "w"), indent=1)
        r, err = run(f"vlm2_single_mb{m}", common + ["--batch-size", str(6 // m)], repo)
        if r is None:
            print(f"single m={m} FAILED", err[-600:], flush=True); continue
        refs[m] = r
        print(f"{'one GPU, m=' + str(m):34s} measured {statistics.median(r['iter_times']) * 1e3:8.1f} ms  peak GB {max(r['peak_mem_gb'].values()):.1f}", flush=True)
    ok = True
    for n in names:
        if n.startswith("single"):
            continue
        name, vs, vc, s, m, pred = cand[n]
        r, err = run(name, common + ["--batch-size", str(6 // m), "--vis-stages", str(vs), "--vis-chunks", str(vc), "--dec-stages", str(s)], repo)
        if r is None:
            print(f"{name} FAILED\n{err}", flush=True); ok = False; continue
        t = statistics.median(r["iter_times"]) * 1e3
        note = ""
        if m in refs:
            worst = max(abs(a - b) for a, b in zip(r["losses"], refs[m]["losses"])); ok &= worst <= TOL; note = f"vs one GPU (m={m}) {worst:.1e}"
        print(f"{name:34s} predicted {pred:8.1f} ms  measured {t:8.1f} ms  error {pred / t - 1:+.1%}  peak GB {max(r['peak_mem_gb'].values()):.1f}  {note}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
