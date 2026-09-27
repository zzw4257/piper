"""Predict a Piper pipeline step from single-GPU stage profiles (log F78).

    python experiments/pipeline_sim.py --clip notes/profile_clip.json --smolvlm notes/profile_smolvlm.json

A stage runs its tasks one at a time in schedule order (GPipe: all forwards, then all
backwards; 1F1B: `depth` warm-up forwards, then alternate). A forward waits for the
same microbatch's forward on every upstream stage, a backward for the same
microbatch's backward on every downstream stage. Durations are the profiled
fwd/bwd times at the microbatch size; no communication or host cost is modelled.
"""
import argparse
import json


def simulate(stages, preds, fwd, bwd, m, order="gpipe"):
    succs = {s: [t for t in stages if s in preds[t]] for s in stages}

    def depth(s):
        return 0 if not succs[s] else 1 + max(depth(t) for t in succs[s])

    seq = {}
    for s in stages:
        F = [("F", i) for i in range(m)]; B = [("B", i) for i in range(m)]
        if order == "gpipe":
            seq[s] = F + B
        else:
            w = min(depth(s), m); q = F[:w]; f, b = w, 0
            while b < m:
                if f < m:
                    q.append(F[f]); f += 1
                q.append(B[b]); b += 1
            seq[s] = q
    done, free, pos = {}, {s: 0.0 for s in stages}, {s: 0 for s in stages}
    while any(pos[s] < len(seq[s]) for s in stages):
        progressed = False
        for s in stages:
            if pos[s] == len(seq[s]):
                continue
            kind, i = seq[s][pos[s]]
            deps = [("F", p, i) for p in preds[s]] if kind == "F" else \
                   [("B", t, i) for t in succs[s]] + [("F", s, i)]
            if all(d in done for d in deps):
                start = max([free[s]] + [done[d] for d in deps])
                free[s] = done[(kind, s, i)] = start + (fwd[s] if kind == "F" else bwd[s])
                pos[s] += 1; progressed = True
        assert progressed, "schedule deadlocks"
    return max(free.values())


def prof(p, name, b):
    f, bw = p[name][str(b)]
    return f, bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip"); ap.add_argument("--smolvlm")
    a = ap.parse_args()
    if a.clip:
        p = json.load(open(a.clip))
        # measured on H200, fixed work 512 pairs (logs F74, F75): (threaded, routed) ms
        meas = {(1, "gpipe"): (558.7, 323.9), (2, "gpipe"): (475.9, 349.5), (4, "gpipe"): (444.6, 374.5),
                (8, "gpipe"): (443.0, 410.9), (8, "1f1b"): (436.8, 411.3)}
        print("CLIP (vision, text, head on three GPUs); predicted vs measured, ms")
        for (m, order), (mt, mr) in meas.items():
            b = 512 // m
            fw = {s: prof(p, s, b)[0] for s in ("vision", "text", "head")}
            bw = {s: prof(p, s, b)[1] for s in ("vision", "text", "head")}
            thr = simulate(["vision", "text", "head"], {"vision": [], "text": ["vision"], "head": ["text"]}, fw, bw, m, order)
            rt = simulate(["vision", "text", "head"], {"vision": [], "text": [], "head": ["vision", "text"]}, fw, bw, m, order)
            print(f"  m={m} {order:5s} threaded {thr:6.1f} vs {mt:6.1f} ({thr / mt - 1:+.0%})   routed {rt:6.1f} vs {mr:6.1f} "
                  f"({rt / mr - 1:+.0%})   ratio {thr / rt:.2f} vs {mt / mr:.2f}")
    if a.smolvlm:
        p = json.load(open(a.smolvlm))
        print("SmolVLM, fixed work 32 pairs; predicted step ms (one GPU = sum of stages)")
        for m in (1, 4, 8):
            b = 32 // m
            fw = {s: prof(p, s, b)[0] for s in p}; bw = {s: prof(p, s, b)[1] for s in p}
            one = m * (fw["vision"] + bw["vision"] + fw["decoder"] + bw["decoder"])
            row = [f"m={m}  one GPU {one:6.1f}"]
            for order in (("gpipe", "1f1b") if m > 1 else ("gpipe",)):
                two = simulate(["vision", "decoder"], {"vision": [], "decoder": ["vision"]}, fw, bw, m, order)
                four = simulate(["vision_1", "vision_2", "vision_3", "decoder"],
                                {"vision_1": [], "vision_2": ["vision_1"], "vision_3": ["vision_2"], "decoder": ["vision_3"]},
                                fw, bw, m, order)
                row.append(f"{order}: 2 stages {two:6.1f} (x{one / two:.2f})  4 stages {four:6.1f} (x{one / four:.2f})")
            print("  " + "   ".join(row))


if __name__ == "__main__":
    main()
