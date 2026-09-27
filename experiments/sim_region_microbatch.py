"""Per-region microbatching for SmolVLM (log F79): vision in mv microbatches, decoder in md (k = mv/md merged).

    python experiments/sim_region_microbatch.py notes/profile_smolvlm.json
"""
import json, sys
p = json.load(open(sys.argv[1]))
T = lambda s, b: p[s][str(b)]  # noqa: E731
V = ["vision_1", "vision_2", "vision_3"]


def sim(mv, md, B=32):
    k = mv // md
    bv, bd = B // mv, B // md
    stages = V + ["decoder"]
    n = {s: mv for s in V}; n["decoder"] = md
    fw = {s: T(s, bv)[0] for s in V}; bw = {s: T(s, bv)[1] for s in V}
    fw["decoder"], bw["decoder"] = T("decoder", bd)
    seq = {s: [("F", i) for i in range(n[s])] + [("B", i) for i in range(n[s])] for s in stages}
    def deps(s, kind, i):
        j = stages.index(s)
        if kind == "F":
            if j == 0: return []
            up = stages[j - 1]
            return [("F", up, i2) for i2 in range(i * k, (i + 1) * k)] if s == "decoder" else [("F", up, i)]
        d = [("F", s, i)]
        if j < 3:
            dn = stages[j + 1]
            d.append(("B", dn, i // k) if dn == "decoder" else ("B", dn, i))
        return d
    done, free, pos = {}, {s: 0.0 for s in stages}, {s: 0 for s in stages}
    while any(pos[s] < len(seq[s]) for s in stages):
        prog = False
        for s in stages:
            if pos[s] == len(seq[s]): continue
            kind, i = seq[s][pos[s]]
            dd = deps(s, kind, i)
            if all(x in done for x in dd):
                st = max([free[s]] + [done[x] for x in dd])
                free[s] = done[(kind, s, i)] = st + (fw[s] if kind == "F" else bw[s]); pos[s] += 1; prog = True
        assert prog
    return max(free.values())


one = sum(sum(T(s, 32)) for s in V) + sum(T("decoder", 32))
print(f"one GPU, m=1: {one:.1f} ms")
for mv, md in [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (4, 1), (8, 1), (8, 2), (16, 2), (16, 4), (2, 1), (4, 2)]:
    t = sim(mv, md)
    print(f"  vision m={mv:2d}, decoder m={md:2d}: {t:6.1f} ms  x{one / t:.2f}" + ("   (one m for all)" if mv == md else ""))
