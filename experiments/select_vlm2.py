"""F98: SmolVLM2-2.2B on DocVQA tiles (17 per example): which placement on four GPUs? Model first, then measure.

    python experiments/select_vlm2.py PROFILE.json [--write-schedules]

6 examples (102 tiles) per step, m microbatches. Candidates:
  hetero k+s: vision tower in k chunks side by side on GPUs 0..k-1 (shared weights, F83), embedding
              and decoder in s pipeline stages on GPUs k..3;
  pipeline v+s: vision tower in v stages, decoder in s stages, one chain;
  one GPU.
Priced with the F78 model from single-GPU profiles (experiments/profile_smolvlm2.py): vision cost
linear in tiles, decoder linear in examples, a decoder stage = decoder / s. No communication.
"""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline_sim import simulate

B, T = 6, 17
SP = "examples/base-schedules/"


def fit(points):
    xs, ys = zip(*points); n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return lambda x: my + a * (x - mx)


def order(pps, m, warm, pas):
    fs = [[{"PP": q, "PASS": "F", "MB": i} for q in pps] for i in range(m)]
    bs = [[{"PP": q, "PASS": "B", "MB": i} for q in reversed(pps)] for i in range(m)]
    if pas == "gpipe":
        seq = fs + bs
    else:
        seq, f, b = fs[:warm], min(warm, m), 0
        while b < m:
            if f < m:
                seq.append(fs[f]); f += 1
            seq.append(bs[b]); b += 1
    return {"op": "order", "filters": seq}


def schedule(kind, a, s, m, pas):
    """kind 'het': a vision chunks; 'pipe': a vision stages. Returns (name, directives, vis_stages, vis_chunks)."""
    d = [{"op": "route", "mode": "consumers"}]
    for c in range(a):
        d.append({"op": "place", "filter": {"PP": c}, "devices": [c]})
    d.append({"op": "place", "filter": {"PP": a}, "devices": [a]})           # embedding with decoder stage 0
    for j in range(s):
        d.append({"op": "place", "filter": {"PP": a + 1 + j}, "devices": [a + j]})
    d.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": m})
    if kind == "het":
        d += [order([c], m, s, pas) for c in range(a)]
    else:
        d.append(order([0], m, a - 1 + s, pas)) if a == 1 else None
        if a > 1:
            d += [order([c], m, a - 1 - c + s, pas) for c in range(a)]
    d.append(order([a, a + 1], m, s - 1, pas))
    d += [order([a + 1 + j], m, s - 1 - j, pas) for j in range(1, s)]
    name = f"vlm2_{kind}{a}_{s}_mb{m}" + ("" if pas == "gpipe" else "_1f1b")
    return name, d, (1 if kind == "het" else a), (a if kind == "het" else 1)


def main():
    p = json.load(open(sys.argv[1]))
    vis = {k: fit([(int(n), v[i]) for n, v in p["vision"].items()]) for k, i in (("f", 0), ("b", 1))}
    dec = {k: fit([(int(key.split("_b")[1]), v[i]) for key, v in p["decoder"].items() if key.startswith(f"T{T}_")]) for k, i in (("f", 0), ("b", 1))}
    rows = []
    for m in (1, 2, 3, 6):
        e = B // m; tiles = e * T
        for pas in ("gpipe", "1f1b"):
            if m == 1 and pas == "1f1b":
                continue
            if pas == "gpipe":
                rows.append(("one GPU", 0, 0, m, pas, m * (vis["f"](tiles) + vis["b"](tiles) + dec["f"](e) + dec["b"](e)), None))
            for kind in ("het", "pipe"):
                for a in (1, 2, 3):
                    s = 4 - a
                    if kind == "het" and (tiles % a or a == 1):
                        continue
                    fw, bw, pre, st = {}, {}, {}, []
                    if kind == "het":
                        for c in range(a):
                            n = f"v{c}"; st.append(n); pre[n] = []; fw[n] = vis["f"](tiles // a); bw[n] = vis["b"](tiles // a)
                        up = [f"v{c}" for c in range(a)]
                    else:
                        for c in range(a):
                            n = f"v{c}"; st.append(n); pre[n] = [f"v{c-1}"] if c else []; fw[n] = vis["f"](tiles) / a; bw[n] = vis["b"](tiles) / a
                        up = [f"v{a-1}"]
                    for j in range(s):
                        n = f"d{j}"; st.append(n); pre[n] = up if j == 0 else [f"d{j-1}"]; fw[n] = dec["f"](e) / s; bw[n] = dec["b"](e) / s
                    t = simulate(st, pre, fw, bw, m, pas)
                    rows.append((f"{kind} {a}+{s}", a, s, m, pas, t, kind))
    rows.sort(key=lambda r: r[5])
    for i, r in enumerate(rows[:14], 1):
        print(f"{i:3d} {r[0]:10s} m={r[3]} {r[4]:5s} predicted {r[5]:8.1f} ms")
    if "--write-schedules" in sys.argv:
        out = []
        for r in rows:
            if r[6] is None:
                continue
            name, d, vs, vc = schedule(r[6], r[1], r[2], r[3], r[4])
            json.dump(d, open(SP + name + ".json", "w"), indent=1)
            out.append((name, vs, vc, r[2], r[3], round(r[5], 1)))
        json.dump(out, open("notes/vlm2_candidates.json", "w"), indent=1)
        print(f"wrote {len(out)} schedules; notes/vlm2_candidates.json")


if __name__ == "__main__":
    main()
