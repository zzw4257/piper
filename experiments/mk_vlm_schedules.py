"""Schedules for SmolVLM placements (log F77). Stages = device groups; each lists its PP regions."""
import json
sp = "examples/base-schedules/"
LAYOUTS = {  # name: (vis_stages, per-device PP regions); regions: vision..., embedding, decoder
    "2st": (1, [[0], [1, 2]]),
    "4st": (3, [[0], [1], [2], [3, 4]]),
}


def order(pps, m, warm, pas):
    grp = lambda k, p: [{"PP": q, "PASS": p, "MB": k} for q in (pps if p == "F" else pps[::-1])]  # noqa: E731
    fs, bs = [grp(k, "F") for k in range(m)], [grp(k, "B") for k in range(m)]
    if pas == "gpipe":
        seq = fs + bs
    else:
        seq, f, b = fs[:warm], min(warm, m), 0
        while b < m:
            if f < m:
                seq.append(fs[f]); f += 1
            seq.append(bs[b]); b += 1
    return {"op": "order", "filters": seq}


for name, (v, devs) in LAYOUTS.items():
    for route in (False, True):
        for m in (1, 2, 4, 8):
            for pas in ("gpipe", "1f1b"):
                if m == 1 and pas == "1f1b":
                    continue
                d = ([{"op": "route", "mode": "consumers"}] if route else [])
                d += [{"op": "place", "filter": {"PP": q}, "devices": [i]} for i, pps in enumerate(devs) for q in pps]
                d.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": m})
                d += [order(pps, m, len(devs) - 1 - i, pas) for i, pps in enumerate(devs)]
                tag = f"vlm_{name}_{'routed' if route else 'threaded'}_mb{m}" + ("" if pas == "gpipe" else "_1f1b")
                json.dump(d, open(sp + tag + ".json", "w"), indent=1)
single = [{"op": "route", "mode": "consumers"}] + [{"op": "place", "filter": {"PP": q}, "devices": [0]} for q in range(5)]
for m in (1, 2, 4, 8):
    json.dump(single + [{"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": m}],
              open(sp + f"vlm_single_v3_mb{m}.json", "w"), indent=1)
