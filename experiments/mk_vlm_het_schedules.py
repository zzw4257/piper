"""SmolVLM with the vision tower data-parallel on its own GPUs (log F84).

Regions: vision chunk c -> PP c on GPU c (c < k); embedding PP k and decoder PP k+1 on GPU k.
The chunks share the vision weights; the runtime sums their gradients (F83).
"""
import json
sp = "examples/base-schedules/"


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


k = 3
for m in (1, 2, 4):
    for pas in ("gpipe", "1f1b"):
        if m == 1 and pas == "1f1b":
            continue
        d = [{"op": "route", "mode": "consumers"}]
        d += [{"op": "place", "filter": {"PP": c}, "devices": [c]} for c in range(k)]
        d += [{"op": "place", "filter": {"PP": k + q}, "devices": [k]} for q in range(2)]
        d.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": m})
        d += [order([c], m, 1, pas) for c in range(k)]
        d.append(order([k, k + 1], m, 0, pas))
        json.dump(d, open(sp + f"vlm_het{k}_mb{m}" + ("" if pas == "gpipe" else "_1f1b") + ".json", "w"), indent=1)
d = [{"op": "route", "mode": "consumers"}] + [{"op": "place", "filter": {"PP": i}, "devices": [0]} for i in range(5)]
json.dump(d + [{"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2}], open(sp + "vlm_single_v3_mb2.json", "w"), indent=1)
