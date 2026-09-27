"""4-stage SmolVLM schedules with the vision tower in k chunks (log F80).

Regions: vision chunk c, third s -> PP c*3+s on device s; embedding PP 3k, decoder PP 3k+1 on device 3.
GPipe per device: every forward (microbatch-major, chunk-minor), then every backward.
"""
import json
sp = "examples/base-schedules/"
for k, m in ((4, 2), (2, 4), (8, 1), (4, 1)):
    d = [{"op": "route", "mode": "consumers"}]
    d += [{"op": "place", "filter": {"PP": c * 3 + s}, "devices": [s]} for c in range(k) for s in range(3)]
    d += [{"op": "place", "filter": {"PP": 3 * k + q}, "devices": [3]} for q in range(2)]
    d.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": m})
    for s in range(3):
        F = [[{"PP": c * 3 + s, "PASS": "F", "MB": i}] for i in range(m) for c in range(k)]
        # backward last chunk first: routing hands pixel_values chunk to chunk, so chunk c+1 feeds c
        B = [[{"PP": c * 3 + s, "PASS": "B", "MB": i}] for i in range(m) for c in reversed(range(k))]
        d.append({"op": "order", "filters": F + B})
    F = [[{"PP": 3 * k, "PASS": "F", "MB": i}, {"PP": 3 * k + 1, "PASS": "F", "MB": i}] for i in range(m)]
    B = [[{"PP": 3 * k + 1, "PASS": "B", "MB": i}, {"PP": 3 * k, "PASS": "B", "MB": i}] for i in range(m)]
    d.append({"op": "order", "filters": F + B} if len(F + B) > 1 else {"op": "order", "filters": F + B + F[:0]})
    json.dump(d, open(sp + f"vlm_4st_routed_c{k}_mb{m}.json", "w"), indent=1)
