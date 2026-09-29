"""F99: every-layer TP on real VLM structures, derived (layout) against written (shard_tensor). CPU only.

    PYTHONPATH=examples:. python experiments/layout_vlm.py

SmolVLM-256M at TP=3 and SmolVLM2-2.2B at TP=2 (two vision and two decoder layers each, real widths).
layout gets only which projections are column- and row-parallel; the DAGs are compared node for node.
"""
import json, sys, time
import torch
sys.path[:0] = ["experiments", "examples", "."]
import probe_route as p  # noqa: E402
from models.smolvlm import SmolVLM, CFG_SMOLVLM2  # noqa: E402

PARAMS = {"q_proj": "colwise", "k_proj": "colwise", "v_proj": "colwise", "fc1": "colwise", "gate_proj": "colwise",
          "up_proj": "colwise", "out_proj": "rowwise", "o_proj": "rowwise", "fc2": "rowwise", "down_proj": "rowwise"}


def run(name, cfg, tp, inputs, offset):
    devs = list(range(tp))
    base = [{"op": "route", "mode": "consumers"}] + [{"op": "place", "filter": {"PP": i}, "devices": devs} for i in range(3)]
    split = [{"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}]
    st = base + [{"op": "shard_tensor", "filter": {"TP": "*"}, "devices": devs, "stream": "tp_stream"}] + split
    ly = base + [{"op": "layout", "filter": {"TP": "*"}, "devices": devs, "axis": "tp", "params": PARAMS, "stream": "tp_stream"}] + split
    make = lambda: (SmolVLM(offset, config=dict(cfg, v_layers=2, t_layers=2, tp=tp)).float(), inputs)  # noqa: E731
    t0 = time.time(); _, a, na = p.lower(st, make); t1 = time.time(); _, b, nb = p.lower(ly, make); t2 = time.time()
    tp_nodes = sum(n.node_kind == "TP_COMM" for n in a[0].nodes.values()) if not na else None
    same = (not na and not nb and p.shape(a) == p.shape(b))
    print(f"{name}: shard_tensor {na or 'ok'} ({tp_nodes} TP all-reduces, {t1 - t0:.1f}s) | layout {nb or 'ok'} ({t2 - t1:.1f}s) | "
          f"node for node: {'identical' if same else 'DIFFERENT'}", flush=True)
    return same


if __name__ == "__main__":
  ok = run("SmolVLM-256M, TP=3", {}, 3,
         (torch.empty(2, 3, 512, 512, device="meta"), torch.zeros(2, 80, dtype=torch.long, device="meta")), 5)
  ok &= run("SmolVLM2-2.2B, TP=2", CFG_SMOLVLM2, 2,
          (torch.empty(2 * 2, 3, 384, 384, device="meta"), torch.zeros(2, 200, dtype=torch.long, device="meta"),
           torch.zeros(2, 2 * 81, dtype=torch.long, device="meta")), None)
  print("PASS" if ok else "FAIL")
