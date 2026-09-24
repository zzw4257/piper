"""Probe B2: consumer routing unchains independent branches and leaves linear models alone.

    PYTHONPATH=examples:. python experiments/probe_route.py

CPU only, lowering only (log F71).
"""
import contextlib
import io
import json
import sys
import tempfile

import torch
import torch.nn as nn

sys.path[:0] = ["examples", "."]
from src.piper import _reset_annotation_state, annotate, piper, piper_metadata  # noqa: E402
from src.schedule import derive_schedule_info, load_schedule_directives  # noqa: E402
from models.tp_mlp import TPMlp  # noqa: E402
from models.ring_attn import RingAttn  # noqa: E402

ROUTE = {"op": "route", "mode": "consumers"}
SPLIT = {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}


class TwoBranch(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.img = nn.Linear(dim, dim, bias=False)
        self.txt = nn.Linear(dim, dim, bias=False)
        self.dec = nn.Linear(dim, dim, bias=False)

    def forward(self, x_img, x_txt):
        with annotate("PP"):
            a = self.img(x_img)
        with annotate("PP"):
            b = self.txt(x_txt)
        with annotate("PP"):
            y = self.dec(a + b)
        return y


def lower(schedule, make):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(schedule, f)
    ds = load_schedule_directives(f.name)
    piper_metadata.schedule_directives = ds
    piper_metadata.schedule_info = derive_schedule_info(ds, f.name)
    piper_metadata.visualize_dag = False
    _reset_annotation_state()
    with torch.device("meta"):
        model, xs = make()
    torch._dynamo.reset()
    note = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            torch.compile(model, backend=piper, fullgraph=True)(*xs)
    except Exception as e:
        note = str(e).splitlines()[0][:200]
    return piper_metadata.training_dag, piper_metadata.per_pp_training_dags, note


def shape(dags):
    out = []
    for dag in dags:
        out.append((
            sorted((u, n.node_kind, tuple(sorted(n.tag.items(), key=str)), tuple(n.device or ()), n.stream) for u, n in dag.nodes.items()),
            sorted((e.src_uid, e.dst_uid, e.dep_kind) for e in dag.edges),
        ))
    return out


def main():
    print("== branches: three stages on devices 0, 1, 2")
    # one order per stage ties its forward to its backward (required once pp > 1, log F14)
    sched = ([{"op": "place", "filter": {"PP": i}, "devices": [i]} for i in range(3)] + [SPLIT] +
             [{"op": "order", "filters": [[{"PP": i, "PASS": "F"}], [{"PP": i, "PASS": "B"}]]} for i in range(3)])
    mk = lambda: (TwoBranch(), (torch.empty(8, 64, device="meta"), torch.empty(8, 64, device="meta")))  # noqa: E731
    for name, s in [("thread (today)", sched), ("consumers", [ROUTE] + sched)]:
        dag, per_rank, note = lower(s, mk)
        # a stage-crossing edge is compute -> send.k ... recv.k -> compute; pair them by index
        succ = {}
        for e in dag.edges:
            if e.dep_kind == "data":
                succ.setdefault(e.src_uid, []).append(e.dst_uid)
        def hop(u):
            out = []
            for v in succ.get(u, []):
                if v.startswith("send."):
                    out += succ.get("recv." + v.split(".")[1], [])
                else:
                    out.append(v)
            return out
        fwd = [(u, v) for u, n in dag.nodes.items() if n.compute_subkind == "FWD"
               for v in hop(u) if dag.nodes[v].compute_subkind == "FWD"]
        sends = sorted((u, n.node_meta.get("peer_pp_rank")) for u, n in dag.nodes.items() if n.node_kind == "SEND_COMM")
        print(f"  {name:15s} forward compute edges: {sorted(fwd)}")
        print(f"  {'':15s} image -> text edge: {('s0.seg0', 's1.seg1') in fwd}   sends: {len(sends)}   per-rank split: {note or 'ok'}")
        if name == "consumers":
            for u in ("s0.seg0", "s1.seg1", "s2.seg2"):
                print(f"  {'':15s} {u} input_sources: {dag.nodes[u].node_meta.get('input_sources')}")

    print("\n== linear models: consumer routing must not change the lowering")
    cases = [
        ("TP=2 MLP", [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
                      {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"}, SPLIT],
         lambda: (TPMlp(32, 128, 2, 1), (torch.empty(8, 32, device="meta"),))),
        ("ZeRO-3, 3 stages", [{"op": "place", "filter": {"PP": i}, "devices": [0, 1]} for i in range(3)] +
         [{"op": "replicate", "filter": {"PP": "*"}, "devices": [0, 1], "reduce_stream": "dp_stream",
           "shard_grads": True, "shard_params": True}, SPLIT],
         lambda: (TPMlp(32, 128, 1, 3), (torch.empty(8, 32, device="meta"),))),
        ("CP=4 ring, hoisted", [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1, 2, 3]},
                                {"op": "ring_exchange", "filter": {"CP": "*"}, "devices": [0, 1, 2, 3], "tensors": ["k", "v"],
                                 "stream": "cp_stream", "hoist": True, "distance": 1}, SPLIT],
         lambda: (RingAttn(64, 2, 4), tuple(torch.empty(2, 8, 64, device="meta") for _ in range(3)))),
    ]
    for name, s, mk in cases:
        _, a, na = lower(s, mk)
        _, b, nb = lower([ROUTE] + s, mk)
        same = shape(a) == shape(b)
        print(f"  {name:20s} identical per-rank DAGs: {same}   ({na or 'ok'} / {nb or 'ok'})")


if __name__ == "__main__":
    main()
