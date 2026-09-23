"""Probe B1: does a model with two independent branches lower as parallel stages?

Two encoders that share no data feed one decoder. Each sits in its own PP scope and is
placed on its own device. The question is whether the training DAG keeps the branches
independent, so the text encoder can start as soon as its own input is ready, or adds an
edge between them.

Prediction, written before the first run: the branches are chained.
split_gm_by_annotations builds contiguous segments in trace order, and a value crossing
several segments is threaded through the intermediate ones as a bare placeholder. The
image encoder's output would then reach the decoder by way of the text encoder's segment,
which adds a stage 0 -> stage 1 data edge, and a transfer onto device 1, that the model
does not have.

    PYTHONPATH=examples:. python experiments/probe_branches.py
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

SCHEDULE = [
    {"op": "place", "filter": {"PP": 0}, "devices": [0]},
    {"op": "place", "filter": {"PP": 1}, "devices": [1]},
    {"op": "place", "filter": {"PP": 2}, "devices": [2]},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
]


class TwoBranch(nn.Module):
    def __init__(self, dim):
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


def lower(schedule, dim=64, batch=8):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(schedule, f)
    directives = load_schedule_directives(f.name)
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, f.name)
    piper_metadata.visualize_dag = False
    _reset_annotation_state()
    with torch.device("meta"):
        model = TwoBranch(dim)
        x = (torch.empty(batch, dim, device="meta"), torch.empty(batch, dim, device="meta"))
    torch._dynamo.reset()
    note = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            torch.compile(model, backend=piper, fullgraph=True)(*x)
    except Exception as e:  # the per-stage split needs an order directive when pp > 1
        msg = [l for l in str(e).splitlines() if "ValueError" in l or "expected" in l]
        note = (msg[0] if msg else str(e).splitlines()[0])[:200]
    return piper_metadata.training_dag, note


def main():
    dag, note = lower(SCHEDULE)
    fwd = sorted(u for u, n in dag.nodes.items() if n.node_kind == "COMPUTE" and n.compute_subkind == "FWD")
    stage = {u: dag.nodes[u].tag.get("PP") for u in fwd}
    print("forward compute nodes:", {u: (stage[u], dag.nodes[u].device) for u in fwd})
    for u in fwd:
        outs = (dag.nodes[u].node_meta.get("a2a_boundary_after") or {}).get("outputs", [])
        if outs:
            print(f"  {u} boundary outputs:", [(o.get("src_name"), "forwarded" if o.get("forwarded") else "produced") for o in outs])

    def reaches(a, b, kinds=("data",)):
        succ = {}
        for e in dag.edges:
            if e.dep_kind in kinds:
                succ.setdefault(e.src_uid, []).append(e.dst_uid)
        for u in dag.nodes:                  # a send and its recv carry no edge; pair by index
            if u.startswith("send.") and "recv." + u[5:] in dag.nodes:
                succ.setdefault(u, []).append("recv." + u[5:])
        seen, stack = set(), [a]
        while stack:
            x = stack.pop()
            for y in succ.get(x, []):
                if y == b:
                    return True
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return False

    img, txt, dec = (next(u for u in fwd if stage[u] == s) for s in (0, 1, 2))
    chained = reaches(img, txt)
    print(f"\nimage encoder ({img}) reaches text encoder ({txt}) along data edges: {chained}")
    x_txt_via_img = any(o.get("forwarded") and "txt" in (o.get("src_name") or "")
                        for o in (dag.nodes[img].node_meta.get("a2a_boundary_after") or {}).get("outputs", []))
    print(f"text encoder's own input threaded through the image encoder's segment: {x_txt_via_img}")
    print(f"image encoder reaches decoder ({dec}): {reaches(img, dec)}")
    sends = [(u, n.tag.get("PP"), n.node_meta.get("peer_pp_rank")) for u, n in dag.nodes.items() if n.node_kind == "SEND_COMM" and n.tag.get("PASS") == "F"]
    print("forward sends (node, from stage, to stage):", sorted(sends))
    if note:
        print("per-stage split:", note)
    print("\nverdict:", "branches CHAINED through the text encoder" if chained else "branches PARALLEL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
