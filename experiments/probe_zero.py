"""Does a TP region coexist with a ZeRO-3 region on the same device group?

Both consume the dp dimension: ZeRO shards parameters by dp_rank, TP shards them
in the model by TP rank, and Piper reports the place-group size as dp_degree for
both. They are applied to different regions here, which is the only composition
shard_tensor permits (it refuses replicate on its own region, F6).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import src.directives as d
from src.dag import build_training_dag
from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate

DEV = [0, 1]


class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16, bias=False)      # PP=0: plain, gets ZeRO-3
        self.up = nn.Linear(16, 32, bias=False)     # PP=1 / TP=0
        self.down = nn.Linear(32, 16, bias=False)
        self.b = nn.Linear(16, 16, bias=False)      # PP=1 tail

    def forward(self, x):
        with annotate("PP"):
            x = self.a(x)
        with annotate("PP"):
            with annotate("TP"):
                x = self.down(F.gelu(self.up(x)))
            return self.b(x)


got = {}
def be(gm, _):
    got["segs"] = split_gm_by_annotations(gm)[1]; return gm.forward
with torch.device("meta"):
    m = Mixed().to(torch.float32)
_reset_annotation_state(); torch._dynamo.reset()
torch.compile(m, backend=be, fullgraph=True)(torch.empty(2, 16, device="meta"))
print("segments:", [s.tag for s in got["segs"]])

dag = build_training_dag(got["segs"])
sched = [
    {"op": "place", "filter": {"PP": 0}, "devices": DEV},
    {"op": "place", "filter": {"PP": 1}, "devices": DEV},
    {"op": "replicate", "filter": {"PP": 0}, "devices": DEV, "shard_params": True},
    {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV, "stream": "tp_stream"},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
]
try:
    d.apply_schedule_directives(dag, sched)
    kinds = {}
    for n in dag.nodes.values():
        kinds[n.node_kind] = kinds.get(n.node_kind, 0) + 1
    print("ACCEPTED:", dict(sorted(kinds.items())))
    for u, n in sorted(dag.nodes.items()):
        if n.node_kind.endswith("_COMM"):
            print(f"  {u:<22s} {n.node_kind:<20s} tag={ {k:v for k,v in n.tag.items() if k!='MB'} }")
    zero_flags = {u: {k: v for k, v in n.node_meta.items()
                      if k.startswith("zero_") or k == "apply_zero"}
                  for u, n in dag.nodes.items() if n.node_kind == "COMPUTE"}
    print("  zero-related metadata:", {u: f for u, f in zero_flags.items() if f})
except ValueError as e:
    print(f"REJECTED: {str(e)[:200]}")
