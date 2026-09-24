"""Derive a region's boundary collectives from DTensor placements, then compare
them with the comm nodes that Piper's directives insert.

For each annotated region the user states only how parameters (and the batch)
are placed. DTensor traces the region under a fake 2-rank process group, and the
collectives that appear in its aten graphs are mapped onto the region's edges in
the baseline DAG (the lowering with no comm directive). The prediction is then
compared node by node with the lowering that uses the hand-written directive.

    PYTHONPATH=examples:. python experiments/derive_boundary_comm.py

CPU only. Covers TP (vs shard_tensor) and ZeRO-3 (vs replicate(shard_params)).
"""
import json
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.testing._internal.distributed.fake_pg import FakeStore

sys.path[:0] = ["examples", "."]
from src.piper import _reset_annotation_state, piper  # noqa: E402
from src.schedule import derive_schedule_info, load_schedule_directives  # noqa: E402
from src.state import piper_metadata  # noqa: E402
from models.tp_mlp import TPMlp  # noqa: E402

DIM, HIDDEN, BATCH = 32, 128, 8
WORLD = 2


# ----------------------------------------------------------------------------- Piper side
def lower(schedule: list, stages: int):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(schedule, f)
    directives = load_schedule_directives(f.name)
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, f.name)
    piper_metadata.visualize_dag = False
    _reset_annotation_state()
    with torch.device("meta"):
        model = TPMlp(DIM, HIDDEN, 1, stages)  # full-width model: placement is the only input
        x = torch.empty(BATCH, DIM, device="meta")
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(x)
    dags = piper_metadata.per_pp_training_dags
    assert len(dags) == 1
    return dags[0]


def comm_nodes(dag, kinds):
    """(kind, pass, producer, consumer) for every comm node of the given kinds."""
    out = set()
    for uid, n in dag.nodes.items():
        if n.node_kind not in kinds:
            continue
        preds = [e.src_uid for e in dag.edges if e.dst_uid == uid and e.dep_kind == "data"
                 and dag.nodes[e.src_uid].node_kind == "COMPUTE"]
        succs = [e.dst_uid for e in dag.edges if e.src_uid == uid and e.dep_kind == "data"
                 and dag.nodes[e.dst_uid].node_kind == "COMPUTE"]
        consumer = n.node_meta.get("compute_uid") or (succs[0] if succs else None)
        producer = n.node_meta.get("bwd_uid") or n.node_meta.get("source_uid") or (preds[0] if preds else None)
        out.add((n.node_kind, n.tag.get("PASS"), producer, consumer))
    return out


# ----------------------------------------------------------------------------- DTensor side
def aten_collectives(fn, x):
    """Collectives in the forward and backward aten graphs, and which graph outputs they reach."""
    graphs = []

    def cap(tag):
        def compiler(g, example_inputs):
            graphs.append((tag, g))
            return g.forward
        return compiler

    torch._dynamo.reset()
    y = torch.compile(fn, backend=aot_autograd(fw_compiler=cap("F"), bw_compiler=cap("B")), fullgraph=True)(x)
    (y.to_local() if isinstance(y, DTensor) else y).sum().backward()
    found = []
    for tag, g in graphs:
        out_node = next(n for n in g.graph.nodes if n.op == "output")
        outs = list(out_node.args[0])
        for n in g.graph.nodes:
            t = str(n.target)
            if "c10d_functional" not in t or "wait" in t:
                continue
            reach, frontier, seen = set(), [n], set()
            while frontier:
                m = frontier.pop()
                if m in seen:
                    continue
                seen.add(m)
                for u in m.users:
                    if u.op == "output":
                        reach |= {i for i, o in enumerate(outs) if o is m}
                    else:
                        frontier.append(u)
            found.append((tag, t.split(".")[1], reach, outs))
    return found


def derive_tp(mesh):
    """Region up -> gelu -> down, placed column- then row-parallel."""
    class Region(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(DIM, HIDDEN, bias=False)
            self.down = nn.Linear(HIDDEN, DIM, bias=False)

        def forward(self, x):
            return self.down(F.gelu(self.up(x)))

    r = Region()
    parallelize_module(r, mesh, {"up": ColwiseParallel(), "down": RowwiseParallel()})
    derived = []
    for tag, kind, reach, outs in aten_collectives(r, torch.randn(BATCH, DIM, requires_grad=True)):
        if tag == "F":
            derived.append((kind, "F", "region output" if 0 in reach else "saved for backward"))
        else:
            shapes = [tuple(o.meta["val"].shape) if o is not None and "val" in o.meta else None for o in outs]
            where = ("gradient of region input" if any(shapes[i] == (BATCH, DIM) for i in reach)
                     else "parameter gradient")
            derived.append((kind, "B", where))
    return derived


def derive_zero3(mesh, n_linears):
    """A region whose weights are stored Shard(0), with the batch Shard(0) as in data parallelism."""
    weights = [nn.Parameter(distribute_tensor(torch.randn(DIM, DIM), mesh, [Shard(0)])) for _ in range(n_linears)]

    def region(x):
        h = DTensor.from_local(x, mesh, [Shard(0)])
        for w in weights:
            h = F.linear(h, w.redistribute(mesh, [Replicate()]))
        return h

    x = torch.randn(BATCH // WORLD, DIM, requires_grad=True)
    return [(kind, tag) for tag, kind, _, _ in aten_collectives(region, x)]


# ----------------------------------------------------------------------------- compare
def region_out_edges(dag, region_uid):
    """Data edges leaving a region's FWD node and its BWD node in the baseline DAG."""
    fwd_out = [(region_uid, e.dst_uid) for e in dag.edges
               if e.src_uid == region_uid and e.dep_kind == "data"
               and dag.nodes[e.dst_uid].compute_subkind == "FWD"]
    bwd = region_uid + ".bwd"
    bwd_out = [(bwd, e.dst_uid) for e in dag.edges
               if e.src_uid == bwd and e.dep_kind == "data"
               and dag.nodes[e.dst_uid].node_kind == "COMPUTE"
               and dag.nodes[e.dst_uid].compute_subkind != "FWD"]
    return fwd_out, bwd_out


def main() -> int:
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=WORLD)
    mesh = init_device_mesh("cpu", (WORLD,))
    place = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]}
    split = {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}
    ok = True

    print("=== TP: derived from placement (up Colwise, down Rowwise) vs shard_tensor")
    derived = derive_tp(mesh)
    for d in derived:
        print("  DTensor collective:", d)
    base = lower([place, split], 1)
    region = next(u for u, n in base.nodes.items()
                  if n.node_kind == "COMPUTE" and n.compute_subkind == "FWD" and "TP" in n.tag)
    fwd_out, bwd_out = region_out_edges(base, region)
    predicted = set()
    for kind, pas, where in derived:
        if kind == "all_reduce" and where == "region output":
            predicted |= {("TP_COMM", "F", u, v) for u, v in fwd_out}
        elif kind == "all_reduce" and where == "gradient of region input":
            predicted |= {("TP_COMM", "B", u, v) for u, v in bwd_out}
    tp = {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"}
    actual = comm_nodes(lower([place, tp, split], 1), {"TP_COMM"})
    print("  predicted on Piper edges:", sorted(predicted))
    print("  shard_tensor inserted:   ", sorted(actual))
    ok &= predicted == actual
    print("  MATCH" if predicted == actual else "  MISMATCH")

    print("\n=== ZeRO-3: derived from placement (weights Shard(0), batch Shard(0)) vs replicate(shard_params)")
    stages = 3
    sched = [{"op": "place", "filter": {"PP": i}, "devices": [0, 1]} for i in range(stages)]
    zero = {"op": "replicate", "filter": {"PP": "*"}, "devices": [0, 1], "reduce_stream": "dp_stream",
            "shard_grads": True, "shard_params": True}
    base = lower(sched + [split], stages)
    fwd_segs = sorted(u for u, n in base.nodes.items()
                      if n.node_kind == "COMPUTE" and n.compute_subkind == "FWD")
    predicted = set()
    for u in fwd_segs:
        d = derive_zero3(mesh, 2 if "TP" in base.nodes[u].tag else 1)  # the TP-tagged segment holds up and down
        if ("all_gather_into_tensor", "F") in d:
            predicted.add(("ALL_GATHER_COMM", "F", None, u))
        if ("all_gather_into_tensor", "B") in d:
            predicted.add(("ALL_GATHER_COMM", "B", None, u + ".bwd"))
        if ("reduce_scatter_tensor", "B") in d:
            predicted.add(("REDUCE_SCATTER_COMM", "B", u + ".bwd", None))
    print(f"  DTensor, one region: {sorted(set(derive_zero3(mesh, 1)))}")
    act = comm_nodes(lower(sched + [zero, split], stages), {"ALL_GATHER_COMM", "REDUCE_SCATTER_COMM"})
    actual = {(k, p, prod if k == "REDUCE_SCATTER_COMM" else None, cons if k == "ALL_GATHER_COMM" else None)
              for k, p, prod, cons in act}
    both, only_d, only_p = predicted & actual, predicted - actual, actual - predicted
    print(f"  regions: {len(fwd_segs)}   agree: {len(both)}   derived only: {len(only_d)}   Piper only: {len(only_p)}")
    for x in sorted(only_d, key=str):
        print("    derived only:", x)
    for x in sorted(only_p, key=str):
        print("    Piper only:  ", x)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
