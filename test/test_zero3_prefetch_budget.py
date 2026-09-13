"""replicate(prefetch_distance=d) bounds how early ZeRO-3 gathers issue (log F37).

The default is unchanged and still characterized by test_zero3_dispatch_order.py.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.tp_mlp import TPMlp  # noqa: E402

from src.compile import _reset_run_state
from src.ordering import _serial_topological_order
from src.piper import _reset_annotation_state, piper
from src.schedule import derive_schedule_info
from src.state import piper_metadata


def _lower(prefetch_distance):
    rep = {"op": "replicate", "filter": {"PP": "*"}, "devices": [0, 1],
           "reduce_stream": "dp_stream", "shard_grads": True, "shard_params": True}
    if prefetch_distance is not None:
        rep["prefetch_distance"] = prefetch_distance
    directives = [
        *({"op": "place", "filter": {"PP": i}, "devices": [0, 1], "stream": "default_stream"} for i in range(3)),
        rep,
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    ]
    _reset_run_state(); _reset_annotation_state()
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, "inline")
    piper_metadata.visualize_dag = False
    with torch.device("meta"):
        model = TPMlp(64, 128, 1, 3).to(torch.float32)
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(torch.empty(4, 64, device="meta"))
    dag = piper_metadata.per_pp_training_dags[0]
    return dag, _serial_topological_order(dag)


def _fwd_ags(dag, order):
    return [u for u in order
            if dag.nodes[u].node_kind == "ALL_GATHER_COMM" and dag.nodes[u].tag.get("PASS") == "F"]


def test_distance_one_issues_each_gather_one_layer_ahead() -> None:
    dag, order = _lower(1)
    pos = {u: i for i, u in enumerate(order)}
    ags = _fwd_ags(dag, order)
    assert len(ags) == 9
    first_compute = min(pos[u] for u in order if dag.nodes[u].node_kind == "COMPUTE")
    # F37's signature -- every gather before the first compute -- is gone.
    assert max(pos[u] for u in ags) > first_compute

    bounded = [u for u in ags if "prefetch_distance" in dag.nodes[u].node_meta]
    assert len(bounded) == 8  # the very first layer has no predecessor to wait on
    for u in bounded:
        target = dag.nodes[u].node_meta["compute_uid"]
        temporal = [e.src_uid for e in dag.edges if e.dst_uid == u and e.dep_kind == "temporal"]
        assert len(temporal) == 1
        back = temporal[0]
        assert dag.nodes[back].node_kind == "COMPUTE" and dag.nodes[back].compute_subkind == "FWD"
        # issued after the previous layer's compute and before its own
        assert pos[back] < pos[u] < pos[target], (back, u, target)
        # and the previous layer really is the previous layer: one data hop
        assert any(e.src_uid == back and e.dst_uid == target and e.dep_kind == "data"
                   for e in dag.edges)


def test_backward_gathers_are_bounded_too() -> None:
    dag, order = _lower(1)
    pos = {u: i for i, u in enumerate(order)}
    bwd = [u for u in order
           if dag.nodes[u].node_kind == "ALL_GATHER_COMM" and dag.nodes[u].tag.get("PASS") == "B"]
    assert bwd
    for u in bwd:
        target = dag.nodes[u].node_meta["compute_uid"]
        gap = pos[target] - pos[u]
        # F37 had backward gathers issued during the forward pass, ~30 slots
        # ahead of their consumer; a one-step budget keeps them adjacent.
        assert 1 <= gap <= 3, (u, gap)


def test_default_is_unchanged() -> None:
    dag, order = _lower(None)
    assert not any("prefetch_distance" in dag.nodes[u].node_meta for u in _fwd_ags(dag, order))
    pos = {u: i for i, u in enumerate(order)}
    first_compute = min(pos[u] for u in order if dag.nodes[u].node_kind == "COMPUTE")
    assert max(pos[u] for u in _fwd_ags(dag, order)) < first_compute  # F37 as shipped
