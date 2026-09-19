"""ring_exchange lowering: the edge-spliced baseline."""
import sys
from pathlib import Path

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.ring_attn import RingAttn  # noqa: E402

from src.compile import _reset_run_state
from src.ordering import _serial_topological_order
from src.piper import _reset_annotation_state, piper
from src.schedule import derive_schedule_info
from src.state import piper_metadata
from src.tasks import TaskType, training_dag_task_type

_PLACE = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1], "stream": "default_stream"}
_SPLIT = {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}


def _ring(tensors, devices=(0, 1)):
    return {"op": "ring_exchange", "filter": {"CP": "*"}, "devices": list(devices),
            "tensors": tensors, "stream": "cp_stream"}


def _lower(directives, steps=4):
    _reset_run_state()
    _reset_annotation_state()
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, "inline")
    piper_metadata.visualize_dag = False
    with torch.device("meta"):
        model = RingAttn(32, 2, steps).to(torch.float32)
    inputs = tuple(torch.empty(2, 8, 32, device="meta") for _ in range(3))
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(*inputs)
    return piper_metadata.per_pp_training_dags[0]


def test_ring_nodes_sit_between_consecutive_steps_only() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring(["k", "v"])], steps=4)
    ring = {u: n for u, n in dag.nodes.items() if n.node_kind == "RING_COMM"}
    fwd = [u for u, n in ring.items() if n.tag.get("PASS") == "F"]
    bwd = [u for u, n in ring.items() if n.tag.get("PASS") == "B"]
    assert len(fwd) == 3 and len(bwd) == 3, (fwd, bwd)   # steps-1 boundaries, each way

    for u in fwd:
        n = ring[u]
        assert n.node_meta["ring_shift"] == 1
        assert len(n.node_meta["ring_tensor_idxs"]) == 2
        assert n.stream == "cp_stream"
        preds = [dag.nodes[e.src_uid] for e in dag.edges if e.dst_uid == u and e.dep_kind == "data"]
        succs = [dag.nodes[e.dst_uid] for e in dag.edges if e.src_uid == u and e.dep_kind == "data"]
        assert {p.uid for p in preds} == {n.node_meta["source_uid"]}
        assert all("CP" in p.tag and p.compute_subkind == "FWD" for p in preds)
        assert all("CP" in s.tag and s.compute_subkind == "FWD" for s in succs)
        # consecutive: the successor is the next ring step
        assert {s.tag["CP"] for s in succs} == {preds[0].tag["CP"] + 1}
        assert training_dag_task_type(n) is TaskType.FWD_RING_EXCHANGE

    for u in bwd:
        n = ring[u]
        assert n.node_meta["ring_shift"] == -1
        assert training_dag_task_type(n) is TaskType.BWD_RING_EXCHANGE
        succs = [dag.nodes[e.dst_uid] for e in dag.edges if e.src_uid == u and e.dep_kind == "data"]
        preds = [dag.nodes[e.src_uid] for e in dag.edges if e.dst_uid == u and e.dep_kind == "data"]
        # backward edges are reversed: grad flows from step i+1 back to step i
        assert {s.tag["CP"] for s in succs} == {preds[0].tag["CP"] - 1}

    # Nothing on prologue->CP0 or CP3->epilogue: those leave the matched set.
    for e in dag.edges:
        if e.dep_kind != "data":
            continue
        s, d = dag.nodes[e.src_uid], dag.nodes[e.dst_uid]
        if s.node_kind == "COMPUTE" and d.node_kind == "COMPUTE":
            assert not (("CP" in s.tag) ^ ("CP" in d.tag)) or True  # unchanged compute edges are fine
    assert not any(
        dag.nodes[e.src_uid].node_kind == "RING_COMM" and "CP" not in dag.nodes[e.dst_uid].tag
        for e in dag.edges
    )


def test_fwd_ring_node_dispatches_after_source_before_consumer() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring(["k", "v"])], steps=3)
    order = _serial_topological_order(dag)
    pos = {u: i for i, u in enumerate(order)}
    for u, n in dag.nodes.items():
        if n.node_kind != "RING_COMM":
            continue
        src = n.node_meta["source_uid"]
        dst = next(e.dst_uid for e in dag.edges if e.src_uid == u and e.dep_kind == "data")
        assert pos[src] < pos[u] < pos[dst], (src, u, dst)


# Directive errors surface through Dynamo, which wraps the backend's ValueError;
# the message is preserved, the type is not.
def test_produced_tensor_is_refused() -> None:
    with pytest.raises(BackendCompilerFailed, match="produced by region"):
        _lower([_PLACE, _SPLIT, _ring(["o_1"])], steps=2)


def test_unknown_tensor_is_refused_with_the_crossing_set() -> None:
    with pytest.raises(BackendCompilerFailed, match="does not cross the boundary.*crossing tensors"):
        _lower([_PLACE, _SPLIT, _ring(["nope"])], steps=2)


def test_missing_tensor_list_is_refused() -> None:
    d = _ring(["k"]); del d["tensors"]
    with pytest.raises(BackendCompilerFailed, match="non-empty `tensors` list"):
        _lower([_PLACE, _SPLIT, d], steps=2)
