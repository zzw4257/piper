"""Collectives derived from placements match the ones the directives hand-insert (log F68)."""
import json
import sys

import pytest
import torch

sys.path[:0] = ["examples"]
from models.tp_mlp import TPMlp  # noqa: E402
from src.piper import _reset_annotation_state, piper  # noqa: E402
from src.schedule import derive_schedule_info, load_schedule_directives  # noqa: E402
from src.state import piper_metadata  # noqa: E402

PLACE = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]}
TP_RULE = {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"}


def _split(n):
    return {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": n}


def _lower(schedule, tmp_path, stages=1):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(schedule))
    directives = load_schedule_directives(str(path))
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, str(path))
    piper_metadata.visualize_dag = False
    _reset_annotation_state()
    with torch.device("meta"):
        model, x = TPMlp(32, 128, 1, stages), torch.empty(8, 32, device="meta")
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(x)
    (dag,) = piper_metadata.per_pp_training_dags
    return dag


def _tp_nodes(dag):
    out = set()
    for uid, n in dag.nodes.items():
        if n.node_kind == "TP_COMM":
            succ = [e.dst_uid for e in dag.edges if e.src_uid == uid and e.dep_kind == "data"]
            out.add((n.tag.get("PASS"), n.node_meta["source_uid"], tuple(succ), n.node_meta["tp_tensor_idx"]))
    return out


@pytest.mark.parametrize("microbatches", [1, 4])
def test_derived_tp_collectives_match_shard_tensor(tmp_path, microbatches) -> None:
    rule = _tp_nodes(_lower([PLACE, TP_RULE, _split(microbatches)], tmp_path))
    derived = _tp_nodes(_lower(
        [PLACE, {**TP_RULE, "params": {"up": "colwise", "down": "rowwise"}}, _split(microbatches)], tmp_path))
    assert len(rule) == 2 * microbatches
    assert derived == rule


def test_replicated_parameters_need_no_collective(tmp_path) -> None:
    dag = _lower([PLACE, {**TP_RULE, "params": {"up": "replicate", "down": "replicate"}}, _split(1)], tmp_path)
    assert _tp_nodes(dag) == set()


def test_placement_dtensor_would_silently_resplit_is_refused(tmp_path) -> None:
    # up column-parallel, down declared whole: DTensor chunks down locally to fit, with no
    # collective, and ends with a partial sum. Piper would run down whole, so refuse.
    with pytest.raises(Exception, match="redistribution inside the region"):
        _lower([PLACE, {**TP_RULE, "params": {"up": "colwise", "down": "replicate"}}, _split(1)], tmp_path)


def test_placement_needing_a_collective_inside_the_region_is_refused(tmp_path) -> None:
    # up row-parallel, down column-parallel: gelu would see a partial sum, so DTensor
    # must all-reduce between the two layers, inside the region.
    with pytest.raises(Exception, match="redistribution inside the region"):
        _lower([PLACE, {**TP_RULE, "params": {"up": "rowwise", "down": "colwise"}}, _split(1)], tmp_path)


def test_regather_false_keeps_one_gather_per_segment(tmp_path) -> None:
    stages = 3
    sched = [{"op": "place", "filter": {"PP": i}, "devices": [0, 1]} for i in range(stages)]
    zero = {"op": "replicate", "filter": {"PP": "*"}, "devices": [0, 1], "reduce_stream": "dp_stream",
            "shard_grads": True, "shard_params": True}

    def counts(dag):
        kinds = [(n.node_kind, n.tag.get("PASS")) for n in dag.nodes.values()]
        return {k: kinds.count(k) for k in set(kinds) if k[0] in ("ALL_GATHER_COMM", "REDUCE_SCATTER_COMM")}

    shipped = counts(_lower(sched + [zero, _split(1)], tmp_path, stages))
    kept = counts(_lower(sched + [{**zero, "regather": False}, _split(1)], tmp_path, stages))
    # 9 segments: shipped regathers for every backward but the last stage's (F65, F68).
    assert shipped == {("ALL_GATHER_COMM", "F"): 9, ("ALL_GATHER_COMM", "B"): 8, ("REDUCE_SCATTER_COMM", "B"): 9}
    # Holding each forward's parameters until its backward leaves exactly what placement derives.
    assert kept == {("ALL_GATHER_COMM", "F"): 9, ("REDUCE_SCATTER_COMM", "B"): 9}


# ----------------------------------------------------------------------------- CP (log F70)
def _lower_ring(schedule, tmp_path, steps):
    from models.ring_attn import RingAttn

    path = tmp_path / "r.json"
    path.write_text(json.dumps(schedule))
    directives = load_schedule_directives(str(path))
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, str(path))
    piper_metadata.visualize_dag = False
    _reset_annotation_state()
    with torch.device("meta"):
        model = RingAttn(64, 2, steps)
        xs = tuple(torch.empty(2, 8, 64, device="meta") for _ in range(3))
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(*xs)
    (dag,) = piper_metadata.per_pp_training_dags
    return dag


def _ring_edges(dag):
    """Every edge touching a ring node, plus the ring nodes' payload and direction."""
    ring = {u for u, n in dag.nodes.items() if n.node_kind == "RING_COMM"}
    edges = {(e.src_uid, e.dst_uid, e.dep_kind) for e in dag.edges if e.src_uid in ring or e.dst_uid in ring}
    meta = {(u, tuple(dag.nodes[u].node_meta["ring_tensor_idxs"]), dag.nodes[u].node_meta["ring_shift"]) for u in ring}
    return edges, meta


@pytest.mark.parametrize("ranks,hoist", [(2, False), (2, True), (4, False), (4, True)])
def test_derived_ring_payload_matches_named_tensors(tmp_path, ranks, hoist) -> None:
    devices = list(range(ranks))
    base = [{"op": "place", "filter": {"PP": 0}, "devices": devices}, _split(1)]
    ring = {"op": "ring_exchange", "filter": {"CP": "*"}, "devices": devices, "stream": "cp_stream",
            "hoist": hoist, "distance": 1}
    named = _lower_ring(base[:1] + [{**ring, "tensors": ["k", "v"]}] + base[1:], tmp_path, ranks)
    derived = _lower_ring(base[:1] + [{**ring, "derive": {"seq_dim": 2}}] + base[1:], tmp_path, ranks)
    assert _ring_edges(derived) == _ring_edges(named)
    # the gather of K/V over n ranks, split into n-1 forward hops (and n-1 backward)
    fwd = [n for n in derived.nodes.values() if n.node_kind == "RING_COMM" and n.tag.get("PASS") == "F"]
    assert len(fwd) == ranks - 1


def test_ring_derivation_refuses_a_region_with_nothing_to_gather(tmp_path) -> None:
    # Split the TP MLP along the batch: a linear layer needs nothing gathered, so no ring payload.
    ring = {"op": "ring_exchange", "filter": {"TP": "*"}, "devices": [0, 1], "derive": {"seq_dim": 0}}
    with pytest.raises(Exception, match="no single ring payload"):
        _lower([PLACE, ring, _split(1)], tmp_path)
