"""Characterize where ZeRO-3 all-gathers land in the dispatch order (log F37).

This is *not* an assertion that the behaviour is desirable; it is the opposite.
It pins the evidence so that a change to the issue policy has a red test to
turn green, and so the finding cannot drift silently.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.tp_mlp import TPMlp  # noqa: E402

from src.compile import _reset_run_state
from src.ordering import _serial_topological_order
from src.piper import _reset_annotation_state, piper
from src.schedule import derive_schedule_info, load_schedule_directives
from src.state import piper_metadata

_SCHEDULE = Path(__file__).resolve().parents[1] / "examples/base-schedules/zero3_s3_mb1.json"


def _lower_zero3():
    _reset_run_state()
    _reset_annotation_state()
    directives = load_schedule_directives(str(_SCHEDULE))
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, str(_SCHEDULE))
    piper_metadata.visualize_dag = False
    with torch.device("meta"):
        model = TPMlp(64, 128, 1, 3).to(torch.float32)
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(torch.empty(4, 64, device="meta"))
    dag = piper_metadata.per_pp_training_dags[0]
    return dag, _serial_topological_order(dag)


def _kind(dag, uid):
    return dag.nodes[uid].node_kind


def test_forward_all_gathers_are_roots_issued_before_any_compute() -> None:
    dag, order = _lower_zero3()
    fwd_ag = [u for u in order
              if _kind(dag, u) == "ALL_GATHER_COMM" and dag.nodes[u].tag.get("PASS") == "F"]
    assert len(fwd_ag) == 9, fwd_ag  # 3 stages x 3 segments, one per param-bearing region
    for u in fwd_ag:
        assert not [e for e in dag.edges if e.dst_uid == u], f"{u} is no longer a root"
    first_compute = next(i for i, u in enumerate(order) if _kind(dag, u) == "COMPUTE")
    assert max(order.index(u) for u in fwd_ag) < first_compute, (
        "forward all-gathers no longer all precede the first compute; "
        "if this is intentional, F37 has been fixed -- update the log"
    )


def test_backward_all_gather_hangs_off_forward_compute_only() -> None:
    dag, order = _lower_zero3()
    bwd_ag = [u for u in order
              if _kind(dag, u) == "ALL_GATHER_COMM" and dag.nodes[u].tag.get("PASS") == "B"]
    assert bwd_ag
    for u in bwd_ag:
        preds = [(e.src_uid, e.dep_kind) for e in dag.edges if e.dst_uid == u]
        consumer = dag.nodes[u].node_meta["compute_uid"]
        fwd_uid = dag.nodes[consumer].node_meta.get("fwd_uid")
        assert preds == [(fwd_uid, "temporal")], (u, preds)
        # Issued as soon as its forward finishes; consumed a full backward pass later.
        assert order.index(u) < order.index(consumer) - 1


def test_issue_distance_is_unbounded() -> None:
    """The gap between issue and consumption grows with model depth: no budget."""
    dag, order = _lower_zero3()
    pos = {u: i for i, u in enumerate(order)}
    gaps = [
        pos[dag.nodes[u].node_meta["compute_uid"]] - pos[u]
        for u in order
        if _kind(dag, u) == "ALL_GATHER_COMM" and dag.nodes[u].tag.get("PASS") == "F"
    ]
    # Layer 0's gather sits right before layer 0; the last layer's was issued
    # 8 slots earlier than that and consumed 16 compute slots later.
    assert min(gaps) >= 1 and max(gaps) >= 16, gaps
