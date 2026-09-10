"""With pp_degree > 1, a schedule without `order` fails, and says why.

build_training_dag bridges forward to backward only at the globally last forward
node, so a non-last stage's forward chain reaches its own backward chain only via
the next stage. Inserting point-to-point communication cuts that path, leaving
forward-only components. `order`'s temporal edges reconnect them.

The bare symptom ("expected distinct device sets") does not point at the cause,
so the message carries the diagnosis.
"""
import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.piper import _split_global_training_dag_by_pp_rank


def _node(uid, subkind, device):
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind=subkind,
        tag={"PASS": "F" if subkind == "FWD" else "B"},
        device=list(device),
        stream="default_stream",
        node_meta={},
    )


def test_forward_only_components_explain_the_missing_order_directive() -> None:
    """Two components share a device set and one of them is forward-only."""
    dag = TrainingDAG()
    # stage 0's forward, cut off from its own backward (as the P2P split leaves it)
    dag.add_node(_node("s0.fwd", "FWD", (0, 2)))
    # stage 0's backward, joined to an update node
    dag.add_node(_node("s0.bwd", "BWD", (0, 2)))
    dag.add_node(TrainingDAGNode(
        uid="upd.0", node_kind="UPD", compute_subkind=None, tag={"PASS": None},
        device=[0, 2], stream="default_stream", node_meta={}))
    dag.add_edge(TrainingDAGEdge("s0.bwd", "upd.0", "temporal"))
    # stage 1, self-contained
    dag.add_node(_node("s1.fwd", "FWD", (1, 3)))
    dag.add_node(_node("s1.bwd", "BWD", (1, 3)))
    dag.add_edge(TrainingDAGEdge("s1.fwd", "s1.bwd", "data"))

    with pytest.raises(ValueError, match="no `order` directive"):
        _split_global_training_dag_by_pp_rank(dag)


def test_message_still_reports_the_device_sets() -> None:
    dag = TrainingDAG()
    dag.add_node(_node("a.fwd", "FWD", (0, 2)))
    dag.add_node(_node("b.fwd", "FWD", (0, 2)))

    with pytest.raises(ValueError, match=r"got \[\(0, 2\), \(0, 2\)\]"):
        _split_global_training_dag_by_pp_rank(dag)
