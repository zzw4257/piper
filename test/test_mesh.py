"""Named mesh axes over a stage's device group (log F82)."""
import json
import sys

import pytest
import torch

sys.path[:0] = ["experiments", "examples"]
import probe_route as p  # noqa: E402
from src.schedule import derive_schedule_info, mesh_coords  # noqa: E402

PLACE4 = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1, 2, 3]}
SPLIT = {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}


def test_mesh_axes_must_cover_the_group() -> None:
    info = derive_schedule_info([{"op": "mesh", "axes": [["dp", 2], ["tp", 2]]}, PLACE4, SPLIT], "x.json")
    assert info["mesh"] == [["dp", 2], ["tp", 2]]
    with pytest.raises(ValueError, match="multiply to 2"):
        derive_schedule_info([{"op": "mesh", "axes": [["tp", 2]]}, PLACE4, SPLIT], "x.json")
    assert "mesh" not in derive_schedule_info([PLACE4, SPLIT], "x.json")


def test_coordinates_are_row_major_first_axis_outermost() -> None:
    info = {"mesh": [["dp", 2], ["tp", 2]]}
    assert [mesh_coords(info, i) for i in range(4)] == [
        {"dp": 0, "tp": 0}, {"dp": 0, "tp": 1}, {"dp": 1, "tp": 0}, {"dp": 1, "tp": 1}]
    assert mesh_coords({}, 3) == {}


def test_tp_and_dp_compose_on_a_mesh() -> None:
    make = lambda: (p.TPMlp(32, 128, 2, 1), (torch.empty(8, 32, device="meta"),))  # noqa: E731
    repl = {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1, 2, 3], "reduce_stream": "dp_stream"}
    tp = {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1, 2, 3], "stream": "tp_stream"}
    # without a mesh the TP group is the DP group: refused, as before
    _, _, note = p.lower([PLACE4, repl, tp, SPLIT], make)
    assert note.startswith("backend='piper' raised")
    dag, _, note = p.lower([{"op": "mesh", "axes": [["dp", 2], ["tp", 2]]}, PLACE4, repl, tp, SPLIT], make)
    assert note == ""
    kinds = [n.node_kind for n in dag.nodes.values()]
    assert kinds.count("TP_COMM") == 2 and kinds.count("REDUCE_COMM") == 3
    assert {n.node_meta["axis"] for n in dag.nodes.values() if n.node_kind == "TP_COMM"} == {"tp"}
