"""layout derives every boundary collective from placements; each lowering is node for node the
one the written directives give (log F85)."""
import json
import sys

import pytest
import torch

sys.path[:0] = ["experiments", "examples"]
import probe_route as p  # noqa: E402

SPLIT = {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}
P2 = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]}
P4 = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1, 2, 3]}
MESH = {"op": "mesh", "axes": [["dp", 2], ["tp", 2]]}
TP_W = {"up": "colwise", "down": "rowwise"}
mlp = lambda tp: lambda: (p.TPMlp(32, 128, tp, 1), (torch.empty(8, 32, device="meta"),))  # noqa: E731
ring = lambda: (p.RingAttn(64, 2, 2), tuple(torch.empty(2, 8, 64, device="meta") for _ in range(3)))  # noqa: E731

CASES = {
    "tp": ([P2, {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"}, SPLIT],
           [P2, {"op": "layout", "filter": {"TP": "*"}, "devices": [0, 1], "axis": "tp", "params": TP_W, "stream": "tp_stream"}, SPLIT],
           mlp(2)),
    "tp_derived": ([P2, {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream", "params": TP_W}, SPLIT],
                   [P2, {"op": "layout", "filter": {"TP": "*"}, "devices": [0, 1], "axis": "tp", "params": TP_W, "stream": "tp_stream"}, SPLIT],
                   mlp(2)),
    "dp": ([P2, {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1], "reduce_stream": "dp_stream"}, SPLIT],
           [P2, {"op": "layout", "filter": {"PP": 0}, "devices": [0, 1], "axis": "dp", "batch": "shard(0)", "reduce_stream": "dp_stream"}, SPLIT],
           mlp(1)),
    "tp_x_dp": ([MESH, P4, {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1, 2, 3], "reduce_stream": "dp_stream"},
                 {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1, 2, 3], "stream": "tp_stream"}, SPLIT],
                [MESH, P4, {"op": "layout", "filter": {"PP": 0}, "devices": [0, 1, 2, 3], "axis": "dp", "batch": "shard(0)", "reduce_stream": "dp_stream"},
                 {"op": "layout", "filter": {"TP": "*"}, "devices": [0, 1, 2, 3], "axis": "tp", "params": TP_W, "stream": "tp_stream"}, SPLIT],
                mlp(2)),
    "cp_with_dp": ([{"op": "place", "filter": {"PP": 0}, "devices": [0, 1], "stream": "default_stream"},
                    {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1], "reduce_stream": "dp_stream"},
                    {"op": "ring_exchange", "filter": {"CP": "*"}, "devices": [0, 1], "stream": "cp_stream", "derive": {"seq_dim": 2}}, SPLIT],
                   [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1], "stream": "default_stream"},
                    {"op": "layout", "filter": {"PP": 0}, "devices": [0, 1], "axis": "cp", "inputs": {"*": "shard(1)"},
                     "reduce_stream": "dp_stream", "ring_stream": "cp_stream"}, SPLIT],
                   ring),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_layout_derives_what_the_directives_write(case) -> None:
    written, layout, make = CASES[case]
    _, a, na = p.lower(written, make)
    _, b, nb = p.lower(layout, make)
    assert na == nb == "", (na, nb)
    assert p.shape(a) == p.shape(b)
