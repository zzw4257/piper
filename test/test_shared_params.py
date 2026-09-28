"""A parameter read by several regions: one tensor on one device group, a gradient-summed copy per group across groups, refused under ZeRO (log F79, F80, F83)."""
import sys

import torch
import torch.nn as nn

sys.path[:0] = ["experiments", "examples"]
import probe_route as p  # noqa: E402
from src.piper import annotate  # noqa: E402


class _Tied(nn.Module):
    def __init__(self, share: bool):
        super().__init__()
        self.share = share
        self.first = nn.Linear(16, 16, bias=False)
        self.mid = nn.Linear(16, 16, bias=False)
        self.last = nn.Linear(16, 16, bias=False)
        self.register_buffer("scale", torch.ones(16))  # read by every region; not trained

    def forward(self, x):
        with annotate("PP"):
            x = self.first(x) * self.scale
        with annotate("PP"):
            x = self.mid(x) * self.scale
        with annotate("PP"):
            return (self.first if self.share else self.last)(x) * self.scale


ONE_GPU = [{"op": "place", "filter": {"PP": i}, "devices": [0]} for i in range(3)] + [p.SPLIT]
THREE_GPUS = ([{"op": "place", "filter": {"PP": i}, "devices": [i]} for i in range(3)] + [p.SPLIT] +
              [{"op": "order", "filters": [[{"PP": i, "PASS": "F"}], [{"PP": i, "PASS": "B"}]]} for i in range(3)])
MAKE = lambda share: lambda: (_Tied(share), (torch.empty(4, 16, device="meta"),))  # noqa: E731


def test_parameter_shared_on_one_device_group_lowers() -> None:
    _, _, note = p.lower(ONE_GPU, MAKE(True))
    assert note == "", note


def test_parameter_shared_across_device_groups_is_recorded() -> None:
    # Each group keeps a copy; the runtime sums their gradients (log F83).
    _, per, note = p.lower(THREE_GPUS, MAKE(True))
    assert note == "", note
    specs = [n.node_meta.get("cross_rank_shared") for d in per for n in d.nodes.values() if n.compute_subkind == "FWD"]
    assert specs and all(sp == specs[0] for sp in specs)
    assert list(specs[0].values()) == [[0, 2]] and "first" in next(iter(specs[0]))


def test_parameter_shared_under_zero_is_refused() -> None:
    sched = ([{"op": "place", "filter": {"PP": i}, "devices": [0, 1]} for i in range(3)]
             + [{"op": "replicate", "filter": {"PP": "*"}, "devices": [0, 1], "reduce_stream": "dp_stream",
                 "shard_grads": True, "shard_params": True}, p.SPLIT]
             + [{"op": "order", "filters": [[{"PP": i, "PASS": "F"}], [{"PP": i, "PASS": "B"}]]} for i in range(3)])
    _, _, note = p.lower(sched, MAKE(True))
    assert note.startswith("backend='piper' raised"), note


def test_buffer_shared_across_device_groups_is_fine() -> None:
    _, _, note = p.lower(THREE_GPUS, MAKE(False))
    assert note == "", note
