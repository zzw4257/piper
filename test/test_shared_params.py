"""A parameter read by several regions: one tensor on one device group, refused across groups (log F79, F80)."""
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


def test_parameter_shared_across_device_groups_is_refused() -> None:
    _, _, note = p.lower(THREE_GPUS, MAKE(True))
    assert note.startswith("backend='piper' raised"), note


def test_buffer_shared_across_device_groups_is_fine() -> None:
    _, _, note = p.lower(THREE_GPUS, MAKE(False))
    assert note == "", note
