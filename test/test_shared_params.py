"""A trainable parameter read by two regions is refused, not silently duplicated (log F79)."""
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


SCHED = [{"op": "place", "filter": {"PP": i}, "devices": [0]} for i in range(3)] + [p.SPLIT]


def test_parameter_shared_across_regions_is_refused() -> None:
    _, _, note = p.lower(SCHED, lambda: (_Tied(True), (torch.empty(4, 16, device="meta"),)))
    assert note.startswith("backend='piper' raised"), note


def test_buffer_shared_across_regions_is_fine() -> None:
    _, _, note = p.lower(SCHED, lambda: (_Tied(False), (torch.empty(4, 16, device="meta"),)))
    assert note == "", note
