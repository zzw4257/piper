"""The smallest model that exercises tensor parallelism in Piper.

Three annotated segments on one device group:

    pre (replicated) -> [ up -> gelu -> down ] (TP) -> post (replicated)

A replicated segment on *each* side of the TP region is deliberate. Without a
segment upstream there is no consumer for the region's input gradient, so the
backward half of Megatron's f/g pair is never inserted (see notes/log.md F6).

Weights are declared in **TP-local shapes**: `up` is column-parallel and `down`
is row-parallel, so `hidden` is divided by the TP degree here and every rank's
GraphModule has identical shapes. That is the same convention Piper's expert
regions already rely on -- the IR has no way to say "this parameter is split
along dim d", so the model says it instead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate


class TPMlp(nn.Module):
    def __init__(self, dim: int, hidden: int, tp_degree: int):
        super().__init__()
        if hidden % tp_degree:
            raise ValueError(f"hidden={hidden} must divide by tp_degree={tp_degree}")
        hidden_local = hidden // tp_degree

        self.dim = dim
        self.hidden = hidden
        self.tp_degree = tp_degree

        self.pre = nn.Linear(dim, dim, bias=False)
        # Column-parallel: output features are sharded, so dim 0 of the weight.
        self.up = nn.Linear(dim, hidden_local, bias=False)
        # Row-parallel: input features are sharded, so dim 1 of the weight. Its
        # output is a partial sum and needs the forward all-reduce.
        self.down = nn.Linear(hidden_local, dim, bias=False)
        self.post = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One outer PP scope keeps all three segments on stage 0, so the whole
        # model sits on a single device group and pp_degree stays 1.
        with annotate("PP"):
            x = self.pre(x)
            with annotate("TP"):
                x = self.down(F.gelu(self.up(x)))
            return self.post(x)
