"""Two independent encoders feeding one decoder: the shape issue #16 asks about.

    x_img -> [image encoder] --\
                                 +--> [decoder] -> y
    x_txt -> [text encoder]  --/

Each encoder is its own PP region and can sit on its own device. With the default
routing the text encoder's input is threaded through the image segment, so the two
run one after the other (log F66); with route {mode: consumers} they are
independent (log F71).

With ``tp_degree`` set, each encoder ends in a Megatron MLP block inside a TP
region followed by a replicated projection (log F73):

    layers (replicated) -> [up -> gelu -> down] (TP) -> post (replicated)

``up``/``down`` are declared in TP-local shapes, as in ``models/tp_mlp.py``, and
``tp_mlp.shard_weights`` slices the global weights for each rank.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate


def global_weights(dim: int, depth_img: int, depth_txt: int, depth_dec: int, seed: int,
                   dtype=torch.float32, hidden: int = 0) -> dict:
    """One set of weights, drawn on CPU from an explicit generator, for every placement."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, depth in (("img", depth_img), ("txt", depth_txt), ("dec", depth_dec)):
        for i in range(depth):
            out[f"{name}.{i}.weight"] = (torch.randn(dim, dim, generator=g) / dim ** 0.5).to(dtype)
    if hidden:
        for name in ("img", "txt"):
            out[f"{name}_tp.up.weight"] = (torch.randn(hidden, dim, generator=g) / dim ** 0.5).to(dtype)
            out[f"{name}_tp.down.weight"] = (torch.randn(dim, hidden, generator=g) / hidden ** 0.5).to(dtype)
            out[f"{name}_post.weight"] = (torch.randn(dim, dim, generator=g) / dim ** 0.5).to(dtype)
    return out


class TPMLP(nn.Module):
    def __init__(self, dim: int, hidden_local: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden_local, bias=False)    # column-parallel
        self.down = nn.Linear(hidden_local, dim, bias=False)  # row-parallel: output is a partial sum

    def forward(self, x):
        with annotate("TP"):
            return self.down(F.gelu(self.up(x)))


class Branches(nn.Module):
    def __init__(self, dim: int, depth_img: int, depth_txt: int, depth_dec: int = 1,
                 hidden: int = 0, tp_degree: int = 1):
        super().__init__()
        self.img = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_img))
        self.txt = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_txt))
        self.dec = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_dec))
        self.hidden = hidden
        if hidden:
            if hidden % tp_degree:
                raise ValueError(f"hidden={hidden} must divide by tp_degree={tp_degree}")
            self.img_tp = TPMLP(dim, hidden // tp_degree)
            self.txt_tp = TPMLP(dim, hidden // tp_degree)
            self.img_post = nn.Linear(dim, dim, bias=False)
            self.txt_post = nn.Linear(dim, dim, bias=False)

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            a = x_img
            for layer in self.img:
                a = F.gelu(layer(a))
            if self.hidden:
                a = self.img_post(a + self.img_tp(a))
        with annotate("PP"):
            b = x_txt
            for layer in self.txt:
                b = F.gelu(layer(b))
            if self.hidden:
                b = self.txt_post(b + self.txt_tp(b))
        with annotate("PP"):
            y = a + b
            for layer in self.dec:
                y = layer(y)
        return y
