"""Two independent encoders feeding one decoder: the shape issue #16 asks about.

    x_img -> [image encoder] --\
                                 +--> [decoder] -> y
    x_txt -> [text encoder]  --/

Each encoder is its own PP region and can sit on its own device. With the default
routing the text encoder's input is threaded through the image segment, so the two
run one after the other (log F66); with route {mode: consumers} they are
independent (log F71).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate


def global_weights(dim: int, depth_img: int, depth_txt: int, depth_dec: int, seed: int,
                   dtype=torch.float32) -> dict:
    """One set of weights, drawn on CPU from an explicit generator, for every placement."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, depth in (("img", depth_img), ("txt", depth_txt), ("dec", depth_dec)):
        for i in range(depth):
            out[f"{name}.{i}.weight"] = (torch.randn(dim, dim, generator=g) / dim ** 0.5).to(dtype)
    return out


class Branches(nn.Module):
    def __init__(self, dim: int, depth_img: int, depth_txt: int, depth_dec: int = 1):
        super().__init__()
        self.img = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_img))
        self.txt = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_txt))
        self.dec = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_dec))

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            a = x_img
            for layer in self.img:
                a = F.gelu(layer(a))
        with annotate("PP"):
            b = x_txt
            for layer in self.txt:
                b = F.gelu(layer(b))
        with annotate("PP"):
            y = a + b
            for layer in self.dec:
                y = layer(y)
        return y
