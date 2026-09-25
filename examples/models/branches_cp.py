"""Two branches, one context-parallel: an image MLP and a ring-attention text encoder.

    x_img          -> [image layers] ----------------------------\
    x_txt, k, v    -> [q proj] [CP step]*n [o proj] [pool (TP)] --+--> [decoder] -> y

The text branch holds one rank's sequence chunk. Each CP region is one ring step,
as in ``models/ring_attn.py``: K/V pass through unchanged and ``ring_exchange``
rotates them between steps. The pool region sums its chunk over the sequence and
divides by the full length, so its output is a partial sum across the CP group;
``shard_tensor`` with ``inputs: {"0": "shard(1)"}`` derives the one all-reduce that
needs and no backward collective (log F76). With ``n_steps=1`` and the whole
sequence on one GPU the same code is the dense reference.

With ``ep=True`` the image branch also has an expert layer in an EP region,
``a = post(a + gelu(expert(a)))``; ``shard`` puts an all-to-all on each side of it.
Given the same expert weights on every rank, EP only moves tokens and back, so the
forward and every non-expert gradient match one GPU.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate
from models.ring_attn import ring_step


def global_weights(dim: int, depth_img: int, depth_dec: int, seed: int, ep: bool = False) -> dict:
    g = torch.Generator().manual_seed(seed)
    w = lambda: torch.randn(dim, dim, generator=g) / dim ** 0.5  # noqa: E731
    out = {f"img.{i}.weight": w() for i in range(depth_img)}
    out.update({"q_proj.weight": w(), "o_proj.weight": w()})
    out.update({f"dec.{i}.weight": w() for i in range(depth_dec)})
    if ep:
        out.update({"img_expert.weight": w(), "img_post.weight": w()})
    return out


class BranchesCP(nn.Module):
    def __init__(self, dim: int, heads: int, n_steps: int, seq_total: int,
                 depth_img: int = 2, depth_dec: int = 1, ep: bool = False):
        super().__init__()
        self.ep = ep
        if ep:
            self.img_expert = nn.Linear(dim, dim, bias=False)
            self.img_post = nn.Linear(dim, dim, bias=False)
        self.heads, self.n_steps, self.seq_total = heads, n_steps, seq_total
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.img = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_img))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dec = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth_dec))

    def _split_heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x_img, x_txt, k, v):
        with annotate("PP"):
            a = x_img
            for layer in self.img:
                a = F.gelu(layer(a))
            if self.ep:
                with annotate("EP"):
                    e = F.gelu(self.img_expert(a))
                a = self.img_post(a + e)
        with annotate("PP"):
            q = self._split_heads(self.q_proj(x_txt))
            k = self._split_heads(k)
            v = self._split_heads(v)
            o = torch.zeros_like(q)
            m = torch.full(q.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)
            l = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)
            for _ in range(self.n_steps):
                with annotate("CP"):
                    o, m, l = ring_step(q, k, v, o, m, l, self.scale)
            o = o / l.unsqueeze(-1)
            b, h, s, d = o.shape
            t = self.o_proj(o.transpose(1, 2).reshape(b, s, h * d))
            with annotate("TP"):
                t = t.sum(dim=1) * (1.0 / self.seq_total)
        with annotate("PP"):
            y = a + t
            for layer in self.dec:
                y = layer(y)
        return y
