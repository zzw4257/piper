"""Head-parallel (DeepSpeed-Ulysses) attention for context parallelism, written for Piper (log F96).

Same math as models/ring_attn.py: q = x W_q, full non-causal attention over the whole
sequence, o W_o. Each rank holds a chunk of the sequence. Around the attention region the
sequence split becomes a head split and back, by one all-to-all each way.

The layout makes that all-to-all shape-preserving, so it is exactly Piper's EP exchange
(`shard`: all_to_all_single along dim 0, equal splits):

    before: [n, 3, b, h/n, s/n, d]   dim 0 = head group j, rows = this rank's tokens
    after:  [n, 3, b, h/n, s/n, d]   dim 0 = sequence chunk i, heads = this rank's group

With n = 1 the exchange is the identity and the model is dense attention, the baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate


class UlyssesAttn(nn.Module):
    def __init__(self, dim: int, heads: int, n: int, fused: bool = True):
        super().__init__()
        assert dim % heads == 0 and heads % n == 0, "heads must split evenly across ranks"
        self.dim, self.heads, self.n = dim, heads, n
        self.head_dim = dim // heads
        self.fused = fused  # False: softmax(q k^T) v written out, the ring model's kernel mix
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, k, v):
        b, s, _ = x.shape                      # s: this rank's tokens
        n, g, hd = self.n, self.heads // self.n, self.head_dim
        with annotate("PP"):
            # [b, s, dim] -> [b, n, g, s, hd] per tensor, head groups leading, packed as one tensor
            split = lambda t: t.view(b, s, n, g, hd).permute(2, 0, 3, 1, 4)  # noqa: E731
            qkv = torch.stack([split(self.q_proj(x)), split(k), split(v)], dim=1).contiguous()
            with annotate("SP"):
                # after the all-to-all: dim 0 indexes sequence chunks; gather them into the sequence
                t = qkv.permute(1, 2, 3, 0, 4, 5).reshape(3, b, g, n * s, hd)
                if self.fused:
                    o = F.scaled_dot_product_attention(t[0], t[1], t[2])      # [b, g, n*s, hd]
                else:
                    o = torch.softmax(t[0] @ t[1].transpose(-1, -2) / hd ** 0.5, dim=-1) @ t[2]
                o = o.view(b, g, n, s, hd).permute(2, 0, 1, 3, 4).contiguous()  # chunk-major for the return trip
            # after the return all-to-all: dim 0 indexes head groups again
            o = o.permute(1, 3, 0, 2, 4).reshape(b, s, self.dim)
            return self.o_proj(o)
