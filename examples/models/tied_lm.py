"""A small language model with tied input and output embeddings: the case of issue #13.

    tokens -> [embed (W)] -> [layers ...] -> [norm, logits = h @ W.T] -> next-token loss

The embedding and the output projection are the same parameter W. Split into PP regions,
W is read by the first and the last region, which sit on different pipeline ranks: each
rank holds a copy and the runtime sums the copies' gradients (log F83).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate


def global_weights(vocab: int, dim: int, layers: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {"embed.weight": torch.randn(vocab, dim, generator=g) * 0.02}
    for i in range(layers):
        out[f"layers.{i}.fc1.weight"] = torch.randn(4 * dim, dim, generator=g) / dim ** 0.5
        out[f"layers.{i}.fc2.weight"] = torch.randn(dim, 4 * dim, generator=g) / (4 * dim) ** 0.5
    return out


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(x)))


class TiedLM(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, stages: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(Block(dim) for _ in range(layers))
        self.stages = stages

    def forward(self, tokens):
        per = -(-len(self.layers) // max(1, self.stages - 2)) if self.stages > 2 else len(self.layers)
        with annotate("PP"):
            h = self.embed(tokens)
            if self.stages <= 2:
                for layer in self.layers:
                    h = layer(h)
        if self.stages > 2:
            for s in range(self.stages - 2):
                with annotate("PP"):
                    for layer in self.layers[s * per:(s + 1) * per]:
                        h = layer(h)
        with annotate("PP"):
            logits = F.layer_norm(h, h.shape[-1:]) @ self.embed.weight.t()
        return logits


def lm_loss(logits, tokens):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1).long())
