"""Gate the online-softmax accumulation itself, independently of Piper.

If this fails, a CP bug and a Piper bug are indistinguishable in the loss.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.ring_attn import ring_step  # noqa: E402


def _ring_reference(q, k, v, n_chunks):
    scale = 1.0 / math.sqrt(q.shape[-1])
    o = torch.zeros_like(q)
    m = torch.full(q.shape[:-1], float("-inf"), dtype=q.dtype)
    l = torch.zeros(q.shape[:-1], dtype=q.dtype)
    for kc, vc in zip(k.chunk(n_chunks, dim=-2), v.chunk(n_chunks, dim=-2)):
        o, m, l = ring_step(q, kc, vc, o, m, l, scale)
    return o / l.unsqueeze(-1)


def test_ring_accumulation_matches_dense_attention() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8, dtype=torch.float64) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    for n_chunks in (1, 2, 4, 8):
        got = _ring_reference(q, k, v, n_chunks)
        assert torch.allclose(got, ref, atol=1e-12), (
            f"{n_chunks} chunks: max err {(got - ref).abs().max().item():.3e}"
        )


def test_chunk_order_does_not_matter() -> None:
    """Ring attention visits chunks in a rank-dependent order; the accumulator
    must be order-invariant or every rank computes something different."""
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 12, 8, dtype=torch.float64) for _ in range(3))
    scale = 1.0 / math.sqrt(8)
    chunks = list(zip(k.chunk(4, dim=-2), v.chunk(4, dim=-2)))

    def run(order):
        o = torch.zeros_like(q)
        m = torch.full(q.shape[:-1], float("-inf"), dtype=q.dtype)
        l = torch.zeros(q.shape[:-1], dtype=q.dtype)
        for i in order:
            o, m, l = ring_step(q, *chunks[i], o, m, l, scale)
        return o / l.unsqueeze(-1)

    base = run([0, 1, 2, 3])
    for order in ([1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]):
        assert torch.allclose(run(order), base, atol=1e-12)
