"""The forward TP all-reduce sums into the producer's own output buffer (log F101)."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from src.executors import CommunicationExecutor


@pytest.fixture(scope="module")
def comm():
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    return CommunicationExecutor(SimpleNamespace(group_for=lambda axis, default: None, ep_group=None), None, None)


def test_in_place_reduces_the_buffer_itself(comm) -> None:
    x = torch.randn(4, 3, requires_grad=True)
    y = x @ torch.randn(3, 5)          # a matmul saves its inputs, not its output
    d = y.detach().requires_grad_(True)
    assert comm.all_reduce_activation(d, None, in_place=True) is d
    y.backward(torch.ones_like(y))     # the producer's backward is unaffected
    assert x.grad is not None


def test_a_producer_that_saved_its_output_is_refused(comm) -> None:
    x = torch.randn(4, requires_grad=True)
    y = torch.tanh(x)                  # tanh saves its output for the backward
    d = y.detach().requires_grad_(True)
    comm.all_reduce_activation(d, None, in_place=True)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward(torch.ones_like(y))


def test_default_still_returns_a_copy(comm) -> None:
    d = torch.randn(4).requires_grad_(True)
    assert comm.all_reduce_activation(d, None) is not d
