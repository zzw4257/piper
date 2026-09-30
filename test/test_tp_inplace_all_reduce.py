"""The forward TP all-reduce sums into the producer's own output buffer (log F101), and a TP output
nothing saved can be freed after its consumer (log F102)."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from src.executors import CommunicationExecutor, _free_unsaved


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


def _case(saved_by_consumer=False, consumer_opaque=False, returned_twice=False):
    x = torch.randn(4, 3, requires_grad=True)
    p = x @ torch.randn(3, 5)               # the TP output: the matmul saved x, not p
    d = p.detach().requires_grad_(True)
    c = d.detach().requires_grad_(True)     # the consumer's leaf
    out = c + 1                             # the residual add saves nothing
    fwd_out = {"pre_detach_outs": [out], "opaque_saves": consumer_opaque,
               "saved_storages": {c.untyped_storage().data_ptr()} if saved_by_consumer else set()}
    producer_outs = [p, p.view(-1)] if returned_twice else [p]
    _free_unsaved([(d, set(), producer_outs)], fwd_out, None)
    return p.untyped_storage().nbytes(), x, p


def test_an_output_nothing_saved_is_freed_and_backward_still_runs() -> None:
    nbytes, x, p = _case()
    assert nbytes == 0
    p.backward(torch.ones(4, 5))            # only the shape is needed
    assert x.grad is not None


@pytest.mark.parametrize("kw", [{"saved_by_consumer": True}, {"consumer_opaque": True}, {"returned_twice": True}])
def test_an_output_that_may_still_be_read_is_kept(kw) -> None:
    assert _case(**kw)[0] > 0
