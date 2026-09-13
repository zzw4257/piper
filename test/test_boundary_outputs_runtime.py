"""The runtime half of the forwarded-boundary rule (log F58).

`boundary_outputs` decides, by object identity, which of a segment's outputs it
merely forwarded. `ComputeExecutor.backward` then keys on
`pre_detach_outs[i] is detached_outs[i]` to skip driving those pairs. The two
must agree: if forward stops detaching a value but backward still drives it, the
downstream gradient is added twice and only shows up from the second optimizer
step onward.
"""
import torch

from src.executors import boundary_outputs


def test_forwarded_output_is_emitted_as_the_same_object() -> None:
    x = torch.randn(4, requires_grad=True)
    produced = x * 2
    out = boundary_outputs([produced, x], [x])
    assert out[0] is not produced and out[0].requires_grad      # produced: detached
    assert out[1] is x                                          # forwarded: untouched


def test_the_pairing_rule_backward_relies_on() -> None:
    """`p is d` must be true exactly for forwarded outputs."""
    x, y = torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)
    outs = [x * 2, x, y, (x + y).detach()]
    det = boundary_outputs(outs, [x, y])
    forwarded = [p is d for p, d in zip(outs, det)]
    assert forwarded == [False, True, True, True], forwarded
    # the last one does not require grad, so it is passed through and never driven
    assert not outs[3].requires_grad


def test_non_tensors_and_grad_free_tensors_pass_through() -> None:
    x = torch.randn(4, requires_grad=True)
    c = torch.randn(4)
    out = boundary_outputs([c, None, 7, x], [x])
    assert out[0] is c and out[1] is None and out[2] == 7 and out[3] is x


def test_gradient_is_counted_once_end_to_end() -> None:
    """The whole point, in miniature: a value both used by a segment and
    forwarded past it must end with one in-segment contribution plus one
    downstream contribution, not two of either."""
    k = torch.randn(4, requires_grad=True)
    q = torch.randn(4, requires_grad=True)
    produced = (q * k).sum()          # uses k
    outs = [produced, k]              # and forwards k
    det = boundary_outputs(outs, [q, k])

    downstream = torch.randn(4)
    det[1].grad = downstream.clone()  # what the executor assigns from inp_grads

    pairs = [(p, d.grad) for p, d in zip(outs, det)
             if isinstance(d, torch.Tensor) and d.requires_grad
             and d.grad is not None and p is not d]
    assert [id(p) for p, _ in pairs] == []       # produced has no grad assigned yet
    det[0].grad = torch.tensor(1.0)
    pairs = [(p, d.grad) for p, d in zip(outs, det)
             if isinstance(d, torch.Tensor) and d.requires_grad
             and d.grad is not None and p is not d]
    torch.autograd.backward([p for p, _ in pairs], [g for _, g in pairs])
    assert torch.allclose(k.grad, downstream + q), (k.grad, downstream + q)
