"""Where each collective comes from under DTensor (log F90). CPU only, fake 2-rank mesh.

    python experiments/dtensor_walk.py

A TP MLP op by op: placements after each op, and the collectives in the forward and
backward. Then the same with the input declared at a region boundary (from_local), and
data parallelism with the batch declared whole or split.
"""
import torch, torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Shard, Replicate
from torch.distributed.tensor.debug import CommDebugMode
dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
mesh = DeviceMesh("cpu", [0, 1])
d, h, b = 8, 32, 4
x = distribute_tensor(torch.randn(b, d), mesh, [Replicate()]).requires_grad_()
w1 = distribute_tensor(torch.randn(h, d), mesh, [Shard(0)]).requires_grad_()   # colwise
w2 = distribute_tensor(torch.randn(d, h), mesh, [Shard(1)]).requires_grad_()   # rowwise
def pl(t): return str(t.placements[0])
with CommDebugMode() as fwd:
    a = x @ w1.t();            print("up   x@w1^T      ->", pl(a), tuple(a.to_local().shape))
    g = torch.nn.functional.gelu(a); print("gelu            ->", pl(g))
    y = g @ w2.t();            print("down g@w2^T      ->", pl(y), tuple(y.to_local().shape))
    z = y.redistribute(mesh, [Replicate()]); print("P->R redistribute ->", pl(z))
print("forward collectives:", {str(k): v for k, v in fwd.get_comm_counts().items()})
with CommDebugMode() as bwd:
    z.sum().backward()
print("backward collectives:", {str(k): v for k, v in bwd.get_comm_counts().items()})
print("grad x:", pl(x.grad), " grad w1:", pl(w1.grad), " grad w2:", pl(w2.grad))
# data parallel: forgetting the batch split
xb = distribute_tensor(torch.randn(b, d), mesh, [Replicate()])
wr = distribute_tensor(torch.randn(d, d), mesh, [Replicate()]).requires_grad_()
with CommDebugMode() as c1: (xb @ wr).sum().backward()
xs = distribute_tensor(torch.randn(b, d), mesh, [Shard(0)])
wr2 = distribute_tensor(torch.randn(d, d), mesh, [Replicate()]).requires_grad_()
with CommDebugMode() as c2:
    out = (xs @ wr2); print("DP: x Shard(0) @ w Replicate ->", pl(out)); out.sum().backward()
print("DP batch replicated: grad w", pl(wr.grad), {str(k): v for k, v in c1.get_comm_counts().items()})
print("DP batch Shard(0):   grad w", pl(wr2.grad), {str(k): v for k, v in c2.get_comm_counts().items()})
from torch.distributed.tensor import DTensor
print("--- boundary declared with from_local (what a region boundary does)")
xl = torch.randn(b, d, requires_grad=True)
x2 = DTensor.from_local(xl, mesh, [Replicate()])
w1b = distribute_tensor(torch.randn(h, d), mesh, [Shard(0)]).requires_grad_()
w2b = distribute_tensor(torch.randn(d, h), mesh, [Shard(1)]).requires_grad_()
with CommDebugMode() as f2:
    out = (torch.nn.functional.gelu(x2 @ w1b.t()) @ w2b.t()).redistribute(mesh, [Replicate()]).to_local()
print("fwd:", {str(k): v for k, v in f2.get_comm_counts().items()})
with CommDebugMode() as b2:
    out.sum().backward()
print("bwd:", {str(k): v for k, v in b2.get_comm_counts().items()}, " grad of local x:", tuple(xl.grad.shape))
print("--- DP: the weight gradient declared Replicate (what an optimizer on whole weights needs)")
with CommDebugMode() as c3:
    gr = wr2.grad.redistribute(mesh, [Replicate()])
print("DP grad P->R:", {str(k): v for k, v in c3.get_comm_counts().items()}, pl(gr))
