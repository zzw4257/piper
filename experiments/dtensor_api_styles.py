"""ParallelStyles, loss_parallel, vocab-parallel embedding, local_map, 2-D redistribution: measured on a fake mesh."""
import copy, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Shard, Replicate, Partial
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (parallelize_module, ColwiseParallel, RowwiseParallel,
    SequenceParallel, PrepareModuleInput, loss_parallel)
dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
mesh = DeviceMesh("cpu", [0, 1])
def cc(m): return {str(k).split(".")[-1]: v for k, v in m.get_comm_counts().items()}
torch.manual_seed(0)
D, H, B, S = 16, 64, 2, 8

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.norm = nn.LayerNorm(D); s.up = nn.Linear(D, H, bias=False); s.down = nn.Linear(H, D, bias=False)
    def forward(s, x): return x + s.down(F.gelu(s.up(s.norm(x))))

def run(plan, x, name):
    m = parallelize_module(Block(), mesh, plan)
    for n, p in m.named_parameters(): pass
    with CommDebugMode() as f: y = m(x)
    ytype = type(y).__name__ + (f" {y.placements}" if isinstance(y, DTensor) else f" {tuple(y.shape)}")
    loss = (y.full_tensor() if isinstance(y, DTensor) else y).sum()
    with CommDebugMode() as b: loss.backward()
    ps = {n: (str(p.placements[0]) if isinstance(p, DTensor) else "plain") for n, p in m.named_parameters()}
    print(f"{name}\n   params {ps}\n   out {ytype}\n   fwd {cc(f)}  bwd {cc(b)}")

x = torch.randn(B, S, D)
print("== A. Megatron TP: up Colwise, down Rowwise (defaults)")
run({"up": ColwiseParallel(), "down": RowwiseParallel()}, x, "  TP")
print("== B. + sequence parallel: norm SequenceParallel, up Colwise(input Shard(1)), down Rowwise(output Shard(1))")
xs = x.chunk(2, dim=1)[0].contiguous()
run({"norm": SequenceParallel(), "up": ColwiseParallel(input_layouts=Shard(1)), "down": RowwiseParallel(output_layouts=Shard(1))}, xs, "  TP+SP")
print("== C. use_local_output=False on down")
try:
    run({"up": ColwiseParallel(), "down": RowwiseParallel(use_local_output=False)}, x, "  TP, DTensor out")
except RuntimeError as e:
    print("   error:", str(e).splitlines()[0][:110])

print("== D. vocab-parallel embedding: RowwiseParallel on nn.Embedding")
emb = parallelize_module(nn.Embedding(100, D), mesh, RowwiseParallel(input_layouts=Replicate()))
print("   weight", emb.weight.placements, tuple(emb.weight.to_local().shape))
ids = torch.randint(0, 100, (B, S))
with CommDebugMode() as f:
    e = emb(ids)
print("   fwd", cc(f))

print("== E. loss_parallel: lm_head Colwise with Shard(-1) logits")
head = parallelize_module(nn.Linear(D, 100, bias=False), mesh, ColwiseParallel(use_local_output=False))
h = torch.randn(B, S, D, requires_grad=True); tgt = torch.randint(0, 100, (B * S,))
with loss_parallel():
    with CommDebugMode() as f:
        logits = head(h); print("   logits", logits.placements, tuple(logits.to_local().shape))
        l = F.cross_entropy(logits.flatten(0, 1), tgt)
    print("   fwd", cc(f), "loss", type(l).__name__, l.placements)
    with CommDebugMode() as b: l.backward()
    print("   bwd", cc(b))
with CommDebugMode() as f:
    lf = F.cross_entropy(head(h).full_tensor().flatten(0, 1), tgt)
print("   without loss_parallel (gather full logits):", cc(f))

print("== F. local_map: a function on local tensors with a declared contract")
from torch.distributed.tensor.experimental import local_map
def mlp_local(x, w1, w2): return F.gelu(x @ w1.t()) @ w2.t()
f_lm = local_map(mlp_local, out_placements=[Partial()], in_placements=([Replicate()], [Shard(0)], [Shard(1)]),
                 in_grad_placements=([Replicate()], [Shard(0)], [Shard(1)]), device_mesh=mesh)
xr = distribute_tensor(torch.randn(B, D), mesh, [Replicate()]).requires_grad_()
w1 = distribute_tensor(torch.randn(H, D), mesh, [Shard(0)]).requires_grad_(); w2 = distribute_tensor(torch.randn(D, H), mesh, [Shard(1)]).requires_grad_()
with CommDebugMode() as f:
    y = f_lm(xr, w1, w2); print("   out", y.placements); z = y.redistribute(mesh, [Replicate()])
with CommDebugMode() as b: z.sum().backward()
print("   fwd", cc(f), "bwd", cc(b), "grad x", xr.grad.placements)
