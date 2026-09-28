"""DTensor API probes on a fake mesh (CPU): what each public API does, measured."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import (DeviceMesh, distribute_tensor, DTensor, Shard, Replicate, Partial)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (parallelize_module, ColwiseParallel, RowwiseParallel,
    SequenceParallel, PrepareModuleInput, loss_parallel)
dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
mesh = DeviceMesh("cpu", [0, 1])
def cc(m): return {str(k).split(".")[-1]: v for k, v in m.get_comm_counts().items()}
torch.manual_seed(0)

print("== 1. uneven Shard: global 5 rows on 2 ranks")
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
for r in (0, 1):
    print("  rank", r, compute_local_shape_and_global_offset((5, 4), mesh, [Shard(0)]) if r == 0 else "(fake pg is rank 0 only)")
print("  torch.chunk sizes:", [c.shape[0] for c in torch.randn(5).chunk(2)])

print("== 2. Partial reduce ops")
for op in ("sum", "avg", "max", "min"):
    t = DTensor.from_local(torch.ones(3), mesh, [Partial(op)], run_check=False)
    with CommDebugMode() as m: r = t.redistribute(mesh, [Replicate()])
    print(f"  Partial({op}) -> R:", cc(m))
with CommDebugMode() as m: DTensor.from_local(torch.ones(4), mesh, [Partial()], run_check=False).redistribute(mesh, [Shard(0)])
print("  Partial -> Shard(0):", cc(m))

print("== 3. from_local run_check / to_local grad_placements / full_tensor")
with CommDebugMode() as m: DTensor.from_local(torch.ones(2, 2), mesh, [Replicate()], run_check=True)
print("  from_local(run_check=True):", cc(m))
with CommDebugMode() as m: DTensor.from_local(torch.ones(2, 2), mesh, [Replicate()], run_check=False)
print("  from_local(run_check=False):", cc(m))
w = distribute_tensor(torch.randn(8, 4), mesh, [Shard(0)]).requires_grad_()
for gp in (None, [Replicate()], [Partial()]):
    y = (w * 2)
    loc = y.to_local(grad_placements=gp) if gp else y.to_local()
    with CommDebugMode() as m: loc.sum().backward()
    print(f"  to_local(grad_placements={gp}) backward:", cc(m), "grad placement", w.grad.placements); w.grad = None
with CommDebugMode() as m: full = w.full_tensor()
print("  full_tensor of Shard(0):", cc(m), tuple(full.shape))
with CommDebugMode() as m: distribute_tensor(torch.randn(8, 4), mesh, [Shard(0)])
print("  distribute_tensor Shard(0):", cc(m))

print("== 4. one op, all candidate strategies: aten.mm on a 1-D mesh")
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy
from torch.distributed.tensor._ops._matrix_ops import mm_strategy
a = distribute_tensor(torch.randn(4, 8), mesh, [Replicate()]); b = distribute_tensor(torch.randn(8, 6), mesh, [Shard(1)])
schema = OpSchema(torch.ops.aten.mm.default, (OpStrategy([a._spec and __import__('torch').distributed.tensor._op_schema.OpSpec(a._spec)]), OpStrategy([__import__('torch').distributed.tensor._op_schema.OpSpec(b._spec)])), {})
st = mm_strategy(schema)
for s in st.strategies:
    print("  ", [str(x.placements[0]) for x in s.input_specs], "->", str(s.output_spec.placements[0]), "cost", [round(float(c), 1) for c in sum(s.redistribute_cost, [])])
