"""local_map in_grad_placements, 2-D redistribution plans, FSDP2 lifetime knobs: measured on a fake mesh (CPU)."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Shard, Replicate, Partial
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental import local_map
dist.init_process_group("fake", rank=0, world_size=4, store=FakeStore())
def cc(m): return {str(k).split(".")[-1]: v for k, v in m.get_comm_counts().items()}
mesh1 = DeviceMesh("cpu", [0, 1])
print("== local_map: in_grad_placements is a declaration")
def mlp_local(x, w1, w2): return F.gelu(x @ w1.t()) @ w2.t()
for xg in (Replicate(), Partial()):
    f_lm = local_map(mlp_local, out_placements=[Partial()], in_placements=([Replicate()], [Shard(0)], [Shard(1)]),
                     in_grad_placements=([xg], [Shard(0)], [Shard(1)]), device_mesh=mesh1)
    xr = distribute_tensor(torch.randn(2, 16), mesh1, [Replicate()]).requires_grad_()
    w1 = distribute_tensor(torch.randn(64, 16), mesh1, [Shard(0)]).requires_grad_(); w2 = distribute_tensor(torch.randn(16, 64), mesh1, [Shard(1)]).requires_grad_()
    z = f_lm(xr, w1, w2).redistribute(mesh1, [Replicate()])
    with CommDebugMode() as b: z.sum().backward()
    with CommDebugMode() as g: xr.grad.full_tensor()
    print(f"  in_grad={xg}: bwd {cc(b)}, grad x {xr.grad.placements}, making it whole {cc(g)}")

print("== 2-D mesh (dp=2, tp=2): redistribution plans")
from torch.distributed.tensor._redistribute import _gen_transform_infos
mesh2 = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"))
for src, dst in [((Shard(0), Shard(1)), (Replicate(), Replicate())), ((Partial(), Shard(1)), (Replicate(), Replicate())),
                 ((Shard(0), Shard(0)), (Replicate(), Replicate())), ((Replicate(), Shard(0)), (Shard(0), Replicate()))]:
    t = distribute_tensor(torch.randn(8, 8), mesh2, list(src)) if not any(isinstance(p, Partial) for p in src) else DTensor.from_local(torch.randn(8, 4), mesh2, list(src), run_check=False)
    infos = _gen_transform_infos(t._spec, type(t._spec)(mesh2, tuple(dst), tensor_meta=t._spec.tensor_meta))
    with CommDebugMode() as m: t.redistribute(mesh2, list(dst))
    print(f"  {src} -> {dst}: steps {[ (i.mesh_dim, str(i.src_dst_placements[0]), str(i.src_dst_placements[1])) for i in infos]} comm {cc(m)}")
print("  mesh2['tp'] ranks for rank 0:", mesh2["tp"].mesh.tolist(), " mesh2['dp']:", mesh2["dp"].mesh.tolist())

print("== FSDP2 fully_shard: reshard_after_forward = lifetime")
from torch.distributed.fsdp import fully_shard
for raf in (True, False):
    torch.manual_seed(0)
    model = nn.Sequential(*[nn.Linear(32, 32) for _ in range(3)])
    for layer in model: fully_shard(layer, mesh=mesh1, reshard_after_forward=raf)
    fully_shard(model, mesh=mesh1, reshard_after_forward=raf)
    w = model[0].weight
    x = torch.randn(4, 32)
    with CommDebugMode() as f: y = model(x)
    with CommDebugMode() as b: y.sum().backward()
    print(f"  reshard_after_forward={raf}: param {type(w).__name__} {w.placements} local {tuple(w.to_local().shape)}; fwd {cc(f)} bwd {cc(b)}")
