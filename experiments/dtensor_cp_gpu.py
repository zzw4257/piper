"""F92 on GPUs: DTensor's own plan for sequence-split attention against gathering K and V, timed.

    torchrun --nproc_per_node 4 experiments/dtensor_cp_gpu.py

Attention written with matmuls on q, k, v Shard(seq) over 4 GPUs (real NCCL). For each (head dim,
tokens per rank): the collectives DTensor issues (forward + backward) and the median time of
forward + backward, against the plan Piper writes (gather K and V once, reduce-scatter their grads).
"""
import os, re, statistics, sys
if os.environ.get("ONE_GPU") == "1":
    # each rank sees only its own card, as under Ray actors or per-rank SLURM binding
    vis = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = vis[int(os.environ["LOCAL_RANK"])]
import torch, torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Shard, Replicate
from torch.distributed.tensor.debug import CommDebugMode
sys.argv = [a for a in sys.argv if not a.startswith("--temp-dir")]
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r % torch.cuda.device_count())
mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))
B, H = 4, 8
def step(q, k, v, d, force):
    kk, vv = (k.redistribute(mesh, [Replicate()]), v.redistribute(mesh, [Replicate()])) if force else (k, v)
    o = ((q @ kk.transpose(-1, -2) / d ** 0.5).softmax(-1) @ vv).redistribute(mesh, [Shard(2)])
    o.to_local().sum().backward()
def bench(sl, d, force, iters=20):
    q, k, v = (DTensor.from_local(torch.randn(B, H, sl, d, device="cuda"), mesh, [Shard(2)], run_check=False).requires_grad_() for _ in range(3))
    with CommDebugMode() as cm: step(q, k, v, d, force)
    counts = {str(a).split(".")[-1].replace("_into_tensor", "").replace("_tensor", ""): n for a, n in cm.get_comm_counts().items()}
    for _ in range(3): step(q, k, v, d, force)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); dist.barrier(); t = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        t.record(); step(q, k, v, d, force); e.record(); torch.cuda.synchronize(); ts.append(t.elapsed_time(e))
    tt = torch.tensor([statistics.median(ts)], device="cuda"); dist.all_reduce(tt, op=dist.ReduceOp.MAX)
    return counts, tt.item()
from torch.distributed.tensor._collective_utils import MeshTopoInfo
if r == 0:
    t = MeshTopoInfo.build_from_mesh(mesh)
    print(f"visible GPUs per rank {torch.cuda.device_count()}: DTensor assumes bandwidth {t.mesh_dim_bandwidth} GB/s, latency {t.mesh_dim_latency} us", flush=True)
for d, sl in [(64, 16), (64, 32), (64, 64), (64, 128), (128, 16), (128, 32), (128, 64)]:
    c1, t1 = bench(sl, d, False); c2, t2 = bench(sl, d, True)
    if r == 0:
        print(f"d={d:4d} tokens/rank={sl:5d} | DTensor {c1} {t1:7.2f} ms | gather K,V {c2} {t2:7.2f} ms | ratio {t1/t2:4.2f}", flush=True)
if r == 0: print("PASS")
dist.destroy_process_group()
