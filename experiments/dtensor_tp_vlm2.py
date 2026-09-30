"""The SmolVLM2 step with PyTorch's own TP (DTensor ``parallelize_module``), as the baseline for Piper's TP.

    CUDA_VISIBLE_DEVICES=0,2 PYTHONPATH=examples:. torchrun --nproc-per-node 2 experiments/dtensor_tp_vlm2.py --data DIR

Same model, weights, batch and loop as eager_vlm2.py. The plan is torchtitan's without sequence
parallel: q/k/v, fc1, gate/up colwise; out/o/fc2/down rowwise; everything else replicated, and
attention runs on local heads (``use_local_output=True``). Prints rank 0's losses, median step
time, and the peak memory of every rank.
"""
import argparse, os, statistics, sys, time
sys.argv = [x for x in sys.argv if not x.startswith("--temp-dir") and not x.startswith("/var/tmp/ziweizho-ray")]
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
sys.path[:0] = ["examples", "."]
from models.smolvlm import SmolVLM, lm_loss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--batch-size", type=int, default=3); ap.add_argument("--mb", type=int, default=2)
ap.add_argument("--mem", action="store_true"); ap.add_argument("--iters", type=int, default=6); ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-5)
a, _ = ap.parse_known_args()
rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
mesh = init_device_mesh("cuda", (tp,))
d = torch.load(os.path.join(a.data, "data.pt"))
b, T = a.batch_size, d.get("tiles", 1)
m = SmolVLM(d["image_offset"], config=d.get("config"))
m.load_state_dict(torch.load(os.path.join(a.data, "weights.pt")))
m = m.cuda()
col, row = ColwiseParallel, RowwiseParallel
for L in m.model.vision_model.encoder.layers:
    parallelize_module(L, mesh, {"self_attn.q_proj": col(), "self_attn.k_proj": col(), "self_attn.v_proj": col(),
                                 "self_attn.out_proj": row(), "mlp.fc1": col(), "mlp.fc2": row()})
    L.self_attn.heads //= tp
for L in m.model.text_model.layers:
    parallelize_module(L, mesh, {"self_attn.q_proj": col(), "self_attn.k_proj": col(), "self_attn.v_proj": col(),
                                 "self_attn.o_proj": row(), "mlp.gate_proj": col(), "mlp.up_proj": col(),
                                 "mlp.down_proj": row()})
    L.self_attn.heads //= tp
    L.self_attn.kv_heads //= tp
# fused Adam, as Piper's actors use; one optimizer per kind, since one call cannot mix DTensor and Tensor
groups = [[p for p in m.parameters() if isinstance(p, DTensor) == k] for k in (True, False)]
opts = [torch.optim.Adam(g, lr=a.lr, fused=True) for g in groups if g]
px = ((d["images"][: b * T].permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).cuda()
ids, lab = d["input_ids"][:b].cuda(), d["labels"][:b].cuda()
extra = [d["img_index"][:b].cuda()] if "img_index" in d else []
del d
losses, times = [], []
for it in range(a.warmup + a.iters):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    mem = lambda tag: a.mem and rank == 0 and it == a.warmup + a.iters - 1 and print(  # noqa: E731
        f"MEM {tag:14s} alloc {torch.cuda.memory_allocated() / 2**30:6.1f}  peak {torch.cuda.max_memory_allocated() / 2**30:6.1f}")
    mem("step start")
    for k in range(a.mb):
        loss = lm_loss(m(px, ids, *extra), lab)
        mem(f"mb{k} fwd")
        loss.backward()
        mem(f"mb{k} bwd")
        if it >= a.warmup:
            losses.append(loss.item())
    for o in opts:
        o.step(); o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if it >= a.warmup:
        times.append(time.perf_counter() - t0)
    if it == a.warmup - 1:
        torch.cuda.reset_peak_memory_stats()
peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device="cuda")
peaks = [torch.zeros_like(peak) for _ in range(tp)]
dist.all_gather(peaks, peak)
if rank == 0:
    print(f"DTENSOR tp={tp} losses", [round(x, 6) for x in losses])
    print(f"DTENSOR tp={tp} median step {statistics.median(times) * 1e3:.1f} ms  peak GB {[round(p.item(), 1) for p in peaks]}")
    print("PASS")
dist.destroy_process_group()
