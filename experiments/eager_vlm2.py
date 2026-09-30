"""F99: the same SmolVLM2 step in plain PyTorch on one GPU, as the baseline for Piper's single-GPU run.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=examples:. python experiments/eager_vlm2.py --data DIR --batch-size 3 --mb 2

m microbatches of the SAME batch (as Piper's `split`, log F74), gradients accumulated, one Adam step.
Prints per-step losses (to compare with Piper's), median step time and peak memory.
"""
import argparse, os, statistics, sys, time
sys.argv = [x for x in sys.argv if not x.startswith("--temp-dir") and not x.startswith("/var/tmp/ziweizho-ray")]
import torch
sys.path[:0] = ["examples", "."]
from models.smolvlm import SmolVLM, lm_loss

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--batch-size", type=int, default=3); ap.add_argument("--mb", type=int, default=2)
ap.add_argument("--iters", type=int, default=4); ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--mem", action="store_true", help="print allocated/peak GB at each phase of the last step")
a, _ = ap.parse_known_args()
d = torch.load(os.path.join(a.data, "data.pt"))
b, T = a.batch_size, d.get("tiles", 1)
m = SmolVLM(d["image_offset"], config=d.get("config")).cuda()
m.load_state_dict(torch.load(os.path.join(a.data, "weights.pt")))
fwd = torch.compile(m) if a.compile else m
opt = torch.optim.Adam(m.parameters(), lr=a.lr, fused=True)  # as Piper's actors
px = ((d["images"][: b * T].permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).cuda()
ids, lab = d["input_ids"][:b].cuda(), d["labels"][:b].cuda()
extra = [d["img_index"][:b].cuda()] if "img_index" in d else []
losses, times = [], []
for it in range(a.warmup + a.iters):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    mem = lambda tag: a.mem and it == a.warmup + a.iters - 1 and print(  # noqa: E731
        f"MEM {tag:14s} alloc {torch.cuda.memory_allocated() / 2**30:6.1f}  peak {torch.cuda.max_memory_allocated() / 2**30:6.1f}")
    mem("step start")
    for k in range(a.mb):
        loss = lm_loss(fwd(px, ids, *extra), lab)
        mem(f"mb{k} fwd")
        loss.backward()
        mem(f"mb{k} bwd")
        if it >= a.warmup:
            losses.append(loss.item())
    opt.step(); mem("opt.step"); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if it >= a.warmup:
        times.append(time.perf_counter() - t0)
    if it == a.warmup - 1:
        torch.cuda.reset_peak_memory_stats()
print("EAGER losses", [round(x, 6) for x in losses])
print(f"EAGER median step {statistics.median(times) * 1e3:.1f} ms  peak GB {torch.cuda.max_memory_allocated() / 2**30:.1f}")
print("PASS")
