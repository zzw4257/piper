"""The SmolVLM2 step with PyTorch's own pipelining (``torch.distributed.pipelining``), as the
baseline for Piper's two-GPU pipeline: vision tower on one GPU, decoder on the other.

    CUDA_VISIBLE_DEVICES=0,2 PYTHONPATH=examples:. torchrun --nproc-per-node 2 \\
        experiments/pipelining_vlm2.py --data DIR --batch-size 3 --mb 3 --schedule 1f1b

Piper's ``split`` repeats the batch in every microbatch (log F74), so the same work here is
``mb`` copies of the batch, split into ``mb`` microbatches: every microbatch loss equals
eager_vlm2.py's. Stage 0 = vision, connector, text embedding and the image scatter; stage 1 =
decoder, norm, lm_head. (Piper's stage 0 ends before the scatter, so it sends the image
features and the text embedding, 71 MB, where this sends the merged hidden state, 37 MB.)
"""
import argparse, os, statistics, sys, time
sys.argv = [x for x in sys.argv if not x.startswith("--temp-dir") and not x.startswith("/var/tmp/ziweizho-ray")]
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
sys.path[:0] = ["examples", "."]
from models.smolvlm import SmolVLM, lm_loss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--batch-size", type=int, default=3); ap.add_argument("--mb", type=int, default=3)
ap.add_argument("--schedule", choices=["gpipe", "1f1b"], default="1f1b")
ap.add_argument("--iters", type=int, default=6); ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-5)
a, _ = ap.parse_known_args()
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
assert world == 2
dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(dev)
dist.init_process_group("nccl", device_id=dev)
d = torch.load(os.path.join(a.data, "data.pt"))
b, T = a.batch_size, d.get("tiles", 1)
m = SmolVLM(d["image_offset"], config=d.get("config"))
m.load_state_dict(torch.load(os.path.join(a.data, "weights.pt")))


class Vision(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.vm, self.connector, self.embed = m.model.vision_model, m.model.connector, m.model.text_model.embed_tokens

    def forward(self, px, ids, img_index):
        img = self.connector(self.vm(px))
        txt = self.embed(ids)
        img = img.reshape(ids.shape[0], -1, img.shape[-1])
        return txt.scatter(1, img_index.unsqueeze(-1).expand(-1, -1, img.shape[-1]), img)


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.layers, self.norm, self.lm_head, self._rope = m.model.text_model.layers, m.model.text_model.norm, m.lm_head, m._rope

    def forward(self, h):
        cos, sin = self._rope(h.shape[1], h.device, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.lm_head(self.norm(h))


mod = (Vision(m) if rank == 0 else Decoder(m)).to(dev)
del m
stage = PipelineStage(mod, rank, 2, dev)
# scale_grads=False: sum microbatch gradients, as eager and Piper do; the default averages them,
# which Adam absorbs except through eps (losses 5e-5 off)
sched = (ScheduleGPipe if a.schedule == "gpipe" else Schedule1F1B)(stage, n_microbatches=a.mb, loss_fn=lm_loss, scale_grads=False)
opt = torch.optim.Adam(mod.parameters(), lr=a.lr, fused=True)  # as Piper's actors
rep = lambda t: t.repeat(a.mb, *([1] * (t.dim() - 1))).to(dev)  # noqa: E731  mb copies of the batch
px = rep((d["images"][: b * T].permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5)
ids, lab, idx = rep(d["input_ids"][:b]), rep(d["labels"][:b]), rep(d["img_index"][:b])
del d
all_losses, times = [], []
for it in range(a.warmup + a.iters):
    torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
    losses = []
    if rank == 0:
        sched.step(px, ids, idx)
    else:
        sched.step(target=lab, losses=losses)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()
    if it >= a.warmup:
        times.append(time.perf_counter() - t0)
        all_losses += [x.item() for x in losses]
    if it == a.warmup - 1:
        torch.cuda.reset_peak_memory_stats()
peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=dev)
peaks = [torch.zeros_like(peak) for _ in range(2)]
dist.all_gather(peaks, peak)
if rank == 1:
    print(f"PIPELINING {a.schedule} mb={a.mb} losses", [round(x, 6) for x in all_losses])
    print(f"PIPELINING {a.schedule} mb={a.mb} median step {statistics.median(times) * 1e3:.1f} ms  peak GB {[round(p.item(), 1) for p in peaks]}")
    print("PASS")
dist.destroy_process_group()
