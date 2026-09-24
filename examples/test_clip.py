"""Fine-tune the released CLIP ViT-B/32 through Piper on real image-caption pairs.

    python experiments/prepare_clip.py --out /var/tmp/ziweizho-clip     # once
    python examples/test_harness.py --test-file examples/test_clip.py \
      --base-schedule examples/base-schedules/clip_routed.json --schedule custom

The image tower, the text tower and the similarity head are three PP regions. Every
schedule starts from the same pretrained weights and the same batch, so losses from
different placements must agree (log F72).
"""
import argparse
import functools
import json
import logging
import os
import time

import ray
import torch
import torch.nn.functional as F

from src.compile import piper_setup
from src.piper import piper_exec_dag, piper_flush_losses
from src.state import piper_metadata

from models.clip import CLIP

logger = logging.getLogger(__name__)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_loss(logits_per_image, labels):
    """Symmetric contrastive loss over one microbatch; pair i is the diagonal."""
    target = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    logits = logits_per_image.float()
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2


def main(args, pg):
    data = torch.load(os.path.join(args.data, "data.pt"))
    weights = torch.load(os.path.join(args.data, "weights.pt"))
    images = data["images"][: args.batch_size].permute(0, 3, 1, 2).float() / 255.0
    images = F.interpolate(images, size=224, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    pixel_values = (images - MEAN) / STD
    input_ids = data["input_ids"][: args.batch_size]
    labels = torch.arange(args.batch_size)

    piper_setup(
        CLIP,
        model_args=(),
        optim_fn=functools.partial(torch.optim.Adam, lr=args.lr),
        example_inputs=[pixel_values, input_ids],
        example_outputs=labels,
        model_dtype=torch.float32,
        pg=pg,
        temp_dir=args.temp_dir,
        param_overrides=weights,
        visualize_dag=False,
        use_inductor=False,
        pp_outer=False,
        schedule_directives_file=args.schedule_directives_file,
    )
    actors = piper_metadata.actors
    for _ in range(args.warmup):
        piper_exec_dag(clip_loss)
    piper_flush_losses()

    iter_times, losses = [], []
    for _ in range(args.iters):
        ray.get([a.drain.remote() for a in actors.values()])
        start = time.perf_counter()
        losses.extend(piper_exec_dag(clip_loss) or [])
        ray.get([a.drain.remote() for a in actors.values()])
        iter_times.append(time.perf_counter() - start)
    losses.extend(piper_flush_losses())

    metrics = {"losses": [float(x) for x in losses], "iter_times": iter_times,
               "dp_rank": int(os.environ.get("PIPER_DP_RANK", 0)),
               "schedule": os.path.basename(args.schedule_directives_file)}
    path = os.path.join(getattr(piper_metadata, "artifact_dir", "out"),
                        f"branches_metrics_dp{metrics['dp_rank']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("losses=%s -> %s", [round(x, 6) for x in metrics["losses"]], path)
    return metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/var/tmp/ziweizho-clip")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    ap.add_argument("--schedule-directives-file", type=str, default="examples/base-schedules/clip_routed.json")
    return ap.parse_args(argv)
