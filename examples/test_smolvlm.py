"""Fine-tune the real SmolVLM-256M on real COCO captions through Piper.

    python experiments/prepare_smolvlm.py ...     # once
    python examples/test_harness.py --test-file examples/test_smolvlm.py \
      --base-schedule examples/base-schedules/vlm_single.json --schedule custom --data DIR

Every schedule starts from the same released weights and the same batch, so losses
from different placements must agree (log F77). The decoder is split into as many
PP regions as the schedule places (``--dec-stages``).
"""
import argparse
import functools
import json
import logging
import os
import time

import ray
import torch

from src.compile import piper_setup
from src.piper import piper_exec_dag, piper_flush_losses
from src.state import piper_metadata

from models.smolvlm import SmolVLM, lm_loss

logger = logging.getLogger(__name__)


def main(args, pg):
    data = torch.load(os.path.join(args.data, "data.pt"))
    weights = torch.load(os.path.join(args.data, "weights.pt"))
    b = args.batch_size
    pixel_values = (data["images"][:b].permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
    input_ids, labels = data["input_ids"][:b], data["labels"][:b]

    piper_setup(
        SmolVLM,
        model_args=(data["image_offset"], args.dec_stages, args.vis_stages),
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
        piper_exec_dag(lm_loss)
    piper_flush_losses()

    ray.get([a.reset_peak_memory.remote() for a in actors.values()])
    iter_times, losses = [], []
    for _ in range(args.iters):
        ray.get([a.drain.remote() for a in actors.values()])
        start = time.perf_counter()
        losses.extend(piper_exec_dag(lm_loss) or [])
        ray.get([a.drain.remote() for a in actors.values()])
        iter_times.append(time.perf_counter() - start)
    losses.extend(piper_flush_losses())

    metrics = {"losses": [float(x) for x in losses], "iter_times": iter_times,
               "dp_rank": int(os.environ.get("PIPER_DP_RANK", 0)),
               "schedule": os.path.basename(args.schedule_directives_file),
               "peak_mem_gb": {int(r): m / 2**30 for r, m in ray.get(
                   [a.get_and_reset_peak_memory_stats.remote() for a in actors.values()])}}
    path = os.path.join(getattr(piper_metadata, "artifact_dir", "out"),
                        f"branches_metrics_dp{metrics['dp_rank']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("losses=%s -> %s", [round(x, 6) for x in metrics["losses"]], path)
    return metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/var/tmp/ziweizho-smolvlm-data")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dec-stages", type=int, default=1)
    ap.add_argument("--vis-stages", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    ap.add_argument("--schedule-directives-file", type=str, default="examples/base-schedules/vlm_single.json")
    return ap.parse_args(argv)
