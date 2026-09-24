"""Train the two-branch model through Piper and record losses and step times.

Run through the harness, as the other examples:

    python examples/test_harness.py --test-file examples/test_branches.py \
      --base-schedule examples/base-schedules/br_routed.json --schedule custom

Every placement starts from the same global weights and data, so losses from
different schedules must agree (log F71).
"""
import argparse
import json
import logging
import os
import time

import ray
import torch

from src.compile import piper_setup
from src.piper import piper_exec_dag, piper_flush_losses
from src.state import piper_metadata

from models.branches import Branches, global_weights

logger = logging.getLogger(__name__)


def main(args, pg):
    loss_fn = lambda output, labels: (output.float() - labels.float()).pow(2).mean()  # noqa: E731
    overrides = global_weights(args.dim, args.depth_img, args.depth_txt, args.depth_dec, args.seed)
    torch.manual_seed(args.seed)
    x_img = torch.randn(args.batch_size, args.dim)
    x_txt = torch.randn(args.batch_size, args.dim)
    y = torch.randn(args.batch_size, args.dim)

    piper_setup(
        Branches,
        model_args=(args.dim, args.depth_img, args.depth_txt, args.depth_dec),
        optim_fn=torch.optim.Adam,
        example_inputs=[x_img, x_txt],
        example_outputs=y,
        model_dtype=torch.float32,
        pg=pg,
        temp_dir=args.temp_dir,
        param_overrides=overrides,
        visualize_dag=False,
        use_inductor=False,
        pp_outer=False,
        schedule_directives_file=args.schedule_directives_file,
    )
    actors = piper_metadata.actors
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
    piper_flush_losses()

    iter_times, losses = [], []
    for _ in range(args.iters):
        ray.get([a.drain.remote() for a in actors.values()])
        start = time.perf_counter()
        losses.extend(piper_exec_dag(loss_fn) or [])
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
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--depth-img", type=int, default=8)
    ap.add_argument("--depth-txt", type=int, default=8)
    ap.add_argument("--depth-dec", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    ap.add_argument("--schedule-directives-file", type=str, default="examples/base-schedules/br_routed.json")
    return ap.parse_args(argv)
