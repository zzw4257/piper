"""Train the CP-branch model through Piper and record losses, step times, checksums.

    python examples/test_harness.py --test-file examples/test_branches_cp.py \
      --base-schedule examples/base-schedules/brcp_routed_cp2.json --schedule custom

Every rank draws the same full tensors from one seed; a CP rank keeps its chunk of
the sequence. With no ring in the schedule the text branch runs one step over the
whole sequence: the dense reference (log F76).
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
from src.piper import piper_exec_dag, piper_flush_losses, piper_param_checksums
from src.schedule import derive_schedule_info, load_schedule_directives
from src.state import piper_metadata

from models.branches_cp import BranchesCP, global_weights

logger = logging.getLogger(__name__)


def main(args, pg):
    loss_fn = lambda output, labels: (output.float() - labels.float()).pow(2).mean()  # noqa: E731
    ds = load_schedule_directives(args.schedule_directives_file)
    cp = (derive_schedule_info(ds, args.schedule_directives_file)["dp_degree"]
          if any(d.get("op") == "ring_exchange" for d in ds) else 1)
    rank = int(os.environ.get("PIPER_DP_RANK", 0)) if cp > 1 else 0
    g = torch.Generator().manual_seed(args.seed)
    x_img = torch.randn(args.batch_size, args.dim, generator=g)
    x_txt, k, v = (torch.randn(args.batch_size, args.seq, args.dim, generator=g) for _ in range(3))
    y = torch.randn(args.batch_size, args.dim, generator=g)
    s = args.seq // cp
    x_txt, k, v = (t[:, rank * s:(rank + 1) * s].contiguous() for t in (x_txt, k, v))

    piper_setup(
        BranchesCP,
        model_args=(args.dim, args.heads, cp, args.seq, args.depth_img, args.depth_dec, args.ep),
        optim_fn=functools.partial(torch.optim.Adam, lr=args.lr),
        example_inputs=[x_img, x_txt, k, v],
        example_outputs=y,
        model_dtype=torch.float32,
        pg=pg,
        temp_dir=args.temp_dir,
        param_overrides=global_weights(args.dim, args.depth_img, args.depth_dec, args.seed, args.ep),
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
               "schedule": os.path.basename(args.schedule_directives_file),
               "param_checksums": piper_param_checksums()}
    path = os.path.join(getattr(piper_metadata, "artifact_dir", "out"),
                        f"branches_metrics_dp{metrics['dp_rank']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("losses=%s -> %s", [round(x, 6) for x in metrics["losses"]], path)
    return metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--depth-img", type=int, default=4)
    ap.add_argument("--depth-dec", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ep", action="store_true", help="expert layer in an EP region in the image branch")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    ap.add_argument("--schedule-directives-file", type=str,
                    default="examples/base-schedules/brcp_routed_cp2.json")
    return ap.parse_args(argv)
