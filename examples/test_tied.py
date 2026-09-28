"""Train the tied-embedding LM through Piper; losses and checksums per placement (log F83).

    python examples/test_harness.py --test-file examples/test_tied.py \
      --base-schedule examples/base-schedules/tied_3gpu.json --schedule custom --stages 3
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
from src.state import piper_metadata

from models.tied_lm import TiedLM, global_weights, lm_loss

logger = logging.getLogger(__name__)


def main(args, pg):
    g = torch.Generator().manual_seed(args.seed)
    tokens = torch.randint(0, args.vocab, (args.batch_size, args.seq), generator=g)
    piper_setup(
        TiedLM, model_args=(args.vocab, args.dim, args.layers, args.stages),
        optim_fn=functools.partial(torch.optim.Adam, lr=args.lr),
        example_inputs=[tokens], example_outputs=tokens, model_dtype=torch.float32, pg=pg,
        temp_dir=args.temp_dir, param_overrides=global_weights(args.vocab, args.dim, args.layers, args.seed),
        visualize_dag=False, use_inductor=False, pp_outer=False, schedule_directives_file=args.schedule_directives_file,
    )
    actors = piper_metadata.actors
    iter_times, losses = [], []
    for _ in range(args.iters):
        ray.get([a.drain.remote() for a in actors.values()])
        start = time.perf_counter()
        losses.extend(piper_exec_dag(lm_loss) or [])
        ray.get([a.drain.remote() for a in actors.values()])
        iter_times.append(time.perf_counter() - start)
    losses.extend(piper_flush_losses())
    ck = piper_param_checksums()
    metrics = {"losses": [float(x) for x in losses], "iter_times": iter_times,
               "dp_rank": int(os.environ.get("PIPER_DP_RANK", 0)), "param_checksums": ck}
    path = os.path.join(getattr(piper_metadata, "artifact_dir", "out"), f"branches_metrics_dp{metrics['dp_rank']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("losses=%s", [round(x, 6) for x in metrics["losses"]])
    return metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    ap.add_argument("--schedule-directives-file", type=str, default="examples/base-schedules/tied_3gpu.json")
    return ap.parse_args(argv)
