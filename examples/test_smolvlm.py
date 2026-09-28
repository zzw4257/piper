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

from models.smolvlm import SmolVLM, lm_loss, shard_weights

logger = logging.getLogger(__name__)


def main(args, pg):
    data = torch.load(os.path.join(args.data, "data.pt"))
    weights = torch.load(os.path.join(args.data, "weights.pt"))
    if args.perturb_ulp:
        # one float32 ulp on one weight: the smallest perturbation there is (log F93)
        w = weights["lm_head.weight"]
        w[0, 0] = torch.nextafter(w[0, 0], torch.tensor(float("inf")))
    b = args.batch_size
    from src.schedule import derive_schedule_info, load_schedule_directives, mesh_coords
    sched = derive_schedule_info(load_schedule_directives(args.schedule_directives_file),
                                 args.schedule_directives_file)
    place = int(os.environ.get("PIPER_DP_RANK", 0))
    coords = mesh_coords(sched, place)
    if args.tp > 1:
        # every rank of the group is a TP rank, or its tp coordinate on a mesh (log F82)
        weights = shard_weights(weights, coords.get("tp", place), args.tp)

    def batch(i):
        # batch i of the stream; with --stream, step i trains on rows i*b.. of the data (log F88)
        n = len(data["input_ids"])
        rows = torch.arange(i * b, (i + 1) * b) % n
        if args.dp_split > 1:
            # replica d trains on rows d, d+n, ...: its dp coordinate, or its place without a mesh
            rows = rows[coords.get("dp", place)::args.dp_split]
        T = data.get("tiles", 1)
        tile_rows = (rows[:, None] * T + torch.arange(T)).reshape(-1)
        px = (data["images"][tile_rows].permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
        ids, lab = data["input_ids"][rows], data["labels"][rows]
        extra = [data["img_index"][rows]] if "img_index" in data else []
        return px, ids, lab, extra

    pixel_values, input_ids, labels, extra_inputs = batch(0)
    config = dict(data.get("config") or {}, tp=args.tp)

    piper_setup(
        SmolVLM,
        model_args=(data["image_offset"], args.dec_stages, args.vis_stages, config, args.vis_chunks),
        optim_fn=functools.partial(torch.optim.Adam, lr=args.lr),
        example_inputs=[pixel_values, input_ids, *extra_inputs],
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
    for i in range(args.iters):
        if args.stream and i > 0:
            px, ids, lab, ext = batch(i)
            refs = ray.put([px, ids, *ext]), ray.put(lab)
            ray.get([a.load_input.remote(refs[0]) for a in actors.values()]
                    + [a.load_labels.remote(refs[1]) for a in actors.values()])
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
    ap.add_argument("--vis-chunks", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp-split", type=int, default=1)
    ap.add_argument("--stream", action="store_true", help="a new batch every step instead of one fixed batch")
    ap.add_argument("--perturb-ulp", action="store_true", help="move lm_head.weight[0, 0] by one float32 ulp")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    ap.add_argument("--schedule-directives-file", type=str, default="examples/base-schedules/vlm_single.json")
    return ap.parse_args(argv)
