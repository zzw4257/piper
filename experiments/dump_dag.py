"""Print the lowered TrainingDAG for a model + schedule, without Ray or GPUs.

    python experiments/dump_dag.py \
      --schedule examples/base-schedules/tp2.json --dim 512 --hidden 2048 --tp 2

Runs the compiler only: trace on meta, split by annotations, apply the schedule
directives, split per PP rank, resolve the per-stream order. Then print each
rank's dispatch order with the metadata the executor actually reads. Useful for
seeing where communication got inserted when `--viz` is unavailable (Graphviz's
`dot` binary is not installed on every host).
"""
import argparse
import sys

import torch

from src.piper import piper
from src.ordering import _serial_topological_order
from src.schedule import derive_schedule_info, load_schedule_directives
from src.state import piper_metadata
from src.tasks import training_dag_task_type

from models.tp_mlp import TPMlp

_COMM_META = ("direction", "tp_tensor_idx", "a2a_tensor_idx", "peer_pp_rank", "source_uid",
              "target_uid", "bwd_uid", "compute_uid")


def _fmt_tag(tag: dict) -> str:
    return "{" + ",".join(f"{k}={v}" for k, v in sorted(tag.items())) + "}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    directives = load_schedule_directives(args.schedule)
    info = derive_schedule_info(directives, args.schedule)
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = info
    piper_metadata.visualize_dag = False
    print(f"schedule: {info}")

    with torch.device("meta"):
        model = TPMlp(args.dim, args.hidden, args.tp).to(torch.float32)
    x = torch.empty(args.batch_size, args.dim, device="meta")

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=piper, fullgraph=True)
    compiled(x)

    dags = piper_metadata.per_pp_training_dags
    print(f"\n{len(dags)} per-PP-rank DAG(s)")
    comm_total = 0
    for rank, dag in enumerate(dags):
        devices = sorted({tuple(n.device) for n in dag.nodes.values() if n.device})
        print(f"\n=== rank {rank}  devices={devices}  "
              f"{len(dag.nodes)} nodes  {len(dag.edges)} edges ===")
        for i, uid in enumerate(_serial_topological_order(dag)):
            n = dag.nodes[uid]
            if n.node_kind.endswith("_COMM"):
                comm_total += 1
            extra = " ".join(
                f"{k}={n.node_meta[k]}" for k in _COMM_META if k in n.node_meta
            )
            print(f"  {i:3d} {training_dag_task_type(n).value:<22s} "
                  f"{n.node_kind:<20s} {_fmt_tag(n.tag):<26s} "
                  f"stream={n.stream:<14s} {uid}"
                  + (f"  [{extra}]" if extra else ""))
    print(f"\ntotal comm nodes: {comm_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
