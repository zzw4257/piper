"""Emit the lowered DAG's shape for the shipped Qwen EP example.

Uses only APIs that exist on upstream/main as well, so the same script runs in
both checkouts and the outputs can be diffed. Answers one question: did lifting
_boundary_info_for_edge out of _insert_shard_a2a_comm_nodes change what EP lowers
to?
"""
import collections, json, sys, tempfile, os
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "examples"))

from src.dag import build_training_dag
from src.directives import apply_schedule_directives
from src.fx import split_gm_by_annotations
from src.schedule import load_schedule_directives

from models.qwen3 import PiperQwen3Model, create_qwen3_config
import examples.test_qwen as tq

sched = json.load(open("examples/base-schedules/pp2_dp2_ep2.json"))
sched.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2})
fd, path = tempfile.mkstemp(suffix=".json"); os.write(fd, json.dumps(sched).encode()); os.close(fd)
directives = load_schedule_directives(path)

args = tq.parse_args([])
cfg = create_qwen3_config(args.model)
NUM_STAGES = 2  # pp2_dp2_ep2.json places PP 0 and PP 1

got = {}
def backend(gm, _inputs):
    got["segs"] = split_gm_by_annotations(gm)[1]
    return gm.forward

with torch.device("meta"):
    model = PiperQwen3Model(cfg, NUM_STAGES).to(torch.bfloat16)
torch._dynamo.reset()
tokens = torch.zeros(2, args.seq_len, dtype=torch.long, device="meta")
torch.compile(model, backend=backend, fullgraph=True)(tokens)

dag = build_training_dag(got["segs"])
apply_schedule_directives(dag, directives)

kinds = collections.Counter(n.node_kind for n in dag.nodes.values())
a2a = sorted(
    (n.node_meta.get("direction"), n.node_meta.get("a2a_tensor_idx"),
     n.node_meta.get("target_uid") or n.node_meta.get("source_uid"))
    for n in dag.nodes.values() if n.node_kind == "A2A_COMM"
)
print("segments:", len(got["segs"]))
print("nodes:", len(dag.nodes), "edges:", len(dag.edges))
print("kinds:", dict(sorted(kinds.items())))
print("a2a nodes:")
for d_, idx, anchor in a2a:
    print(f"  direction={d_} tensor_idx={idx} anchor={anchor}")
os.unlink(path)
