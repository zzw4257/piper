"""Does any compute node in the shipped Qwen EP lowering have BOTH a compute
data predecessor and a boundary-comm data predecessor? Decides whether the
executor's first-match base-input rule was ever ambiguous upstream."""
import collections, json, os, sys, tempfile
import torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "examples"))
from src.dag import build_training_dag
from src.directives import apply_schedule_directives
from src.fx import split_gm_by_annotations
from src.schedule import load_schedule_directives
from models.qwen3 import PiperQwen3Model, create_qwen3_config
import examples.test_qwen as tq

sched = json.load(open("examples/base-schedules/pp2_dp2_ep2.json"))
sched.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2})
fd, path = tempfile.mkstemp(suffix=".json"); os.write(fd, json.dumps(sched).encode()); os.close(fd)
directives = load_schedule_directives(path); os.unlink(path)
args = tq.parse_args([]); cfg = create_qwen3_config(args.model)
got = {}
def backend(gm, _inputs):
    got["segs"] = split_gm_by_annotations(gm)[1]
    return gm.forward
with torch.device("meta"):
    model = PiperQwen3Model(cfg, 2).to(torch.bfloat16)
    tokens = torch.zeros(args.batch_size, args.seq_len, dtype=torch.long)
torch._dynamo.reset()
torch.compile(model, backend=backend, fullgraph=True)(tokens)
dag = build_training_dag(got["segs"])
apply_schedule_directives(dag, directives)   # global DAG, no per-PP split (as compare_ep_lowering.py)
BOUNDARY = {"A2A_COMM", "TP_COMM", "RING_COMM"}
both = 0; total = 0; hist = collections.Counter()
for uid, n in dag.nodes.items():
    if n.node_kind != "COMPUTE" or n.compute_subkind != "FWD": continue
    total += 1
    preds = [dag.nodes[e.src_uid] for e in dag.edges if e.dst_uid == uid and e.dep_kind == "data"]
    c = sum(1 for p in preds if p.node_kind == "COMPUTE" and p.compute_subkind == "FWD")
    b = sum(1 for p in preds if p.node_kind in BOUNDARY)
    hist[(c, b)] += 1
    if c and b:
        both += 1; print(f"  {uid} tag={dict(n.tag)} compute_preds={c} boundary_preds={b}")
print(f"FWD compute nodes: {total}; with both a compute and a boundary-comm data predecessor: {both}")
print("histogram (compute_preds, boundary_preds) -> count:", dict(sorted(hist.items())))
