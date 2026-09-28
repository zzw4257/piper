"""F92: the plan DTensor picks for sequence-split attention, and its bytes (CPU, fake 4-rank mesh).

    python experiments/dtensor_cp_plans.py OUT.json

Attention written with matmuls, q, k, v Shard(seq). Bytes sent per rank are computed from each
placement change DTensor makes (fake CPU groups replace all-to-all by all-gather, so collective
counts would mislead): S->R all-gather n*(N-1), S(i)->S(j) all-to-all n*(N-1)/N, P->R all-reduce
2n*(N-1)/N, P->S reduce-scatter n*(N-1)/N, n the local bytes. Compared with the plan Piper writes:
gather K and V once (forward), reduce-scatter dK and dV (backward).
"""
import os, re, math, json, sys, torch, torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DeviceMesh, DTensor, Shard, Replicate
from torch.utils._debug_mode import DebugMode
N = 4
B, H = int(os.environ.get('B', 1)), int(os.environ.get('H', 4))
DS = [int(x) for x in os.environ.get('DS', '16,32,64,128').split(',')]
SLS = [int(x) for x in os.environ.get('SLS', '4,8,16,32,64,128,256').split(',')]
dist.init_process_group("fake", rank=0, world_size=N, store=FakeStore())
mesh = DeviceMesh("cpu", list(range(N)))
TR = re.compile(r"redistribute_input\(t: f32\[([\d, ]*)\], trace: (.*)\)\s*$")
def cost(shape, trace):
    n = math.prod(int(x) for x in shape.split(",") if x.strip()) * 4
    total, kinds, cur = 0.0, [], None
    for step in trace.split("->"):
        step = step.strip()
        if cur is not None:
            if cur.startswith("S") and step == "R": total += n * (N - 1); kinds.append("AG"); n *= N
            elif cur.startswith("S") and step.startswith("S"): total += n * (N - 1) / N; kinds.append("A2A")
            elif cur.startswith("P") and step == "R": total += 2 * n * (N - 1) / N; kinds.append("AR")
            elif cur.startswith("P") and step.startswith("S"): total += n * (N - 1) / N; kinds.append("RS"); n //= N
            elif cur == "R" and step.startswith("S"): kinds.append("slice"); n //= N
        cur = step
    return total, kinds
def run(sl, d, force):
    q, k, v = (DTensor.from_local(torch.randn(B, H, sl, d), mesh, [Shard(2)], run_check=False).requires_grad_() for _ in range(3))
    with DebugMode() as dm:
        kk, vv = (k.redistribute(mesh, [Replicate()]), v.redistribute(mesh, [Replicate()])) if force else (k, v)
        o = ((q @ kk.transpose(-1, -2) / d ** 0.5).softmax(-1) @ vv).redistribute(mesh, [Shard(2)])
        split = len(dm.debug_string().splitlines())
        o.to_local().sum().backward()
    fwd = bwd = 0.0; kinds = {"fwd": [], "bwd": []}
    for i, line in enumerate(dm.debug_string().splitlines()):
        m = TR.search(line.strip())
        if not m: continue
        c, ks = cost(m.group(1), m.group(2))
        if i < split: fwd += c; kinds["fwd"] += ks
        else: bwd += c; kinds["bwd"] += ks
    return fwd, bwd, kinds
rows = []
for d in DS:
    for sl in SLS:
        f, b, ks = run(sl, d, False); gf, gb, gks = run(sl, d, True)
        rows.append(dict(B=B, H=H, d=d, sl=sl, fwd=f, bwd=b, plan_fwd=" ".join(ks["fwd"]), plan_bwd=" ".join(ks["bwd"]),
                         gather_fwd=gf, gather_bwd=gb, gplan_bwd=" ".join(gks["bwd"])))
        print(f"d={d:4d} sl={sl:4d} | DTensor fwd [{' '.join(ks['fwd']):12s}] {f/1024:8.1f}  bwd [{' '.join(ks['bwd']):28s}] {b/1024:8.1f} KiB"
              f" | gather fwd {gf/1024:7.1f} bwd [{' '.join(gks['bwd'])}] {gb/1024:7.1f} | ratio {(f+b)/(gf+gb):4.2f}", flush=True)
json.dump(rows, open(sys.argv[1], "w"))
