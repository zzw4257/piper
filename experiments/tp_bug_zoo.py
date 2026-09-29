"""F99: TP regions written wrong. Does the hand-written directive notice? Does the derived one? (CPU only)

    PYTHONPATH=examples:. python experiments/tp_bug_zoo.py

Each case is one TP region between two replicated layers, written in TP-local shapes as Piper
models are (tp=2). For each:
  - shard_tensor: lowers? (its rule: all-reduce whatever leaves the region, forward and backward)
  - what that rule computes, simulated in one process: the sum over ranks of the region run on
    each rank's shard, against the unsharded model (max relative error of the output);
  - layout (placements derived by DTensor): lowers, or refuses and says why.
"""
import contextlib, io, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path[:0] = ["examples", "experiments", "."]
import probe_route as p  # noqa: E402
from src.piper import annotate  # noqa: E402

D, H, TP = 32, 128, 2
CASES = {
    "correct: up (col) -> gelu -> down (row)": "ok",
    "row-parallel bias inside the region": "bias",
    "LayerNorm after the row-parallel matmul": "norm",
    "activation after the row-parallel matmul": "act",
    "residual add inside the region": "residual",
    "region ends after the column-parallel matmul": "colonly",
}


class Block(nn.Module):
    def __init__(self, kind, hidden):
        super().__init__()
        self.kind = kind
        self.pre = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, hidden, bias=False)
        self.down = nn.Linear(hidden, D, bias=(kind == "bias"))
        self.norm = nn.LayerNorm(D)
        self.post = nn.Linear(hidden if kind == "colonly" else D, D, bias=False)

    def region(self, h):
        if self.kind == "colonly":
            return F.gelu(self.up(h))
        y = self.down(F.gelu(self.up(h)))
        if self.kind == "norm":
            y = self.norm(y)
        if self.kind == "act":
            y = F.gelu(y)
        if self.kind == "residual":
            y = h + y
        return y

    def forward(self, x):
        with annotate("PP"):
            h = self.pre(x)
            with annotate("TP"):
                y = self.region(h)
            return self.post(y)


def lower(kind, directive):
    """p.lower, keeping the whole refusal message (p.lower keeps its first line only)."""
    import json, tempfile
    from src.piper import _reset_annotation_state, piper, piper_metadata
    from src.schedule import derive_schedule_info, load_schedule_directives
    sched = [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1]}, directive,
             {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sched, f)
    ds = load_schedule_directives(f.name)
    piper_metadata.schedule_directives = ds
    piper_metadata.schedule_info = derive_schedule_info(ds, f.name)
    piper_metadata.visualize_dag = False
    _reset_annotation_state()
    with torch.device("meta"):
        model = Block(kind, H // TP).float()
    torch._dynamo.reset()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch.compile(model, backend=piper, fullgraph=True)(torch.empty(8, D, device="meta"))
    except Exception as e:  # noqa: BLE001
        msg = [l for l in str(e).splitlines() if l.strip() and "backend='piper'" not in l]
        return "refused: " + (msg[0] if msg else str(e))[:230]
    dag = piper_metadata.per_pp_training_dags[0]
    return f"lowers ({sum(n.node_kind == 'TP_COMM' for n in dag.nodes.values())} TP all-reduces)"


def simulate(kind):
    """What shard_tensor's rule computes: sum over ranks of the region on each shard."""
    torch.manual_seed(0)
    full = Block(kind, H).double()
    x = torch.randn(8, D, dtype=torch.float64)
    ref = full(x)
    h = full.pre(x)
    parts = []
    for r in range(TP):
        loc = Block(kind, H // TP).double()
        loc.load_state_dict({k: v for k, v in full.state_dict().items() if not k.startswith(("up.", "down.weight", "post."))}, strict=False)
        n = H // TP
        loc.up.weight.data = full.up.weight.data[r * n:(r + 1) * n].clone()
        loc.down.weight.data = full.down.weight.data[:, r * n:(r + 1) * n].clone()
        parts.append(loc.region(h))
    if kind == "colonly":
        # the rule sums the column shards; the next layer expects them side by side
        return float("nan")
    got = full.post(sum(parts))
    return ((got - ref).abs().max() / ref.abs().max()).item()


def main():
    print(f"{'case':46s} | {'shard_tensor':24s} | {'rule, simulated':16s} | layout")
    for name, kind in CASES.items():
        st = lower(kind, {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"})
        ly = lower(kind, {"op": "layout", "filter": {"TP": "*"}, "devices": [0, 1], "axis": "tp",
                          "params": {"up": "colwise", "down": "rowwise"}, "stream": "tp_stream"})
        err = simulate(kind)
        print(f"{name:46s} | {st:24s} | {'wrong shapes' if err != err else f'{err:.1e} rel':16s}\n    layout: {ly}", flush=True)


if __name__ == "__main__":
    main()
