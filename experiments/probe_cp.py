"""Can Piper's annotate/directive layer express context parallelism?

Ring attention's shape: each rank holds a shard of K/V, and for cp_degree steps it
computes partial attention against the block it currently holds while passing that
block to the next rank. The optimization that makes it worth doing is that the
exchange overlaps the compute of the *same* step.

TP fitted Piper because its collectives sit on region boundaries (log F6). This
asks whether CP does too, by tracing a ring and looking at where the segment
boundaries actually land.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate

D, S_LOCAL, CP = 64, 32, 2


class RingBlock(nn.Module):
    """One ring step, annotated. Exchange is a pure permutation between steps."""
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D, D, bias=False)
        self.wo = nn.Linear(D, D, bias=False)

    def forward(self, x, k, v):
        acc = None
        for _ in range(CP):
            with annotate("CP"):
                q = self.wq(x)
                a = torch.softmax(q @ k.transpose(-1, -2) / (D ** 0.5), dim=-1)
                p = a @ v
                acc = p if acc is None else acc + p
            # Where a ring exchange would go: rank i sends (k, v) to i+1.
            # Written as a no-op permutation so tracing sees the data dependency
            # without any compute node outside an annotate scope.
            k, v = v, k
        with annotate("CP"):
            return self.wo(acc)


def main():
    captured = {}

    def backend(gm, example_inputs):
        captured["gm"] = gm
        _, captured["segs"] = split_gm_by_annotations(gm)
        return gm.forward

    with torch.device("meta"):
        m = RingBlock().to(torch.float32)
    x = torch.empty(2, S_LOCAL, D, device="meta")
    k = torch.empty(2, S_LOCAL, D, device="meta")
    v = torch.empty(2, S_LOCAL, D, device="meta")

    _reset_annotation_state()
    torch._dynamo.reset()
    torch.compile(m, backend=backend, fullgraph=True)(x, k, v)

    segs = captured["segs"]
    print(f"{len(segs)} segments (cp_degree={CP}, so {CP} ring steps + 1 output)\n")
    for s in segs:
        b = s.a2a_boundary_after
        print(f"  seg{s.segment_id} tag={s.tag} inputs={len(s.input_idxs)} "
              f"params={len(s.param_idxs)}")
        print(f"       boundary_after={'tensor_idx=' + str(b['tensor_idx']) if b else None}")

    print("\nwhat a directive could attach a collective to:")
    print("  - a boundary carries exactly ONE tensor index "
          "(_select_boundary_tensor_idx picks one)")
    print("  - ring attention must move TWO tensors (k and v) per step")
    print("  - and must move them DURING the step's compute, not between steps")
    n_cross = []
    for s in segs[:-1]:
        outs = [n.name for n in s.gm.graph.nodes if n.op == "output"]
        print(f"  seg{s.segment_id} output node args: {outs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
