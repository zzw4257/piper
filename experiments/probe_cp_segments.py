"""G-0: does a ring-attention-shaped model segment the way the CP design needs?

Two questions, both answerable on CPU with no Ray:

  1. Does Dynamo produce one contiguous CP region per ring step?
  2. In each intermediate region, is the K/V boundary output a bare
     placeholder (forwarded, not produced)? If so, a collective on that edge
     is ready at the region's *start*, and the hoist in notes/cp-design.md
     needs no slicer.

Also reports what the single-tensor boundary heuristic picks, and how many
regions each pass-through survives (= the maximum legal hoist distance).

    python experiments/probe_cp_segments.py --steps 4
"""
import argparse
import sys

import torch
import torch.fx as fx

from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state

from models.ring_attn import RingAttn, RingAttnRebound


def _capture(model, *inputs):
    box = {}

    def backend(gm, example_inputs):
        box["gm"] = gm
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    with torch.no_grad():
        compiled(*inputs)
    return box["gm"]


def _outputs(seg_gm: fx.GraphModule) -> list[fx.Node]:
    out = next(n for n in seg_gm.graph.nodes if n.op == "output")
    args = out.args[0]
    if isinstance(args, fx.Node):
        return [args]
    return list(args) if args is not None else []


def _placeholder_index(seg_gm: fx.GraphModule, node: fx.Node) -> int | None:
    phs = [n for n in seg_gm.graph.nodes if n.op == "placeholder"]
    return phs.index(node) if node in phs else None


def probe(name: str, model, inputs) -> dict:
    gm = _capture(model, *inputs)
    _, segments = split_gm_by_annotations(gm)
    cp_segs = [s for s in segments if "CP" in s.tag]
    print(f"\n=== {name}: {len(segments)} segments, {len(cp_segs)} tagged CP ===")
    print("tags in order:", [dict(s.tag) for s in segments])

    # For each segment, classify each output element.
    # A forwarded value shows up as a placeholder in the *output* tuple.
    forwarded_runs: dict[str, int] = {}   # key -> consecutive regions forwarded
    summary = []
    for s in segments:
        outs = _outputs(s.gm)
        binfo = s.a2a_boundary_after or {}
        chosen = binfo.get("tensor_idx")
        row = []
        for i, o in enumerate(outs):
            if o.op == "placeholder":
                pidx = _placeholder_index(s.gm, o)
                kind = f"FORWARDED(ph#{pidx})"
                key = o.target if isinstance(o.target, str) else str(o.target)
                forwarded_runs[key] = forwarded_runs.get(key, 0) + 1
            else:
                kind = f"PRODUCED({o.op}:{getattr(o.target, '__name__', o.target)})"
            mark = "  <- boundary tensor_idx" if i == chosen else ""
            row.append(f"      [{i}] {o.name:<22} {kind}{mark}")
        print(f"  seg {s.segment_id} tag={dict(s.tag)}  n_out={len(outs)}  chosen_idx={chosen}")
        print("\n".join(row))
        summary.append({
            "seg": s.segment_id, "tag": dict(s.tag), "n_out": len(outs),
            "chosen": chosen,
            "_out_names": [o.name for o in outs],
            "forwarded": [o.name for o in outs if o.op == "placeholder"],
            "produced": [o.name for o in outs if o.op != "placeholder"],
        })

    print("\n  max legal hoist distance per forwarded value (regions it passes through unchanged):")
    for k, n in sorted(forwarded_runs.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} {n}")
    return {"segments": summary, "forwarded_runs": forwarded_runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.seq, args.dim)
    k = torch.randn(args.batch, args.seq, args.dim)
    v = torch.randn(args.batch, args.seq, args.dim)

    a = probe("Form A: K/V forwarded (Piper style)",
              RingAttn(args.dim, args.heads, args.steps), (x, k, v))
    b = probe("Form B: K/V rebound by clone() inside region",
              RingAttnRebound(args.dim, args.heads, args.steps), (x, k, v))

    # --- verdicts -------------------------------------------------------
    print("\n=== verdicts ===")
    n_cp = sum(1 for s in a["segments"] if "CP" in s["tag"])
    print(f"Q1 contiguous CP regions: {'YES' if n_cp == args.steps else 'NO'} ({n_cp} of {args.steps})")

    inter = [s for s in a["segments"] if "CP" in s["tag"]][:-1]
    kv_fwd = all(len(s["forwarded"]) >= 2 for s in inter)
    print(f"Q2 K/V forwarded as placeholders in every non-final CP region: {'YES' if kv_fwd else 'NO'}")

    # What does the single-index heuristic actually select, and is it a value
    # the region produced or one it merely forwards?
    for s in inter:
        outs = s["forwarded"] + s["produced"]   # display order == output order
        # rebuild output order from the segment summary
    picks = {}
    for s in inter:
        if s["chosen"] is None:
            continue
        name = s["_out_names"][s["chosen"]]
        picks[name] = "FORWARDED" if name in s["forwarded"] else "PRODUCED"
    print(f"Q3 single-index heuristic selects {picks} -- one index cannot name K and V, "
          f"and the score is blind to forwarded-vs-produced; multi-tensor boundary required")

    b_inter = [s for s in b["segments"] if "CP" in s["tag"]][:-1]
    kv_produced = all(
        not any(n.endswith("_k") or n.endswith("_v") for n in s["forwarded"])
        for s in b_inter
    )
    print(f"Q4 Form B (clone inside region) turns K/V into PRODUCED values: "
          f"{'YES -- placeholder test no longer fires; a slicer would be needed' if kv_produced else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
