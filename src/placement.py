"""Derive a region's boundary collectives from how its parameters are placed.

A directive such as ``shard_tensor`` states *where* a collective goes by a fixed
edge rule. Here the user states only how each parameter is split, and the
collective follows from the placements: the segment runs once under DTensor on
a fake mesh, and each output (and each input gradient) comes out Replicate,
Shard or Partial. Leaving the region, every tensor must be Replicate again, so a
Partial result needs an all-reduce there (log F68).

Only boundary collectives are derived. A placement that makes DTensor
redistribute anything *inside* the segment is refused: a collective there, or a
parameter silently re-split to fit (a replicated weight chunked to Shard), would
not happen when Piper runs the segment on the shards the user declared.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist


def _placement(spec: str):
    from torch.distributed.tensor import Replicate, Shard

    # nn.Linear weights are [out, in]: column-parallel splits outputs, row-parallel splits inputs.
    table = {"colwise": Shard(0), "rowwise": Shard(1), "replicate": Replicate()}
    m = re.fullmatch(r"shard\((\d+)\)", spec)
    if m:
        return Shard(int(m.group(1)))
    if spec not in table:
        raise ValueError(f"unknown placement {spec!r}; expected one of {sorted(table)} or 'shard(d)'")
    return table[spec]


@contextmanager
def _fake_mesh(world_size: int):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    if dist.is_initialized():
        # ponytail: derivation owns a throwaway fake process group; it runs at compile
        # time on the driver, where none exists. Give it a private group if that changes.
        raise RuntimeError("placement derivation needs a process without a default process group")
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world_size)
    try:
        yield init_device_mesh("cpu", (world_size,))
    finally:
        dist.destroy_process_group()


def derive_boundary_placements(
    gm: torch.fx.GraphModule,
    graphargs: list[Any],
    input_idxs: list[int],
    param_idxs: list[int],
    params: dict[str, str],
    world_size: int,
    inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Placements of a segment's outputs and runtime-input gradients.

    ``params`` maps a module name (``"up"``) to ``"colwise"``, ``"rowwise"`` or
    ``"replicate"``; unnamed parameters are replicated. ``inputs`` maps the position
    of a runtime input (``"0"`` is the first) to ``"replicate"`` or ``"shard(d)"``;
    unnamed inputs are replicated. A sharded input is built from a local tensor of
    the traced shape, since the region was traced on one rank's chunk (log F76).
    Returns ``{"outputs": [...], "input_grads": {graphargs_idx: placement},
    "input_placements": {graphargs_idx: placement}}``.
    """
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
    from torch.utils._debug_mode import DebugMode
    from torch.utils._pytree import tree_leaves

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    patterns = {k: re.compile(rf"(^|_)modules_{re.escape(k)}_parameters_") for k in params}
    input_grads: dict[int, Any] = {}
    input_placements: dict[int, Any] = {}
    with _fake_mesh(world_size) as mesh:
        args = []
        for i, (g, name) in enumerate(zip(graphargs, names)):
            if not isinstance(g, torch.Tensor):
                args.append(g)
                continue
            if i in param_idxs:
                spec = next((params[k] for k, p in patterns.items() if p.search(name)), "replicate")
                t = distribute_tensor(torch.randn(tuple(g.shape)), mesh, [_placement(spec)])
            else:
                pos = input_idxs.index(i) if i in input_idxs else -1
                pl = _placement((inputs or {}).get(str(pos), "replicate"))
                input_placements[i] = pl
                t = DTensor.from_local(torch.randn(tuple(g.shape)), mesh, [pl], run_check=False)
            args.append(t.requires_grad_(i in param_idxs or i in input_idxs))
            if i in input_idxs:
                # Record the input gradient as computed. Accumulating a Partial gradient into
                # a Replicate leaf would all-reduce it, and that all-reduce is the boundary
                # collective being derived, not one inside the region; stop it here.
                def _hook(grad, i=i):
                    input_grads[i] = grad.placements[0]
                    if not grad.placements[0].is_partial():
                        return grad
                    return DTensor.from_local(grad.to_local(), mesh, [Replicate()], run_check=False)
                t.register_hook(_hook)
        with DebugMode() as fwd:
            outs = [o for o in tree_leaves(gm(*args)) if isinstance(o, torch.Tensor)]
        loss = sum(o.redistribute(mesh, [Replicate()]).to_local().sum() for o in outs)
        with DebugMode() as bwd:
            loss.backward()
        inside = [line.strip() for mode in (fwd, bwd) for line in mode.debug_string().splitlines()
                  if re.match(r"\s*redistribute_input\(\d+,", line)]
        if inside:
            raise ValueError(
                f"these placements need redistribution inside the region {inside}; "
                f"split the region so every collective sits on its boundary, "
                f"or declare the parameter the way DTensor re-split it"
            )
        return {
            "outputs": [o.placements[0] for o in outs],
            "input_grads": input_grads,
            "input_placements": input_placements,
        }


def derive_gathered_inputs(
    gm: torch.fx.GraphModule,
    graphargs: list[Any],
    input_idxs: list[int],
    seq_dim: int,
    world_size: int,
) -> list[str]:
    """Placeholder names of the runtime inputs a region needs whole when its inputs and
    outputs are split along ``seq_dim``: the payload of the all-gather that split implies.

    Context parallelism keeps every query-side value split along the sequence. An input
    must then be gathered exactly when an output position depends on *other* positions
    of that input: perturb one position of the input and see whether any other output
    position moves. Row-local inputs (the query chunk, the running softmax state) stay
    split; K and V, read by every query position, are gathered. ``ring_exchange`` lowers
    that gather as n-1 neighbour hops between consecutive regions (log F70).

    This asks for the rule, not for DTensor's plan. At the CP example's real sizes
    DTensor gathers only K and moves the probabilities instead of V (all-to-all, then
    reduce-scatter), because that moves fewer bytes; the ring is the plan that gathers
    both, and choosing it is a schedule decision. ``world_size`` is unused: the answer
    does not depend on how many chunks there are.
    """
    del world_size
    from torch.utils._pytree import tree_leaves

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    gen = torch.Generator().manual_seed(0)
    base = [torch.randn(tuple(g.shape), generator=gen, dtype=torch.float64)
            if isinstance(g, torch.Tensor) else g for g in graphargs]
    gm64 = gm.to(torch.float64) if any(True for _ in gm.parameters()) else gm

    def run(args):
        with torch.no_grad():
            return [o for o in tree_leaves(gm64(*args)) if isinstance(o, torch.Tensor)]

    ref = run(base)
    gathered = []
    for i in input_idxs:
        g = base[i]
        if not isinstance(g, torch.Tensor) or g.dim() <= seq_dim or g.shape[seq_dim] < 2:
            continue
        bumped = list(base)
        bumped[i] = g.clone()
        bumped[i].select(seq_dim, 0).add_(1.0)
        for a, b in zip(run(bumped), ref):
            if a.dim() > seq_dim and a.shape[seq_dim] == g.shape[seq_dim]:
                if not torch.equal(a.narrow(seq_dim, 1, a.shape[seq_dim] - 1),
                                   b.narrow(seq_dim, 1, b.shape[seq_dim] - 1)):
                    gathered.append(i)
                    break
    return [names[i] for i in gathered]
