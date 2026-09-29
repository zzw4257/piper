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
    of a runtime input (``"0"`` is the first; ``"*"`` any other) to ``"replicate"`` or
    ``"shard(d)"``; unnamed inputs are replicated. ``param_grads`` in the result: a
    Partial one needs the gradient sync data parallelism writes by hand (log F85). A sharded input is built from a local tensor of
    the traced shape, since the region was traced on one rank's chunk (log F76).
    Returns ``{"outputs": [...], "input_grads": {graphargs_idx: placement},
    "input_placements": {graphargs_idx: placement}}``.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    from torch.utils._debug_mode import DebugMode
    from torch.utils._pytree import tree_leaves

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    patterns = {k: re.compile(rf"(^|_)modules_{re.escape(k)}_parameters_") for k in params}
    input_grads: dict[int, Any] = {}
    input_placements: dict[int, Any] = {}
    param_grads: dict[int, Any] = {}
    col: set[str] = set()
    row: set[str] = set()
    with _fake_mesh(world_size) as mesh:
        args = []
        for i, (g, name) in enumerate(zip(graphargs, names)):
            if not isinstance(g, torch.Tensor):
                args.append(g)
                continue
            if i in param_idxs:
                spec = next((params[k] for k, p in patterns.items() if p.search(name)), "replicate")
                pl = _placement(spec)
                if g.dim() == 1 and spec in ("colwise", "rowwise"):
                    # a bias follows its layer's output (Megatron): split with a column-parallel
                    # layer, whole with a row-parallel one (log F99)
                    pl = Shard(0) if spec == "colwise" else Replicate()
                # from_local: the traced shape is one rank's shard, as Piper models are written
                t = DTensor.from_local(torch.randn(tuple(g.shape)), mesh, [pl], run_check=False)
                if spec in ("colwise", "rowwise"):
                    (col if spec == "colwise" else row).add(name)
            else:
                pos = input_idxs.index(i) if i in input_idxs else -1
                spec_in = inputs or {}
                pl = _placement(spec_in.get(str(pos), spec_in.get("*", "replicate")))
                input_placements[i] = pl
                t = DTensor.from_local(torch.randn(tuple(g.shape)), mesh, [pl], run_check=False)
            # an input needs a gradient only if it did when traced (rotary tables do not: log F99)
            needs = i in param_idxs or (i in input_idxs and bool(getattr(g, "requires_grad", True)))
            args.append(t.requires_grad_(needs))
            if i in param_idxs:
                # A replicated weight read by split data gets a Partial gradient; record it
                # before accumulation, which would all-reduce it: that all-reduce is the
                # gradient sync being derived (data parallelism), not one inside the region.
                def _phook(grad, i=i):
                    param_grads[i] = grad.placements[0]
                    if not grad.placements[0].is_partial():
                        return grad
                    return DTensor.from_local(grad.to_local(), mesh, [Replicate()], run_check=False)
                t.register_hook(_phook)
            if i in input_idxs and needs:
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
            outs = [o for o in tree_leaves(_between_linears(gm, mesh, col, row).run(*args)) if isinstance(o, torch.Tensor)]
        if any(not isinstance(o, DTensor) for o in outs):
            raise ValueError(
                "the region's output is a local shard (it ends between a column- and a row-parallel "
                "layer), not a partial sum: an all-reduce on it would add different slices together. "
                "End the region after a row-parallel layer (log F99)")
        # A split output feeds a split consumer: take its local part. Anything else is
        # made whole, as a replicated consumer would see it.
        loss = sum(o.to_local().sum() if o.placements[0].is_shard() else o.redistribute(mesh, [Replicate()]).to_local().sum()
                   for o in outs)
        with DebugMode() as bwd:
            loss.backward()
        inside = []
        for pass_, mode in (("forward", fwd), ("backward", bwd)):
            op = "?"
            for line in mode.debug_string().splitlines():
                st = line.strip()
                if st.startswith("aten::") and "dt:" in st:
                    op = st.split("(")[0]
                m = re.match(r"redistribute_input\((\d+), (.*)\)$", st)
                if m:
                    inside.append(f"{pass_} {op} input {m.group(1)}: {m.group(2)}")
        if inside:
            raise ValueError(
                "these placements need a redistribution inside the region, where Piper cannot put one: "
                + "; ".join(dict.fromkeys(inside))
                + ". Move that operator out of the region, so any collective sits on its boundary, "
                "or declare the parameter the way DTensor re-split it"
            )
        return {
            "outputs": [o.placements[0] for o in outs],
            "input_grads": input_grads,
            "input_placements": input_placements,
            "param_grads": param_grads,
        }


def _between_linears(gm, mesh, col: set, row: set):
    """Run a region as a TP region runs: DTensor up to the column-parallel layers and from the
    row-parallel ones, plain local tensors in between (log F99).

    Piper regions are written in TP-local shapes, as torchtitan's styles make attention run on local
    tensors (``ColwiseParallel(use_local_output=True)``): reshapes by the local head count are
    right on local tensors and meaningless on a global DTensor. So a column-parallel layer's output
    is taken local, and a row-parallel layer's activation re-enters as ``Shard(-1)``. What lies
    before the column layers and after the row layers still goes through DTensor, and that is where
    a misplaced bias, norm, activation or residual shows up as a redistribution inside the region.
    """
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
    from torch.utils._pytree import tree_map

    def plain(x):
        return isinstance(x, torch.Tensor) and not isinstance(x, DTensor)

    class _Run(torch.fx.Interpreter):
        def run_node(self, n):
            if n.op not in ("call_function", "call_method", "call_module"):
                return super().run_node(n)
            names = {a.name for a in n.all_input_nodes}
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            if names & row:
                split = lambda x: DTensor.from_local(x, mesh, [Shard(x.dim() - 1)], run_check=False) if plain(x) else x  # noqa: E731
                args, kwargs = tree_map(split, args), tree_map(split, kwargs)
            else:
                leaves = [x for x in list(args) + list(kwargs.values()) if isinstance(x, torch.Tensor)]
                if any(plain(x) for x in leaves) and any(isinstance(x, DTensor) for x in leaves):
                    # a replicated value read in the local zone (rotary tables, a mask): its gradient
                    # from there is one rank's part of the sum
                    local = lambda x: x.to_local(grad_placements=[Partial()]) if isinstance(x, DTensor) and all(  # noqa: E731
                        isinstance(q, Replicate) for q in x.placements) else x
                    args, kwargs = tree_map(local, args), tree_map(local, kwargs)
            out = getattr(self, n.op)(n.target, args, kwargs)
            if names & col:
                out = tree_map(lambda x: x.to_local() if isinstance(x, DTensor) else x, out)
            return out

    return _Run(gm)


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


def derive_split_outputs(
    gm: torch.fx.GraphModule,
    graphargs: list[Any],
    input_idxs: list[int],
    seq_dim: int,
) -> list[str | None]:
    """How each output of a region is split when its inputs are split along ``seq_dim``
    (log F85), found the way F70 finds gathered inputs: perturb position 0 of every
    split input in float64 and see where each output moves.

    ``"shard(d)"``: only position 0 along output dimension d moved (the split follows the
    rows, possibly to another dimension, as a head split moves it). ``None``: the output
    moved elsewhere too, so the region reads other positions whole. An output that does
    not move at all (a fresh buffer shaped like a split input) takes the dimension whose
    size is the local sequence length; if none, ``"replicate"``.
    """
    gm = _on_cpu(gm)
    torch.manual_seed(0)
    args, local_len = [], None
    for i, g in enumerate(graphargs):
        if not isinstance(g, torch.Tensor):
            args.append(g)
            continue
        if g.is_floating_point():
            t = torch.randn(tuple(g.shape), dtype=torch.float64)
        else:
            t = torch.zeros(tuple(g.shape), dtype=g.dtype)
        if i in input_idxs and t.dim() > seq_dim:
            local_len = t.shape[seq_dim]
        args.append(t)
    with torch.no_grad():
        base = [o for o in _leaves(gm(*args)) if isinstance(o, torch.Tensor)]
        moved = list(args)
        for i in input_idxs:
            t = moved[i]
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dim() > seq_dim:
                t = t.clone()
                t.select(seq_dim, 0).add_(1.0)
                moved[i] = t
        pert = [o for o in _leaves(gm(*moved)) if isinstance(o, torch.Tensor)]
    out = []
    for a, b in zip(base, pert):
        if not a.is_floating_point():
            out.append("replicate")
            continue
        diff = (a - b).abs() > 1e-9
        if not diff.any():
            dims = [d for d in range(a.dim()) if a.shape[d] == local_len]
            out.append(f"shard({dims[0]})" if dims else "replicate")
            continue
        found = None
        for d in range(a.dim()):
            others = tuple(k for k in range(a.dim()) if k != d)
            rows = diff.any(dim=others) if others else diff
            if rows[0] and not rows[1:].any():
                found = d
                break
        out.append(f"shard({found})" if found is not None else None)
    return out


def _leaves(x):
    from torch.utils._pytree import tree_leaves
    return tree_leaves(x)


def _on_cpu(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """A copy of ``gm`` whose baked ``device='meta'`` literals (from ``torch.full(...,
    device=x.device)`` at trace time) read ``cpu``, so it can run on real tensors; the
    actor does the same for its own device (``_relocate_meta_devices``)."""
    import copy
    gm = copy.deepcopy(gm)
    meta, cpu = torch.device("meta"), torch.device("cpu")
    fix = lambda v: cpu if (isinstance(v, torch.device) and v == meta) or v == "meta" else v  # noqa: E731
    for node in gm.graph.nodes:
        node.args = tuple(fix(a) for a in node.args)
        node.kwargs = {k: fix(v) for k, v in node.kwargs.items()}
    gm.recompile()
    return gm
