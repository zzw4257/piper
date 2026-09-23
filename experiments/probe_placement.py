"""Probe A: does the placement on an edge predict the collective on it?

Direction A proposes an IR that carries a placement on every edge. This probe asks
whether placements written from the model's declared layouts, and nothing else, predict
where the directives put communication and which kind, in the DAGs that execute.

Independence is the point. Communication nodes are first spliced out of the lowered DAG,
leaving compute-to-compute edges. A prediction for each edge is made from the tags of its
two compute ends (TP region, CP region, pipeline stage, pass), the edge's tensor name,
and what the schedule declares (which regions replicate with ZeRO, which tensors a ring
rotates). The kind of the spliced node is never read. Predictions are then compared with
what was spliced out, in both directions: a predicted collective with nothing there, and
a node with nothing predicted.

Two lattices are scored. `spmd` has R, S and P only. `perm` adds a device permutation to
S, so that a chunk rotated one hop is S under a different assignment and the change is a
collective permute. Parameter-side nodes (gathers before compute, gradient reductions
after backward) are predicted from the ZeRO and replication flags.

Written before the first run: TP, DP and ZeRO predicted in both lattices; CP rings missed
by `spmd` and predicted by `perm`; stage crossings predicted only once placement carries a
device group; ZeRO backward gathers follow the forward's free-after flag.

    PYTHONPATH=examples:. python experiments/probe_placement.py
"""
import contextlib
import io
import sys
from collections import Counter, defaultdict

sys.path[:0] = ["experiments", "examples", "."]
import dump_dag  # noqa: E402
from src.piper import _reset_annotation_state, piper_metadata  # noqa: E402

KIND = {"TP_COMM": "all_reduce", "REDUCE_COMM": "all_reduce", "ALL_GATHER_COMM": "all_gather",
        "REDUCE_SCATTER_COMM": "reduce_scatter", "A2A_COMM": "all_to_all",
        "RING_COMM": "permute", "SEND_COMM": "send_recv", "RECV_COMM": "send_recv"}
PARAM_SIDE = {"ALL_GATHER_COMM", "REDUCE_SCATTER_COMM", "REDUCE_COMM"}

CASES = {
    "TP=2": ["--schedule", "examples/base-schedules/tp2.json", "--tp", "2"],
    "CP=2 + DP": ["--model", "ring", "--steps", "3", "--schedule", "examples/base-schedules/cp2_ring_dp.json"],
    "ZeRO-3, 3 stages": ["--schedule", "examples/base-schedules/zero3_s3_mb1.json", "--stages", "3", "--tp", "1"],
    "TP=2 x PP=2": ["--schedule", "examples/base-schedules/pp2_tp2_mb4_1f1b.json", "--stages", "2", "--tp", "2"],
}


class Union:
    """The per-rank DAGs that actually execute, joined into one graph.

    Scored after the per-stage split, because ZeRO's lifetime pass prunes gathers there
    (the last stage's backward keeps its forward's parameters). A send and its recv land
    on different ranks with no edge between them; pairing them by index restores the
    cross-device edge.
    """
    def __init__(self, dags):
        self.nodes, self.edges = {}, []
        for d in dags:
            self.nodes.update(d.nodes)
            self.edges.extend(d.edges)
        Edge = type(self.edges[0])
        for uid, n in self.nodes.items():
            if n.node_kind == "SEND_COMM":
                peer = "recv." + uid.split(".", 1)[1]
                if peer in self.nodes:
                    self.edges.append(Edge(src_uid=uid, dst_uid=peer, dep_kind="data"))


def lower(argv):
    _reset_annotation_state()
    with contextlib.redirect_stdout(io.StringIO()):
        dump_dag.main(argv)
    return Union(piper_metadata.per_pp_training_dags), piper_metadata.schedule_directives


def declared(directives):
    d = {"zero": False, "shard_grads": False, "replicate": False, "ring_tensors": set(), "tp": False}
    for x in directives:
        if x.get("op") == "replicate":
            d["replicate"] = True
            d["zero"] |= bool(x.get("shard_params"))
            d["shard_grads"] |= bool(x.get("shard_grads") or x.get("shard_params"))
        if x.get("op") == "shard_tensor":
            d["tp"] = True
        if x.get("op") == "ring_exchange":
            d["ring_tensors"] |= set(x.get("tensors") or [])
    return d


def region(n, decl):
    """TP only where a shard_tensor directive declares it; the tag alone is an annotation."""
    if "TP" in n.tag and decl["tp"]:
        return "TP"
    return "CP" if "CP" in n.tag else "rep"


def splice(dag):
    """Compute-to-compute data edges, with the communication nodes found on each path."""
    comp = {u for u, n in dag.nodes.items() if n.node_kind in ("COMPUTE", "UPD")}
    out = defaultdict(list)
    for e in dag.edges:
        if e.dep_kind == "data":
            out[e.src_uid].append(e)
    edges = []
    for e in dag.edges:
        if e.dep_kind != "data" or e.src_uid not in comp:
            continue
        stack = [(e.dst_uid, [], e.tensor_name)]
        while stack:
            w, path, name = stack.pop()
            if w in comp:
                edges.append((e.src_uid, w, name, tuple(path)))
                continue
            for f in out[w]:
                stack.append((f.dst_uid, path + [w], name or f.tensor_name))
    return edges


def predict_edge(dag, u, v, name, decl, lattice):
    a, b = dag.nodes[u], dag.nodes[v]
    if b.node_kind == "UPD":
        return None                       # gradient side, predicted from flags below
    if tuple(a.device or ()) != tuple(b.device or ()):
        return "send_recv" if lattice == "perm+group" else "none"
    fwd = a.tag.get("PASS") == "F"
    ra, rb = region(a, decl), region(b, decl)
    if ra == "TP" and rb != "TP":
        return "all_reduce"               # fwd: partial output to replicated; bwd: partial input grad
    if ra == "CP" and rb == "CP" and carries_ring_tensor(dag, a if fwd else b, decl):
        return "permute" if lattice != "spmd" else "none"
    return "none"


def carries_ring_tensor(dag, node, decl):
    """Does the forward boundary behind this edge forward a tensor the ring rotates?

    Read from the segment's own boundary record (name, source name, forwarded), which
    the frontend writes before any directive runs.
    """
    fwd = dag.nodes.get(node.node_meta.get("fwd_uid", node.uid), node)
    outs = (fwd.node_meta.get("a2a_boundary_after") or {}).get("outputs", [])
    return any(o.get("forwarded") and o.get("src_name") in decl["ring_tensors"] for o in outs)


def param_predictions(dag, decl):
    """Gathers before compute and gradient reductions after backward, from the flags."""
    pred = Counter()
    for n in dag.nodes.values():
        if n.node_kind != "COMPUTE":
            continue
        f = dag.nodes.get(n.node_meta.get("fwd_uid", n.uid), n)
        if not f.node_meta.get("param_idxs") or region(n, decl) == "TP":
            continue
        fwd = n.compute_subkind == "FWD"
        if decl["zero"]:
            if fwd:
                pred[("all_gather", n.uid)] += 1
            elif f.node_meta.get("zero_free_full_params_after"):
                pred[("all_gather", n.uid)] += 1
        if not fwd and decl["replicate"]:
            pred[("reduce_scatter" if decl["shard_grads"] else "all_reduce", n.uid)] += 1
    return pred


def param_actual(dag):
    act = Counter()
    for n in dag.nodes.values():
        if n.node_kind not in PARAM_SIDE:
            continue
        m = n.node_meta
        anchor = m.get("compute_uid") or m.get("bwd_uid")
        act[(KIND[n.node_kind], anchor)] += 1
    return act


def score(case, argv, lattice):
    dag, directives = lower(argv)
    decl = declared(directives)
    r = Counter()
    misses = []
    for u, v, name, path in splice(dag):
        p = predict_edge(dag, u, v, name, decl, lattice)
        if p is None:
            continue
        actual = {KIND[dag.nodes[c].node_kind] for c in path if dag.nodes[c].node_kind not in PARAM_SIDE}
        a = actual.pop() if len(actual) == 1 else ("none" if not actual else "+".join(sorted(actual)))
        if p == a:
            r["agree, comm" if a != "none" else "agree, none"] += 1
        else:
            r["missed" if p == "none" else "spurious" if a == "none" else "wrong kind"] += 1
            misses.append((u, v, name, p, a))
    pp, pa = param_predictions(dag, decl), param_actual(dag)
    for k in set(pp) | set(pa):
        agree = min(pp[k], pa[k])
        r["agree, param"] += agree
        r["missed"] += pa[k] - agree
        r["spurious"] += pp[k] - agree
        if pp[k] != pa[k]:
            misses.append(("param", k[1], "", f"{pp[k]}x {k[0]}", f"{pa[k]}x"))
    return r, misses


def main():
    cols = ["agree, comm", "agree, param", "agree, none", "missed", "spurious", "wrong kind"]
    rc = 0
    for lattice in ("spmd", "perm", "perm+group"):
        print(f"\n=== lattice: {lattice}")
        print(f"{'case':<18}" + "".join(f"{c:>14}" for c in cols))
        for case, argv in CASES.items():
            r, misses = score(case, argv, lattice)
            print(f"{case:<18}" + "".join(f"{r.get(c, 0):>14}" for c in cols))
            for m in misses[:4]:
                print(f"    {m[0]} -> {m[1]}  [{m[2]}]  predicted {m[3]}, found {m[4]}")
            if len(misses) > 4:
                print(f"    ... {len(misses) - 4} more")
            if lattice == "perm+group" and misses:
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
