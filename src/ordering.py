from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    _has_path,
    _topological_levels,
    _topological_order,
)

_DEFAULT_STREAM = "default_stream"
_CRITICAL_PATH_COMM_KINDS = {
    "ALL_GATHER_COMM",
    "A2A_COMM",
    "TP_COMM",
    "RING_COMM",
}
_REDUCTION_COMM_KINDS = {
    "REDUCE_COMM",
    "REDUCE_SCATTER_COMM",
}


def _serial_topological_order(
    dag: TrainingDAG,
    topo_levels: dict[str, int] | None = None,
) -> list[str]:
    """Serialize topological levels into a deterministic dispatch order.

    Nodes with lower topological levels always come first. Within a level,
    priority order is: SEND > critical-path comm > reduction comm > compute/other > RECV.
    """
    if topo_levels is None:
        topo_levels = _topological_levels(dag)

    base_order = _topological_order(dag)
    topo_idx = {uid: i for i, uid in enumerate(base_order)}

    def node_priority(uid: str) -> int:
        kind = dag.nodes[uid].node_kind
        if kind == "SEND_COMM":
            return 0
        if kind in _CRITICAL_PATH_COMM_KINDS:
            return 1
        if kind in _REDUCTION_COMM_KINDS:
            return 2
        if kind == "RECV_COMM":
            return 4
        return 3

    return sorted(
        dag.nodes,
        key=lambda uid: (topo_levels[uid], node_priority(uid), topo_idx[uid]),
    )


def _resolve_default_stream_order(dag: TrainingDAG) -> None:
    """Create a total ordering over default-stream COMPUTE nodes.

    Whenever multiple default-stream compute nodes share a topological level,
    chain them with temporal edges in descending order of downstream
    dependencies (more downstream nodes -> earlier in the chain). Downstream
    count is the size of each compute node's transitive successor set, which in
    a well-formed training DAG terminates at UPD.

    The new edges shift topological levels, so after each chain insertion we
    recompute levels and rescan from the earliest level. The pass terminates
    when every default-stream compute node sits at a unique level.
    """
    def _is_default_compute(uid: str) -> bool:
        node = dag.nodes[uid]
        return node.stream == _DEFAULT_STREAM and node.node_kind == "COMPUTE"

    compute_uids = [uid for uid in dag.nodes if _is_default_compute(uid)]
    if len(compute_uids) < 2:
        return

    def _downstream_count(uid: str) -> int:
        seen: set[str] = set()
        stack = list(dag.succs.get(uid, set()))
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(dag.succs.get(v, set()))
        return len(seen)

    while True:
        topo_levels = _topological_levels(dag)
        by_level: dict[int, list[str]] = {}
        for uid in compute_uids:
            by_level.setdefault(topo_levels[uid], []).append(uid)

        conflict_level: int | None = None
        for level in sorted(by_level):
            if len(by_level[level]) > 1:
                conflict_level = level
                break
        if conflict_level is None:
            return

        # Same-level nodes have no path between them, otherwise their levels
        # would differ. Order by downstream count desc; break ties by uid.
        group = by_level[conflict_level]
        ordered = sorted(group, key=lambda u: (-_downstream_count(u), u))
        for src, dst in zip(ordered, ordered[1:]):
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=src,
                    dst_uid=dst,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )


def resolve_total_order_per_stream(dag: TrainingDAG) -> None:
    """Serialize the default stream, then strictly chain non-default streams.

    The default-stream pass runs first so the per-non-default-stream anchors
    below see the post-serialization topological order of default-stream nodes.

    Non-default stream nodes are ordered by the topological order of the
    default-stream nodes they directly depend on or are depended on by.
    """
    _resolve_default_stream_order(dag)

    topo = _topological_order(dag)
    topo_idx = {uid: i for i, uid in enumerate(topo)}
    topo_levels = _topological_levels(dag)
    default_uids = [
        uid for uid in topo
        if dag.nodes[uid].stream == _DEFAULT_STREAM
    ]
    streams = sorted({
        n.stream for n in dag.nodes.values()
        if n.stream != _DEFAULT_STREAM
    })
    if not default_uids or not streams:
        return

    for stream in streams:
        associated_by_default: dict[str, list[str]] = {uid: [] for uid in default_uids}
        default_anchors_by_stream_uid: dict[str, list[str]] = {}
        for edge in dag.edges:
            src = dag.nodes[edge.src_uid]
            dst = dag.nodes[edge.dst_uid]
            if src.stream == _DEFAULT_STREAM and dst.stream == stream:
                default_anchors_by_stream_uid.setdefault(edge.dst_uid, []).append(edge.src_uid)
            elif src.stream == stream and dst.stream == _DEFAULT_STREAM:
                default_anchors_by_stream_uid.setdefault(edge.src_uid, []).append(edge.dst_uid)

        for stream_uid in {
            uid for uid, node in dag.nodes.items()
            if node.stream == stream
        }:
            default_anchors = default_anchors_by_stream_uid.get(stream_uid, [])
            if not default_anchors:
                continue
            anchor_uid = min(default_anchors, key=lambda uid: (topo_levels[uid], topo_idx[uid]))
            associated_by_default.setdefault(anchor_uid, []).append(stream_uid)

        current_uid: str | None = None
        for default_uid in default_uids:
            stream_uids = sorted(
                set(associated_by_default.get(default_uid, [])),
                key=lambda u: (topo_levels[u], topo_idx[u]),
            )
            for next_uid in stream_uids:
                if current_uid is not None and current_uid != next_uid:
                    if _has_path(dag, next_uid, current_uid):
                        raise ValueError(
                            "resolve_total_order_per_stream would create a cycle while ordering "
                            f"stream={stream}: {current_uid} -> {next_uid}"
                        )
                    if not _has_path(dag, current_uid, next_uid):
                        dag.add_edge(
                            TrainingDAGEdge(
                                src_uid=current_uid,
                                dst_uid=next_uid,
                                dep_kind="temporal",
                                tensor_name=None,
                            )
                        )
                current_uid = next_uid



def align_p2p_order(dags: list) -> int:
    """Give every pair of ranks one order for the point-to-point transfers between them.

    NCCL pairs point-to-point operations between two ranks by issue order, not by name.
    Each rank serializes its own DAG, breaking ties by uid string, so two same-shaped
    transfers between one pair could be matched crosswise and silently swapped
    (log F81: SmolVLM's vision chunks; the receiver ordered recv.10 before recv.8, and
    a last-chunk-first backward received in the reverse of the send order).

    Run on the per-rank DAGs before each stream is serialized. For each (sender,
    receiver) pair, the order the sender's DAG already forces between its sends and the
    order the receiver's forces between its recvs are merged; ties follow the sender's
    order. Ties go to the transfer the receiver needs first (the topological level of
    the recv's earliest consumer), then to the one the sender has first (the send's
    level): a last-chunk-first backward then receives the last chunk first, and a
    forward whose consumer needs all chunks at once receives them as they are produced.
    A contradiction is an error. Both sides then get temporal edges in that order,
    repeated until no pair changes. Returns the number of edges added.
    """
    import heapq

    from .dag import TrainingDAGEdge

    added = 0
    for _ in range(4 * len(dags) + 4):
        pairs: dict[tuple[int, int], list[str]] = {}
        for r, d in enumerate(dags):
            for uid in _serial_topological_order(d):
                n = d.nodes[uid]
                if n.node_kind == "SEND_COMM":
                    pairs.setdefault((r, int(n.node_meta["peer_pp_rank"])), []).append(uid.split(".", 1)[1])
        changed = False
        levels = [_topological_levels(d) for d in dags]
        for (src, dst), keys in pairs.items():
            sd, rd = dags[src], dags[dst]
            keys = [k for k in keys if f"recv.{k}" in rd.nodes]

            def need(k):
                cons = rd.succs.get(f"recv.{k}", set())
                return min((levels[dst][c] for c in cons), default=levels[dst][f"recv.{k}"])

            rank = {k: (need(k), levels[src][f"send.{k}"], i) for i, k in enumerate(keys)}
            succ = {k: set() for k in keys}
            indeg = {k: 0 for k in keys}
            for a in keys:
                for b in keys:
                    if a != b and (_has_path(sd, f"send.{a}", f"send.{b}") or _has_path(rd, f"recv.{a}", f"recv.{b}")):
                        succ[a].add(b)
            for a in keys:
                for b in succ[a]:
                    indeg[b] += 1
            heap = [(rank[k], k) for k in keys if indeg[k] == 0]
            heapq.heapify(heap)
            order = []
            while heap:
                _, k = heapq.heappop(heap)
                order.append(k)
                for b in succ[k]:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        heapq.heappush(heap, (rank[b], b))
            if len(order) != len(keys):
                raise ValueError(
                    f"ranks {src} and {dst} force opposite orders on their transfers {sorted(keys)[:6]}; "
                    "the schedule orders the two ranks inconsistently")
            for a, b in zip(order, order[1:]):
                for d, u, v in ((sd, f"send.{a}", f"send.{b}"), (rd, f"recv.{a}", f"recv.{b}")):
                    if not _has_path(d, u, v):
                        d.add_edge(TrainingDAGEdge(src_uid=u, dst_uid=v, dep_kind="temporal", tensor_name=None))
                        added += 1
                        changed = True
        if not changed:
            return added
    raise ValueError("point-to-point order did not settle across ranks")
