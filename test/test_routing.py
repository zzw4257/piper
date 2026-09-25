"""Consumer routing: branches stop chaining, linear models lower exactly as before (log F71)."""
import sys

import torch

sys.path[:0] = ["experiments", "examples"]
import probe_route as p  # noqa: E402

ORDERS = [{"op": "order", "filters": [[{"PP": i, "PASS": "F"}], [{"PP": i, "PASS": "B"}]]} for i in range(3)]
PLACE3 = [{"op": "place", "filter": {"PP": i}, "devices": [i]} for i in range(3)]


def _branches():
    return p.TwoBranch(), (torch.empty(8, 64, device="meta"), torch.empty(8, 64, device="meta"))


def _stage_edges(dag):
    succ = {}
    for e in dag.edges:
        if e.dep_kind == "data":
            succ.setdefault(e.src_uid, []).append(e.dst_uid)
    out = set()
    for u, n in dag.nodes.items():
        if n.compute_subkind != "FWD":
            continue
        for v in succ.get(u, []):
            vs = succ.get("recv." + v.split(".")[1], []) if v.startswith("send.") else [v]
            out |= {(u, w) for w in vs if dag.nodes[w].compute_subkind == "FWD"}
    return out


def test_threading_chains_the_branches() -> None:
    dag, _, note = p.lower(PLACE3 + [p.SPLIT] + ORDERS, _branches)
    assert note == ""
    assert _stage_edges(dag) == {("s0.seg0", "s1.seg1"), ("s1.seg1", "s2.seg2")}


def test_consumer_routing_unchains_the_branches() -> None:
    dag, _, note = p.lower([p.ROUTE] + PLACE3 + [p.SPLIT] + ORDERS, _branches)
    assert note == ""
    assert _stage_edges(dag) == {("s0.seg0", "s2.seg2"), ("s1.seg1", "s2.seg2")}
    assert dag.nodes["s1.seg1"].node_meta["input_sources"] == [("model", 1, 0)]
    assert dag.nodes["s2.seg2"].node_meta["input_sources"] == [("seg", 0, 0), ("seg", 1, 0)]


def test_consumer_routing_leaves_tp_ep_and_cp_lowerings_unchanged() -> None:
    import probe_route
    for name, sched, make in [
        ("tp", [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
                {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"}, p.SPLIT],
         lambda: (p.TPMlp(32, 128, 2, 1), (torch.empty(8, 32, device="meta"),))),
        ("tp_derived", [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
                        {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream",
                         "params": {"up": "colwise", "down": "rowwise"}}, p.SPLIT],
         lambda: (p.TPMlp(32, 128, 1, 1), (torch.empty(8, 32, device="meta"),))),
        ("ep", [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
                {"op": "shard", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "ep_stream"}, p.SPLIT],
         lambda: (p.TPMlp(32, 128, 2, 1), (torch.empty(8, 32, device="meta"),))),
        ("cp", [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1, 2, 3]},
                {"op": "ring_exchange", "filter": {"CP": "*"}, "devices": [0, 1, 2, 3], "tensors": ["k", "v"],
                 "stream": "cp_stream", "hoist": True, "distance": 1}, p.SPLIT],
         lambda: (p.RingAttn(64, 2, 4), tuple(torch.empty(2, 8, 64, device="meta") for _ in range(3)))),
    ]:
        _, a, na = probe_route.lower(sched, make)
        _, b, nb = probe_route.lower([p.ROUTE] + sched, make)
        assert na == nb == "", (name, na, nb)
        assert probe_route.shape(a) == probe_route.shape(b), name


def _tp_branches_schedule(route):
    s = [{"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
         {"op": "place", "filter": {"PP": 1}, "devices": [2, 3]},
         {"op": "place", "filter": {"PP": 2}, "devices": [2, 3]},
         {"op": "shard_tensor", "filter": {"PP": 0, "TP": "*"}, "devices": [0, 1], "stream": "tp_stream"},
         {"op": "shard_tensor", "filter": {"PP": 1, "TP": "*"}, "devices": [2, 3], "stream": "tp_stream"},
         p.SPLIT,
         {"op": "order", "filters": [[{"PP": 0, "PASS": "F"}], [{"PP": 0, "PASS": "B"}]]},
         {"op": "order", "filters": [[{"PP": 1, "PASS": "F"}], [{"PP": 2, "PASS": "F"}],
                                     [{"PP": 2, "PASS": "B"}], [{"PP": 1, "PASS": "B"}]]}]
    return ([p.ROUTE] if route else []) + s


def _tp_branches():
    from models.branches import Branches
    return Branches(64, 2, 2, 1, hidden=256, tp_degree=2), (torch.empty(8, 64, device="meta"),
                                                            torch.empty(8, 64, device="meta"))


def test_tp_inside_a_branch_needs_routing() -> None:
    # Threaded, the image TP region would relay x_txt and its own input: one all-reduce cannot serve three tensors.
    _, _, note = p.lower(_tp_branches_schedule(False), _tp_branches)
    assert note, "threaded lowering must refuse a TP region that relays other values"
    dag, _, note = p.lower(_tp_branches_schedule(True), _tp_branches)
    assert note == ""
    # The TP region reads the residual and emits only its partial sum; the residual goes straight to post.
    assert dag.nodes["s0.seg1"].node_meta["input_sources"] == [("seg", 0, 0)]
    assert dag.nodes["s0.seg2"].node_meta["input_sources"] == [("seg", 0, 0), ("seg", 1, 0)]
    data = {(e.src_uid, e.dst_uid) for e in dag.edges if e.dep_kind == "data"}
    assert {("s0.seg1", "tp_all_reduce.0"), ("tp_all_reduce.0", "s0.seg2"), ("s0.seg0", "s0.seg2")} <= data
    # Backward: the residual's gradient reaches seg0 twice, once all-reduced through the TP region.
    assert {("s0.seg2.bwd", "s0.seg0.bwd"), ("s0.seg1.bwd", "tp_all_reduce.1"),
            ("tp_all_reduce.1", "s0.seg0.bwd")} <= data
    assert sum(n.node_kind == "TP_COMM" for n in dag.nodes.values()) == 4
