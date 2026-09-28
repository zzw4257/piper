"""Megatron TP on a real transformer (SmolVLM, log F87): the backward all-reduce lands on the
input the region reads and needs a gradient for, never on the residual or the rotary tables."""
import json
import sys

import torch

sys.path[:0] = ["experiments", "examples"]
import probe_route as p  # noqa: E402
from models.smolvlm import SmolVLM  # noqa: E402

SCHED = json.load(open("examples/base-schedules/vlm_tp3.json"))


def make():
    m = SmolVLM(5, 1, 1, dict(v_layers=2, t_layers=2, tp=3)).float()
    return m, (torch.empty(2, 3, 512, 512, device="meta"), torch.zeros(2, 80, dtype=torch.long, device="meta"))


def test_one_all_reduce_each_way_per_tp_region_on_the_read_input() -> None:
    _, dags, note = p.lower(SCHED, make)
    assert note == "", note
    dag = dags[0]
    regions = [n for n in dag.nodes.values() if n.node_kind == "COMPUTE" and n.compute_subkind == "FWD" and "TP" in n.tag]
    assert len(regions) == 8  # 2 vision + 2 decoder layers, attention and MLP each
    comms = [n for n in dag.nodes.values() if n.node_kind == "TP_COMM"]
    assert sorted(n.tag["PASS"] for n in comms) == ["B"] * 8 + ["F"] * 8
    for n in comms:
        if n.tag["PASS"] != "B":
            continue
        fwd = dag.nodes[dag.nodes[n.node_meta["source_uid"]].node_meta["fwd_uid"]].node_meta
        slot = n.node_meta["tp_tensor_idx"]
        assert fwd["graphargs"][fwd["input_idxs"][slot]].requires_grad
