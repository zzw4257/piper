"""Every pair of ranks issues its point-to-point transfers in one order (log F81).

NCCL matches sends and recvs between two ranks by issue order. Same-shaped transfers
whose order differs between the two sides are silently swapped.
"""
import json
import sys

import pytest
import torch

sys.path[:0] = ["experiments", "examples"]
import probe_route as p  # noqa: E402
from src.ordering import _serial_topological_order  # noqa: E402


def _orders(per):
    sends, recvs = {}, {}
    for r, d in enumerate(per):
        for uid in _serial_topological_order(d):
            n = d.nodes[uid]
            if n.node_kind == "SEND_COMM":
                sends.setdefault((r, int(n.node_meta["peer_pp_rank"])), []).append(uid.split(".", 1)[1])
            elif n.node_kind == "RECV_COMM":
                recvs.setdefault((int(n.node_meta["peer_pp_rank"]), r), []).append(uid.split(".", 1)[1])
    return sends, recvs


def _vlm(k):
    from models.smolvlm import SmolVLM
    return lambda: (SmolVLM(5, 1, 3, None, k), (torch.empty(8, 3, 512, 512, device="meta"),
                                                torch.zeros(8, 112, dtype=torch.long, device="meta")))


def _clip():
    from models.clip import CLIP
    return CLIP(), (torch.empty(4, 3, 224, 224, device="meta"), torch.zeros(4, 77, dtype=torch.long, device="meta"))


CASES = [
    ("vlm_4st_routed_c4_mb1", _vlm(4)),     # 4 same-shaped chunks from rank 2 to rank 3, backward last-first
    ("vlm_4st_routed_c8_mb1", _vlm(8)),
    ("vlm_4st_routed_c2_mb4", _vlm(2)),
    ("vlm_4st_routed_mb8_1f1b", _vlm(1)),
    ("clip_routed_mb8_1f1b", _clip),
    ("pp2_tp2_mb4_1f1b", lambda: (p.TPMlp(32, 128, 2, 2), (torch.empty(8, 32, device="meta"),))),
]


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_both_sides_of_every_pair_use_one_order(name, make) -> None:
    _, per, note = p.lower(json.load(open(f"examples/base-schedules/{name}.json")), make)
    assert note == "", note
    sends, recvs = _orders(per)
    assert sends.keys() == recvs.keys()
    for pair in sends:
        assert sends[pair] == recvs[pair], (pair, sends[pair], recvs[pair])
