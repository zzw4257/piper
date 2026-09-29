"""layout derives every-layer TP on real VLM structures node for node with shard_tensor (log F99):
attention reshapes by the local head count, so the region runs local between its column- and
row-parallel layers, as torchtitan's styles do."""
import sys

import torch

sys.path[:0] = ["experiments", "examples"]
import layout_vlm as lv  # noqa: E402
from models.smolvlm import CFG_SMOLVLM2  # noqa: E402


def test_smolvlm_tp3() -> None:
    assert lv.run("SmolVLM-256M, TP=3", {}, 3,
                  (torch.empty(2, 3, 512, 512, device="meta"), torch.zeros(2, 80, dtype=torch.long, device="meta")), 5)


def test_smolvlm2_tp2() -> None:
    assert lv.run("SmolVLM2-2.2B, TP=2", CFG_SMOLVLM2, 2,
                  (torch.empty(4, 3, 384, 384, device="meta"), torch.zeros(2, 200, dtype=torch.long, device="meta"),
                   torch.zeros(2, 162, dtype=torch.long, device="meta")), None)
