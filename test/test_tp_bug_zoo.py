"""TP regions written wrong (log F99): shard_tensor lowers every one of them; layout refuses each at
compile time and names the operator that would need a collective inside the region."""
import sys

import pytest

sys.path[:0] = ["experiments", "examples"]
import tp_bug_zoo as z  # noqa: E402

ST = {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream"}
LY = {"op": "layout", "filter": {"TP": "*"}, "devices": [0, 1], "axis": "tp",
      "params": {"up": "colwise", "down": "rowwise"}, "stream": "tp_stream"}
WRONG = {"bias": "aten::addmm", "norm": "aten::native_layer_norm", "act": "aten::gelu",
         "residual": "aten::add", "colonly": "local shard"}


def test_correct_region_lowers_the_same_both_ways() -> None:
    assert z.lower("ok", ST) == z.lower("ok", LY) == "lowers (2 TP all-reduces)"
    assert z.simulate("ok") < 1e-12


@pytest.mark.parametrize("kind", sorted(WRONG))
def test_wrong_region_passes_shard_tensor_and_is_refused_by_layout(kind) -> None:
    assert z.lower(kind, ST).startswith("lowers")          # the hand-written rule accepts it
    err = z.simulate(kind)
    assert err != err or err > 0.05                         # and would train wrong
    got = z.lower(kind, LY)
    assert got.startswith("refused") and WRONG[kind] in got, got
