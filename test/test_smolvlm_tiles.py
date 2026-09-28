"""SmolVLM tiled mode (log F97): scattering image features at given positions equals the contiguous
merge when the positions are contiguous, and a tiled model lowers with the positions as a model input."""
import json
import sys

import torch

sys.path[:0] = ["experiments", "examples"]
from models.smolvlm import SmolVLM  # noqa: E402

CFG = dict(v_layers=1, t_layers=1)


def test_scatter_equals_contiguous_merge() -> None:
    torch.manual_seed(0)
    a = SmolVLM(5, config=CFG).eval()
    b = SmolVLM(None, config=CFG).eval()
    b.load_state_dict(a.state_dict())
    px, ids = torch.randn(2, 3, 512, 512), torch.randint(0, 1000, (2, 80))
    idx = torch.arange(5, 5 + a.n_img).expand(2, -1)
    with torch.no_grad():
        assert torch.equal(a(px, ids), b(px, ids, idx))


def test_tiled_model_lowers_with_positions_as_input() -> None:
    import probe_route as p
    sched = [{"op": "route", "mode": "consumers"}] + [{"op": "place", "filter": {"PP": i}, "devices": [0]} for i in range(3)] + \
            [{"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}]
    make = lambda: (SmolVLM(None, config=CFG).float(),  # noqa: E731
                    (torch.empty(6, 3, 512, 512, device="meta"), torch.zeros(2, 400, dtype=torch.long, device="meta"),
                     torch.zeros(2, 3 * 64, dtype=torch.long, device="meta")))
    _, dags, note = p.lower(sched, make)
    assert note == "", note
    dec = [n for n in dags[0].nodes.values() if n.node_kind == "COMPUTE" and n.compute_subkind == "FWD" and n.tag.get("PP") == 2]
    assert dec and ("model", 2, 0) in [tuple(s) for s in dec[0].node_meta["input_sources"]]
