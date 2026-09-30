"""Synthetic SmolVLM2-2.2B inputs at DocVQA's real shapes, for timing only (log F99).

    PYTHONPATH=examples:. python experiments/make_synth_vlm2.py --out DIR [--n 12 --tiles 17 --length 1488]

Random tiles, token ids, labels and weights (seeded); image tokens at one contiguous block per
example. Same file layout as prepare_smolvlm2.py writes, so test_smolvlm.py reads it unchanged.
"""
import argparse, os, sys
import torch
sys.path[:0] = ["examples", "."]
from models.smolvlm import CFG_SMOLVLM2, SmolVLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--tiles", type=int, default=17); ap.add_argument("--length", type=int, default=1488)
a = ap.parse_args()
g = torch.Generator().manual_seed(0)
n_img = (384 // 14) ** 2 // 9
T, L = a.tiles, a.length
start = 30
ids = torch.randint(0, 49000, (a.n, L), generator=g)
labels = torch.full_like(ids, -100); labels[:, start + T * n_img + 10:] = ids[:, start + T * n_img + 10:]
img_index = (start + torch.arange(T * n_img)).expand(a.n, -1).contiguous()
images = torch.randint(0, 256, (a.n * T, 384, 384, 3), generator=g, dtype=torch.uint8)
os.makedirs(a.out, exist_ok=True)
torch.save({"images": images, "tiles": T, "input_ids": ids, "labels": labels, "img_index": img_index,
            "config": CFG_SMOLVLM2, "image_offset": None}, os.path.join(a.out, "data.pt"))
torch.manual_seed(0)
m = SmolVLM(None, config=CFG_SMOLVLM2)
torch.save({k: v for k, v in m.state_dict().items()}, os.path.join(a.out, "weights.pt"))
print("wrote", a.out, "examples", a.n, "tiles", T, "length", L)
