"""Forward and backward time of each pipeline stage at each microbatch size, one GPU (log F78).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=examples:. python experiments/profile_stages.py --model clip --data DIR
    ... --model smolvlm --data DIR

A stage is the set of regions one GPU runs. Its input is detached, as at a Piper
stage boundary, so its backward stops there. Prints JSON: {stage: {batch: [fwd_ms, bwd_ms]}}.
"""
import argparse
import json
import sys
import time

import torch

sys.path[:0] = ["examples", "."]


def timed(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
    return sorted(ts)[n // 2] * 1e3


def fwd_bwd(stage, inputs):
    """(fwd_ms, bwd_ms) of stage(*inputs) -> tensor, with a unit upstream gradient."""
    ins = [x.detach().requires_grad_(x.is_floating_point()) for x in inputs]
    t_f = timed(lambda: stage(*ins))
    def fb():
        out = stage(*ins)
        out.backward(torch.ones_like(out))
    return t_f, max(timed(fb) - t_f, 0.0)


def smolvlm_stages(data_dir):
    from models.smolvlm import SmolVLM
    d = torch.load(f"{data_dir}/data.pt")
    m = SmolVLM(d["image_offset"]).cuda()
    m.load_state_dict(torch.load(f"{data_dir}/weights.pt"))
    vm, tm, off = m.model.vision_model, m.model.text_model, d["image_offset"]
    L = vm.encoder.layers

    def vis(lo, hi, first, last):
        def f(x):
            x = vm.embeddings(x) if first else x
            for layer in L[lo:hi]:
                x = layer(x)
            return m.model.connector(vm.post_layernorm(x)) if last else x
        return f

    def dec(ids, img):
        txt = tm.embed_tokens(ids)
        h = torch.cat([txt[:, :off], img, txt[:, off + 64:]], 1)
        cos, sin = m._rope(h.shape[1], h.device, h.dtype)
        for layer in tm.layers:
            h = layer(h, cos, sin)
        return m.lm_head(tm.norm(h))

    def make(b):
        px = ((d["images"][:b].permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).cuda()
        ids = d["input_ids"][:b].cuda()
        with torch.no_grad():
            h1 = vis(0, 4, True, False)(px); h2 = vis(4, 8, False, False)(h1)
            img = vis(8, 12, False, True)(h2); img_all = vis(0, 12, True, True)(px)
        return {"vision": (vis(0, 12, True, True), [px]), "vision_1": (vis(0, 4, True, False), [px]),
                "vision_2": (vis(4, 8, False, False), [h1]), "vision_3": (vis(8, 12, False, True), [h2]),
                "decoder": (dec, [ids, img_all])}
    return make, (32, 16, 8, 4, 2)


def clip_stages(data_dir):
    import torch.nn.functional as F
    from models.clip import CLIP
    d = torch.load(f"{data_dir}/data.pt")
    m = CLIP().cuda(); m.load_state_dict(torch.load(f"{data_dir}/weights.pt"))
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def head(img, txt):
        img = img / img.norm(dim=-1, keepdim=True); txt = txt / txt.norm(dim=-1, keepdim=True)
        return m.logit_scale.exp() * img @ txt.t()

    def make(b):
        x = d["images"][:b].permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        px, ids = ((x - mean) / std).cuda(), d["input_ids"][:b].cuda()
        with torch.no_grad():
            img, txt = m.visual_projection(m.vision_model(px)), m.text_projection(m.text_model(ids))
        return {"vision": (lambda p: m.visual_projection(m.vision_model(p)), [px]),
                "text": (lambda i: m.text_projection(m.text_model(i)), [ids]),
                "head": (head, [img, txt])}
    return make, (512, 256, 128, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("clip", "smolvlm"), required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    make, sizes = (clip_stages if args.model == "clip" else smolvlm_stages)(args.data)
    out = {}
    for b in sizes:
        for name, (fn, ins) in make(b).items():
            out.setdefault(name, {})[b] = [round(v, 2) for v in fwd_bwd(fn, ins)]
        torch.cuda.empty_cache()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
