"""Fetch real CLIP weights and real image-caption pairs, and check examples/models/clip.py.

    HF_HOME=/var/tmp/ziweizho-hf PYTHONPATH=examples:. python experiments/prepare_clip.py --out /var/tmp/ziweizho-clip

Writes, under --out:
  weights.pt  the released openai/clip-vit-base-patch32 weights, by our parameter names
  data.pt     CIFAR-10 test images (uint8, 32x32) with one caption each (CLIP's CIFAR
              prompt templates), tokenized with CLIP's tokenizer to 77 tokens
and checks (1) our logits against Hugging Face's CLIPModel on the same inputs, and
(2) zero-shot CIFAR-10 accuracy with the pretrained weights. transformers and
datasets are needed here only; Piper and the model file do not import them.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path[:0] = ["examples", "."]
from models.clip import CLIP  # noqa: E402

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
# CLIP's prompt templates for CIFAR-10 (openai/CLIP, notebooks/Prompt_Engineering_for_ImageNet.ipynb and data/prompts.md)
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
             "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
             "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.",
             "a photo of the {}.", "a blurry photo of the {}.", "a black and white photo of the {}.",
             "a low contrast photo of the {}.", "a high contrast photo of the {}.", "a bad photo of the {}.",
             "a good photo of the {}.", "a photo of the small {}.", "a photo of the big {}."]


def pixels(images_u8):
    """uint8 [N, 32, 32, 3] -> CLIP-normalised float [N, 3, 224, 224]."""
    x = images_u8.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/tmp/ziweizho-clip")
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset
    from transformers import CLIPModel, CLIPTokenizer

    hf = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    ours = CLIP().eval()
    sd = {k: v.float() for k, v in hf.state_dict().items() if not k.endswith("position_ids")}
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    torch.save(sd, os.path.join(args.out, "weights.pt"))

    ds = load_dataset("uoft-cs/cifar10", split=f"test[:{args.n}]")
    names = ds.features["label"].names
    import numpy as np
    images = torch.from_numpy(np.stack([np.asarray(im.convert("RGB")) for im in ds["img"]]))
    labels = torch.tensor(ds["label"])
    g = torch.Generator().manual_seed(0)
    tpl = torch.randint(len(TEMPLATES), (len(labels),), generator=g)
    captions = [TEMPLATES[t].format(names[c]) for t, c in zip(tpl.tolist(), labels.tolist())]
    ids = tok(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    torch.save({"images": images, "labels": labels, "captions": captions, "input_ids": ids,
                "class_names": names}, os.path.join(args.out, "data.pt"))

    with torch.no_grad():
        px, tid = pixels(images[:16]), ids[:16]
        ref = hf(pixel_values=px, input_ids=tid).logits_per_image
        got = ours(px, tid)
        print(f"logits vs Hugging Face CLIPModel: max |diff| {(ref - got).abs().max():.2e} (16 pairs)")

        prompts = tok([f"a photo of a {c}." for c in names], padding="max_length", max_length=77,
                      return_tensors="pt").input_ids
        correct = 0
        for i in range(0, len(labels), 256):
            logits = ours(pixels(images[i:i + 256]), prompts)
            correct += (logits.argmax(-1) == labels[i:i + 256]).sum().item()
        print(f"zero-shot CIFAR-10 accuracy, {len(labels)} test images: {correct / len(labels):.3f}")
    print(f"wrote {args.out}/weights.pt, {args.out}/data.pt ({len(labels)} pairs)")


if __name__ == "__main__":
    main()
