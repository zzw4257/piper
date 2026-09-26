"""Real SmolVLM-256M weights and real COCO image-caption pairs for examples/models/smolvlm.py.

    PYTHONPATH=examples:. python experiments/prepare_smolvlm.py --model DIR --coco FILE.parquet --out DIR

Writes, under --out:
  weights.pt  the released HuggingFaceTB/SmolVLM-256M-Instruct weights, by our parameter names
  data.pt     n COCO val2017 images (uint8, 512x512) and their first caption, in SmolVLM's
              chat template, tokenized and right-padded to one length, with next-token
              labels on the caption only
and checks our logits and loss against Hugging Face's Idefics3ForConditionalGeneration,
then lets the reference caption a few images. transformers is needed here only.
"""
import argparse
import io
import os
import sys

import torch

sys.path[:0] = ["examples", "."]
from models.smolvlm import SmolVLM, lm_loss  # noqa: E402

PROMPT = "Describe this image in one sentence."


def pixels(u8):
    """uint8 [N, 512, 512, 3] -> SmolVLM-normalised float [N, 3, 512, 512]."""
    return (u8.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    proc.image_processor.do_image_splitting = False
    tok = proc.tokenizer
    image_id = proc.tokenizer.convert_tokens_to_ids("<image>")

    rows = pq.read_table(args.coco, columns=["image", "answer"]).slice(0, args.n).to_pylist()
    imgs = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").resize((512, 512), Image.BICUBIC) for r in rows]
    caps = [r["answer"][0].strip() for r in rows]

    def chat(cap=None):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        if cap is None:
            return proc.apply_chat_template(m, add_generation_prompt=True)
        m.append({"role": "assistant", "content": [{"type": "text", "text": cap}]})
        return proc.apply_chat_template(m, add_generation_prompt=False)

    enc = [proc(text=[chat(c)], images=[[im]], return_tensors="pt") for c, im in zip(caps, imgs)]
    n_prompt = proc(text=[chat()], images=[[imgs[0]]], return_tensors="pt")["input_ids"].shape[1]
    lens = [e["input_ids"].shape[1] for e in enc]
    L = -(-max(lens) // 16) * 16
    ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(enc), L), -100, dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :lens[i]] = e["input_ids"][0]
        labels[i, n_prompt:lens[i]] = e["input_ids"][0, n_prompt:]
    offsets = {int((row == image_id).nonzero()[0]) for row in ids}
    counts = {int((row == image_id).sum()) for row in ids}
    assert len(offsets) == 1 and counts == {64}, (offsets, counts)
    offset = offsets.pop()
    # Keep the processor's own pixels (it resamples again): back to uint8, exactly.
    px = torch.cat([e["pixel_values"][:, 0] for e in enc])
    u8 = ((px * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    print(f"{len(enc)} pairs, length {L} (max {max(lens)}), prompt {n_prompt} tokens, image tokens at {offset}..{offset + 63}; "
          f"uint8 round trip of the processor's pixels max |diff| {(pixels(u8) - px).abs().max():.1e}")
    torch.save({"images": u8, "input_ids": ids, "labels": labels, "captions": caps,
                "image_offset": offset, "prompt": PROMPT}, os.path.join(args.out, "data.pt"))

    hf = Idefics3ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    ours = SmolVLM(offset).eval()
    sd = {k: v.float() for k, v in hf.state_dict().items()}
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    torch.save({k: v.cpu() for k, v in ours.state_dict().items()}, os.path.join(args.out, "weights.pt"))

    with torch.no_grad():
        # On CPU: the two models then run the same kernels and must agree to the bit. On
        # GPU they pick different attention kernels, and activations near 600 in the
        # decoder's residual stream turn that into ~0.1 on the logits.
        px, tid, lab = pixels(u8[:2]), ids[:2], labels[:2]
        assert next(hf.parameters()).dtype == torch.float32
        ref = hf(pixel_values=px[:, None], input_ids=tid, attention_mask=torch.ones_like(tid)).logits
        got = ours(px, tid)
        print(f"logits vs Hugging Face Idefics3ForConditionalGeneration (CPU, 2 pairs): max |diff| "
              f"{(ref - got).abs().max():.2e} (scale {ref.abs().max():.1f}); loss {lm_loss(ref, lab):.5f} vs {lm_loss(got, lab):.5f}")
        hf.to(args.device)
        for i in range(3):
            g = proc(text=[chat()], images=[[imgs[i]]], return_tensors="pt").to(args.device)
            out = hf.generate(**g, max_new_tokens=30, do_sample=False)
            print(f"  [{i}] reference: {tok.decode(out[0, g['input_ids'].shape[1]:], skip_special_tokens=True).strip()!r}"
                  f"   COCO: {caps[i]!r}")
    print(f"wrote {args.out}/weights.pt, {args.out}/data.pt")


if __name__ == "__main__":
    main()
