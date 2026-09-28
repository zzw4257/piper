"""SmolVLM2-2.2B weights and DocVQA/ChartQA (the_cauldron) examples for examples/models/smolvlm.py (log F97).

    PYTHONPATH=examples:. python experiments/prepare_smolvlm2.py --model DIR --parquet A.parquet B.parquet --out DIR

The processor splits each image into 384-pixel tiles plus a global view (longest edge 1536), so
the number of tiles, and of image tokens (81 each), depends on the image. Piper traces fixed
shapes, so examples are bucketed by tile count; each bucket gets its own data.pt under
OUT/T<tiles>/ with the tiles as uint8, input_ids, labels on the answer only, and img_index, the
positions of the image tokens. Checks our logits against Hugging Face's
SmolVLMForConditionalGeneration on CPU.
"""
import argparse, collections, io, os, sys
import torch
sys.path[:0] = ["examples", "."]
from models.smolvlm import CFG_SMOLVLM2, SmolVLM, lm_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--scan", type=int, default=400)
    ap.add_argument("--per-bucket", type=int, default=96); ap.add_argument("--buckets", type=int, default=3)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    image_id = tok.convert_tokens_to_ids("<image>")
    n_img = (384 // 14) ** 2 // 9

    def chat(q, ans=None):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        if ans is None:
            return proc.apply_chat_template(m, add_generation_prompt=True)
        m.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})
        return proc.apply_chat_template(m, add_generation_prompt=False)

    ex = []
    for path in a.parquet:
        src = os.path.basename(os.path.dirname(path))
        for r in pq.read_table(path, columns=["images", "texts"]).slice(0, a.scan).to_pylist():
            im = Image.open(io.BytesIO(r["images"][0]["bytes"])).convert("RGB")
            q, ans = r["texts"][0]["user"], r["texts"][0]["assistant"]
            e = proc(text=[chat(q, ans)], images=[[im]], return_tensors="pt")
            n_prompt = proc(text=[chat(q)], images=[[im]], return_tensors="pt")["input_ids"].shape[1]
            T = e["pixel_values"].shape[1]
            ids = e["input_ids"][0]
            assert int((ids == image_id).sum()) == T * n_img, (T, int((ids == image_id).sum()))
            ex.append(dict(src=src, size=im.size, T=T, ids=ids, n_prompt=n_prompt, px=e["pixel_values"][0], q=q, a=ans))
    hist = collections.Counter(e["T"] for e in ex)
    by_src = {s: collections.Counter(e["T"] for e in ex if e["src"] == s) for s in {e["src"] for e in ex}}
    print("tiles per example:", dict(sorted(hist.items())), "| by source:", {s: dict(sorted(c.items())) for s, c in by_src.items()})
    print("tokens per example: min", min(len(e["ids"]) for e in ex), "max", max(len(e["ids"]) for e in ex))
    os.makedirs(a.out, exist_ok=True)
    for T, cnt in hist.most_common(a.buckets):
        sel = [e for e in ex if e["T"] == T][:a.per_bucket]
        L = -(-max(len(e["ids"]) for e in sel) // 16) * 16
        ids = torch.full((len(sel), L), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(sel), L), -100, dtype=torch.long)
        for i, e in enumerate(sel):
            n = len(e["ids"]); ids[i, :n] = e["ids"]; labels[i, e["n_prompt"]:n] = e["ids"][e["n_prompt"]:]
        img_index = torch.stack([(row == image_id).nonzero()[:, 0] for row in ids])
        px = torch.cat([e["px"] for e in sel])
        u8 = ((px * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        back = (u8.permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
        d = os.path.join(a.out, f"T{T}"); os.makedirs(d, exist_ok=True)
        torch.save({"images": u8, "tiles": T, "input_ids": ids, "labels": labels, "img_index": img_index,
                    "config": CFG_SMOLVLM2, "image_offset": None, "sources": [e["src"] for e in sel],
                    "questions": [e["q"] for e in sel], "answers": [e["a"] for e in sel]}, os.path.join(d, "data.pt"))
        if not os.path.exists(os.path.join(d, "weights.pt")):
            os.symlink(os.path.join(a.out, "weights.pt"), os.path.join(d, "weights.pt"))
        print(f"bucket T={T}: {len(sel)} examples ({collections.Counter(e['src'] for e in sel)}), length {L}, "
              f"{T * n_img} image tokens each; uint8 round trip max |diff| {(back - px).abs().max():.1e}")

    hf = SmolVLMForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).eval()
    ours = SmolVLM(None, config=CFG_SMOLVLM2).eval()
    missing, unexpected = ours.load_state_dict({k: v.float() for k, v in hf.state_dict().items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    torch.save({k: v.cpu() for k, v in ours.state_dict().items()}, os.path.join(a.out, "weights.pt"))
    e = min(ex, key=lambda e: e["T"])
    with torch.no_grad():
        ids = e["ids"][None]; lab = torch.full_like(ids, -100); lab[0, e["n_prompt"]:] = ids[0, e["n_prompt"]:]
        ref = hf(pixel_values=e["px"][None], input_ids=ids, attention_mask=torch.ones_like(ids),
                 pixel_attention_mask=torch.ones(1, e["T"], 384, 384, dtype=torch.bool)).logits
        got = ours(e["px"], ids, (ids[0] == image_id).nonzero()[:, 0][None])
    print(f"logits vs Hugging Face SmolVLMForConditionalGeneration (CPU, one example, {e['T']} tiles, {ids.shape[1]} tokens): "
          f"max |diff| {(ref - got).abs().max():.2e} (scale {ref.abs().max():.1f}); loss {lm_loss(ref, lab):.5f} vs {lm_loss(got, lab):.5f}")
    print("PREP_DONE")


if __name__ == "__main__":
    main()
