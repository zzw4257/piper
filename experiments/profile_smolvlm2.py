"""F97: stage costs of SmolVLM2-2.2B on one GPU, for the pipeline model and to see where the work is.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=examples:. python experiments/profile_smolvlm2.py --data DIR

Vision tower + connector on N 384-pixel tiles; decoder (embedding, 24 layers, head) on b examples
of L tokens with T*81 image tokens scattered in. Forward and forward+backward, fp32, minimum of
interleaved rounds (log F94).
"""
import argparse, json, os, sys, time
sys.argv = [x for x in sys.argv if not x.startswith("--temp-dir") and not x.startswith("/var/tmp/ziweizho-ray")]
import torch
sys.path[:0] = ["examples", ".", "experiments"]
from models.smolvlm import SmolVLM, CFG_SMOLVLM2
from profile_stages import fwd_bwd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--rounds", type=int, default=2)
    a, _ = ap.parse_known_args()
    m = SmolVLM(None, config=CFG_SMOLVLM2).cuda()
    m.load_state_dict(torch.load(os.path.join(a.data, "weights.pt")))
    vm, tm = m.model.vision_model, m.model.text_model
    def vis(px):
        x = vm.embeddings(px)
        for layer in vm.encoder.layers:
            x = layer(x)
        return m.model.connector(vm.post_layernorm(x))
    def dec(ids, img, idx):
        txt = tm.embed_tokens(ids)
        h = txt.scatter(1, idx.unsqueeze(-1).expand(-1, -1, img.shape[-1]), img.reshape(ids.shape[0], -1, img.shape[-1]))
        cos, sin = m._rope(h.shape[1], h.device, h.dtype)
        for layer in tm.layers:
            h = layer(h, cos, sin)
        return m.lm_head(tm.norm(h))
    buckets = {}
    for T in (9, 13, 17):
        d = torch.load(os.path.join(a.data, f"T{T}", "data.pt"))
        buckets[T] = d
    out = {"vision": {}, "decoder": {}}
    for _ in range(a.rounds):
        for n in [int(x) for x in os.environ.get('VIS_TILES', '9,13,17,34,51,68').split(',')]:
            d = buckets[17]
            px = ((d["images"][:n].permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).cuda()
            v = [round(x, 2) for x in fwd_bwd(vis, [px], robust=True)]
            old = out["vision"].get(n); out["vision"][n] = v if old is None or sum(v) < sum(old) else old
            torch.cuda.empty_cache()
        for T in (9, 13, 17):
            for b in [int(x) for x in os.environ.get('DEC_B', '1,2,4').split(',')]:
                d = buckets[T]
                ids, idx = d["input_ids"][:b].cuda(), d["img_index"][:b].cuda()
                with torch.no_grad():
                    img = vis(((d["images"][:b * T].permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).cuda())
                v = [round(x, 2) for x in fwd_bwd(lambda i, im, j: dec(i, im, j), [ids, img, idx], robust=True)]
                key = f"T{T}_b{b}"; old = out["decoder"].get(key); out["decoder"][key] = v if old is None or sum(v) < sum(old) else old
                torch.cuda.empty_cache()
    print(json.dumps(out))
    for T in (9, 13, 17):
        if T not in out["vision"] or f"T{T}_b1" not in out["decoder"]:
            continue
        vv = sum(out["vision"][T]); dd = sum(out["decoder"][f"T{T}_b1"])
        print(f"T={T}: vision {vv:7.1f} ms, decoder {dd:7.1f} ms per example (fwd+bwd); vision/decoder {vv / dd:4.2f}; "
              f"sequence {buckets[T]['input_ids'].shape[1]} tokens")
    print("PASS")


if __name__ == "__main__":
    main()
