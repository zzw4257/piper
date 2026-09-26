"""SmolVLM-256M (Idefics3: SigLIP vision tower, pixel-shuffle connector, Llama decoder),
written for Piper: an image branch and a text branch that meet in the decoder.

    pixel_values -> [vision tower + connector] --\
                                                   +--> [decoder layers ...] -> logits
    input_ids    -> [token embedding]          --/

Parameter names match Hugging Face's ``Idefics3ForConditionalGeneration``, so the
released ``HuggingFaceTB/SmolVLM-256M-Instruct`` weights load by name
(``experiments/prepare_smolvlm.py`` checks the logits against the reference).

Simplifications that keep the math identical for the inputs used here: one
512x512 tile per image (no image splitting), so every patch is valid and the
position ids are 0..1023; the prompt template is fixed, so the 64 image tokens sit
at one known offset and the merge is a concatenation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate

CFG = dict(v_width=768, v_layers=12, v_heads=12, v_mlp=3072, patch=16, image=512, scale=4,
           t_width=576, t_layers=30, t_heads=9, t_kv_heads=3, t_mlp=1536, vocab=49280,
           rope_theta=100000.0, rms_eps=1e-5, ln_eps=1e-6)


class VisionAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, w = x.shape
        split = lambda t: t.view(b, n, self.heads, w // self.heads).transpose(1, 2)  # noqa: E731
        o = F.scaled_dot_product_attention(split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)))
        return self.out_proj(o.transpose(1, 2).reshape(b, n, w))


class VisionMLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VisionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = VisionAttention(c["v_width"], c["v_heads"])
        self.layer_norm1 = nn.LayerNorm(c["v_width"], eps=c["ln_eps"])
        self.mlp = VisionMLP(c["v_width"], c["v_mlp"])
        self.layer_norm2 = nn.LayerNorm(c["v_width"], eps=c["ln_eps"])

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList(VisionLayer(c) for _ in range(c["v_layers"]))


class VisionEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, c["v_width"], c["patch"], stride=c["patch"])
        self.position_embedding = nn.Embedding((c["image"] // c["patch"]) ** 2, c["v_width"])

    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return x + self.position_embedding.weight  # every patch valid: position ids 0..n-1


class VisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = VisionEmbeddings(c)
        self.encoder = VisionEncoder(c)
        self.post_layernorm = nn.LayerNorm(c["v_width"], eps=c["ln_eps"])

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.post_layernorm(x)


class ModalityProjection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Linear(c["v_width"] * c["scale"] ** 2, c["t_width"], bias=False)


class Connector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c["scale"]
        self.modality_projection = ModalityProjection(c)

    def forward(self, x):
        # Idefics3Connector.pixel_shuffle, verbatim in effect
        b, seq, d = x.shape
        h = w = int(seq ** 0.5)
        s = self.scale
        x = x.view(b, h, w, d).view(b, h, w // s, d * s).permute(0, 2, 1, 3)
        x = x.reshape(b, w // s, h // s, d * s * s).permute(0, 2, 1, 3).reshape(b, seq // (s * s), d * s * s)
        return self.modality_projection.proj(x)


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class TextAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads = c["t_heads"], c["t_kv_heads"]
        self.head_dim = c["t_width"] // c["t_heads"]
        self.q_proj = nn.Linear(c["t_width"], self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(c["t_width"], self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c["t_width"], self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, c["t_width"], bias=False)

    def forward(self, x, cos, sin):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(b, n, -1))


class TextMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c["t_width"], c["t_mlp"], bias=False)
        self.up_proj = nn.Linear(c["t_width"], c["t_mlp"], bias=False)
        self.down_proj = nn.Linear(c["t_mlp"], c["t_width"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = TextAttention(c)
        self.mlp = TextMLP(c)
        self.input_layernorm = RMSNorm(c["t_width"], c["rms_eps"])
        self.post_attention_layernorm = RMSNorm(c["t_width"], c["rms_eps"])

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab"], c["t_width"])
        self.layers = nn.ModuleList(TextLayer(c) for _ in range(c["t_layers"]))
        self.norm = RMSNorm(c["t_width"], c["rms_eps"])


class Idefics3Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.vision_model = VisionModel(c)
        self.connector = Connector(c)
        self.text_model = TextModel(c)


class SmolVLM(nn.Module):
    """``image_offset``: index of the first image token in every prompt.
    ``dec_stages`` / ``vis_stages``: how many PP regions the decoder layers and the
    vision layers are split into. Regions in order: vision, text embedding, decoder."""

    def __init__(self, image_offset: int, dec_stages: int = 1, vis_stages: int = 1, config: dict | None = None):
        super().__init__()
        self.c = c = dict(CFG, **(config or {}))
        self.image_offset, self.dec_stages, self.vis_stages = image_offset, dec_stages, vis_stages
        self.n_img = (c["image"] // c["patch"]) ** 2 // c["scale"] ** 2
        self.model = Idefics3Model(c)
        self.lm_head = nn.Linear(c["t_width"], c["vocab"], bias=False)

    def _rope(self, n, device, dtype):
        hd = self.c["t_width"] // self.c["t_heads"]
        inv = 1.0 / (self.c["rope_theta"] ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        f = torch.outer(torch.arange(n, device=device, dtype=torch.float32), inv)
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

    def forward(self, pixel_values, input_ids):
        vm = self.model.vision_model
        vper = -(-len(vm.encoder.layers) // self.vis_stages)
        for s in range(self.vis_stages):
            with annotate("PP"):
                img = vm.embeddings(pixel_values) if s == 0 else img
                for layer in vm.encoder.layers[s * vper:(s + 1) * vper]:
                    img = layer(img)
                if s == self.vis_stages - 1:
                    img = self.model.connector(vm.post_layernorm(img))
        with annotate("PP"):
            txt = self.model.text_model.embed_tokens(input_ids)
        layers = self.model.text_model.layers
        per = -(-len(layers) // self.dec_stages)
        for s in range(self.dec_stages):
            with annotate("PP"):
                if s == 0:
                    o = self.image_offset
                    h = torch.cat([txt[:, :o], img, txt[:, o + self.n_img:]], dim=1)
                cos, sin = self._rope(h.shape[1], h.device, h.dtype)
                for layer in layers[s * per:(s + 1) * per]:
                    h = layer(h, cos, sin)
                if s == self.dec_stages - 1:
                    h = self.lm_head(self.model.text_model.norm(h))
        return h


def lm_loss(logits, labels):
    """Next-token cross entropy; labels are -100 outside the caption."""
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1).long(), ignore_index=-100)
