"""CLIP (ViT-B/32), written for Piper: an image tower and a text tower that share
nothing until the similarity head. The multimodal shape issue #16 asks about.

Parameter names match Hugging Face's ``CLIPModel``, so the released
``openai/clip-vit-base-patch32`` weights load by name
(``experiments/prepare_clip.py`` checks the logits against the reference).

    pixel_values -> [vision tower] --\
                                       +--> [head: cosine logits] -> logits_per_image
    input_ids    -> [text tower]   --/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.piper import annotate

VIT_B32 = dict(embed=512, v_width=768, v_layers=12, v_heads=12, patch=32, image=224,
               t_width=512, t_layers=12, t_heads=8, vocab=49408, context=77)


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self, width, heads, causal):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, w = x.shape
        split = lambda t: t.view(b, n, self.heads, w // self.heads).transpose(1, 2)  # noqa: E731
        o = F.scaled_dot_product_attention(split(self.q_proj(x)), split(self.k_proj(x)),
                                           split(self.v_proj(x)), is_causal=self.causal)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, w))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, width, heads, causal):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(width)
        self.self_attn = Attention(width, heads, causal)
        self.layer_norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, width, layers, heads, causal):
        super().__init__()
        self.layers = nn.ModuleList(Layer(width, heads, causal) for _ in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionEmbeddings(nn.Module):
    def __init__(self, width, patch, image):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.patch_embedding = nn.Conv2d(3, width, patch, stride=patch, bias=False)
        self.position_embedding = nn.Embedding((image // patch) ** 2 + 1, width)

    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(x.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1) + self.position_embedding.weight


class VisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = VisionEmbeddings(c["v_width"], c["patch"], c["image"])
        self.pre_layrnorm = nn.LayerNorm(c["v_width"])  # sic: Hugging Face's name
        self.encoder = Encoder(c["v_width"], c["v_layers"], c["v_heads"], causal=False)
        self.post_layernorm = nn.LayerNorm(c["v_width"])

    def forward(self, pixel_values):
        x = self.encoder(self.pre_layrnorm(self.embeddings(pixel_values)))
        return self.post_layernorm(x[:, 0])


class TextEmbeddings(nn.Module):
    def __init__(self, width, vocab, context):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.position_embedding = nn.Embedding(context, width)

    def forward(self, input_ids):
        return self.token_embedding(input_ids) + self.position_embedding.weight[: input_ids.shape[1]]


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = TextEmbeddings(c["t_width"], c["vocab"], c["context"])
        self.encoder = Encoder(c["t_width"], c["t_layers"], c["t_heads"], causal=True)
        self.final_layer_norm = nn.LayerNorm(c["t_width"])

    def forward(self, input_ids):
        x = self.final_layer_norm(self.encoder(self.embeddings(input_ids)))
        # the end-of-text token has the largest id in CLIP's vocabulary
        return x[torch.arange(x.shape[0], device=x.device), input_ids.argmax(dim=-1)]


class CLIP(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        c = dict(VIT_B32, **(config or {}))
        self.vision_model = VisionModel(c)
        self.text_model = TextModel(c)
        self.visual_projection = nn.Linear(c["v_width"], c["embed"], bias=False)
        self.text_projection = nn.Linear(c["t_width"], c["embed"], bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(self, pixel_values, input_ids):
        with annotate("PP"):
            image = self.visual_projection(self.vision_model(pixel_values))
        with annotate("PP"):
            text = self.text_projection(self.text_model(input_ids))
        with annotate("PP"):
            image = image / image.norm(dim=-1, keepdim=True)
            text = text / text.norm(dim=-1, keepdim=True)
            logits_per_image = self.logit_scale.exp() * image @ text.t()
        return logits_per_image


def clip_loss(logits_per_image, labels):
    """Symmetric contrastive loss; labels[i] is the caption index for image i."""
    labels = labels.long()
    return (F.cross_entropy(logits_per_image.float(), labels)
            + F.cross_entropy(logits_per_image.t().float(), labels)) / 2
