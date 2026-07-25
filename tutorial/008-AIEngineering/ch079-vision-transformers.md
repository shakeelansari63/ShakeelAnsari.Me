# Vision Transformers (ViT)

> Cut the image into patches, treat each patch as a word, run a standard transformer. Don't look back.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 02 (Self-Attention), Phase 4 Lesson 04 (Image Classification)
**Time:** ~45 minutes

## Learning Objectives

- Implement patch embedding, CLS token, positional embedding, and transformer blocks
- Explain why ViT needed massive pretraining data until DeiT and MAE
- Compare ViT, Swin, and ConvNeXt
- Fine-tune a pretrained ViT with `timm`

## The Problem

CNNs had strong inductive bias. Dosovitskiy et al. (2020) showed a plain transformer on image patches beats CNNs at scale.

## The Concept

### The pipeline

```mermaid
flowchart LR
    IMG["(3, 224, 224)"] --> PATCH["Patch embedding<br/>conv 16x16 s=16"] --> FLAT["(196, 768) tokens"]
    FLAT --> CAT["Prepend [CLS]"] --> POS["Add pos embed"]
    POS --> ENC["N transformer blocks"]
    ENC --> CLS["Take [CLS]"] --> HEAD["MLP classifier"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

### Patch embedding

`Conv2d(3, 768, kernel_size=16, stride=16)` patchifies and projects in one step.

### Pre-LN

```
x = x + sublayer(LN(x))
```

Stable without warmup. Every modern LLM uses this.

### MAE pretraining

Mask 75% of patches, encode visible 25%, decode to reconstruct masked pixels. After pretraining, discard decoder.

## Build It

### Step 1: Patch embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

### Step 2: Transformer block

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

### Step 3: The ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

### Step 4: Sanity check

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"probs: {logits.softmax(-1)}")
```

## Use It

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

## Ship It

- `outputs/prompt-vit-vs-cnn-picker.md`
- `outputs/skill-vit-patch-and-pos-embed-inspector.md`

## Exercises

1. **(Easy)** Print intermediate shapes for a forward pass.
2. **(Medium)** Fine-tune ViT-S/16 vs ResNet-18 on synthetic-CIFAR.
3. **(Hard)** Implement MAE pretraining for the tiny ViT.

## Key Terms

## Further Reading

- [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [DeiT (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877)
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377)
- [timm docs](https://huggingface.co/docs/timm)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/14-vision-transformers)
