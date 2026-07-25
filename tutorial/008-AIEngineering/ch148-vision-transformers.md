# Vision Transformers (ViT)

> An image is a grid of patches. A sentence is a grid of tokens. The same transformer eats both.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## The Problem

Before 2020, computer vision meant convolutions. Every SOTA on ImageNet, COCO, and detection benchmarks used a CNN backbone. Transformers were for language.

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — showed you can drop the convolutions entirely. Slice an image into fixed-size patches, linearly project each patch into an embedding, feed the sequence to a vanilla transformer encoder. At sufficient scale (ImageNet-21k pretraining or bigger), ViT matches or beats ResNet-based models.

ViT was the start of a broader pattern in 2026: one architecture, many modalities. Whisper tokenizes audio. ViT tokenizes images. Action tokens for robotics. Pixel tokens for video. The transformer doesn't care — feed it a sequence and it learns.

## The Concept

### Step 1 — patchify

Split a `H × W × C` image into an `N × (P·P·C)` sequence of flat patches. Typical setup: `224 × 224` image, `16 × 16` patches → 196 patches of 768 values each.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

### Step 2 — linear embedding

A single learned matrix projects each flat patch to `d_model`. Equivalent to a convolution of kernel size `P` and stride `P`. In PyTorch this is literally `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`.

### Step 3 — prepend `[CLS]` token, add positional embeddings

Prepend a learnable `[CLS]` token. Its final hidden state is the image representation used for classification. Add learnable positional embeddings (ViT-original) or sinusoidal 2D (later variants).

### Step 4 — standard transformer encoder

Stack L blocks of `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Identical to BERT. No vision-specific layers.

### Step 5 — head

For classification: take `[CLS]` hidden state → linear → softmax. For DINOv2 or SAM, discard `[CLS]`, use the patch embeddings directly.

### Variants that mattered

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | The original. Fixed patch size, full global attention |
| DeiT | 2021 | Distillation; trainable on ImageNet-1k only |
| Swin | 2021 | Hierarchical with shifted windows. Fixed sub-quadratic cost |
| DINOv2 | 2023 | Self-supervised (no labels). Best general vision features |
| ViT-22B | 2023 | 22B params; scaling laws apply |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder |

## Build It

### Step 1: fake image

A 24 × 24 RGB image as a list of rows of `(R, G, B)` tuples. We use 6×6 patches → 16 patches, 48-d embedding vector each.

### Step 2: patchify

```python
def patchify(image, patch_size):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = []
            for di in range(patch_size):
                for dj in range(patch_size):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order: row-major across the grid. Every ViT uses this ordering.

### Step 3: linear embed

```python
def linear_project(patches, d_model, rng=None):
    in_dim = len(patches[0])
    scale = math.sqrt(2.0 / (in_dim + d_model))
    W = [[rng.gauss(0, scale) for _ in range(d_model)] for _ in range(in_dim)]
    out = []
    for patch in patches:
        row = [0.0] * d_model
        for i, x in enumerate(patch):
            if x == 0.0:
                continue
            for j in range(d_model):
                row[j] += x * W[i][j]
        out.append(row)
    return out
```

Verify output shape is `(N_patches + 1, d_model)` after prepending `[CLS]`.

### Step 4: 2D positional encoding

```python
def pos_2d(H, W, d_model):
    assert d_model % 4 == 0
    half = d_model // 2
    pe = [[[0.0] * d_model for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            for k in range(half // 2):
                theta_row = i / (10000 ** (2 * k / half))
                pe[i][j][2 * k] = math.sin(theta_row)
                pe[i][j][2 * k + 1] = math.cos(theta_row)
            for k in range(half // 2):
                theta_col = j / (10000 ** (2 * k / half))
                pe[i][j][half + 2 * k] = math.sin(theta_col)
                pe[i][j][half + 2 * k + 1] = math.cos(theta_col)
    return pe
```

### Step 5: count parameters for a realistic ViT

```python
def param_count_vit(d_model, n_layers, n_heads, ffn_expansion, num_patches, num_classes):
    per_layer = 4 * d_model ** 2 + 2 * d_model * int(ffn_expansion * d_model) + 4 * d_model
    pos_emb = (num_patches + 1) * d_model
    head = d_model * num_classes
    return per_layer * n_layers + pos_emb + d_model + head + 2 * d_model
```

ViT-Base: ~86M parameters. ViT-Large: ~307M. ViT-Huge: ~632M.

## Use It

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768)
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings are the 2026 default for image features.** Freeze the backbone, train a tiny head.

## Ship It

See `outputs/skill-vit-configurator.md`. The skill picks a ViT variant and patch size for a new vision task.

## Exercises

1. **Easy.** Verify the number of patches equals `(H/P) * (W/P)` and the flat patch dimension equals `P*P*C`.
2. **Medium.** Implement 2D sinusoidal positional embeddings in a tiny PyTorch ViT. Compare vs learnable positional embeddings on CIFAR-10.
3. **Hard.** Build a 3-layer ViT, train on 1,000 MNIST images with 4×4 patches. Add DINOv2-style pretraining — does accuracy improve?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Patch | "The vision-transformer token" | Flat vector of pixel values for a `P × P × C` region |
| Patchify | "Chop + flatten" | Slice image into non-overlapping patches, flatten each to a vector |
| `[CLS]` token | "The image summary" | Prepended learnable token; its final embedding is the image representation |
| Inductive bias | "What the model assumes" | ViT has fewer priors than CNNs; needs more data |
| DINOv2 | "Self-supervised ViT" | Trained without labels using image augmentation + momentum teacher |
| SigLIP | "CLIP's successor" | ViT + text encoder trained with sigmoid contrastive loss |
| Swin | "Windowed ViT" | Hierarchical ViT with local attention + shifted windows |

## Further Reading

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Touvron et al. (2021). Training data-efficient image transformers & distillation](https://arxiv.org/abs/2012.12877)
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/09-vision-transformers)
