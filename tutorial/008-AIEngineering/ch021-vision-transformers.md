# Vision Transformers & Patch Tokens

> Combined lessons (3 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch079): Vision Transformers (ViT)

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

---

## Part 2 (ch148): Vision Transformers (ViT)

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

---

## Part 3 (ch222): Vision Transformers and the Patch-Token Primitive

> Before anything multimodal, an image has to become a sequence of tokens a transformer can eat. The 2020 ViT paper answered this with 16×16 pixel patches, a linear projection, and a position embedding. Five years later every 2026 frontier model (Claude Opus 4.7 at 2576px native, Gemini 3.1 Pro, Qwen3.5-Omni) still begins this way — the encoder changed from ViT to DINOv2 to SigLIP 2, register tokens were added, the positional scheme became 2D-RoPE, but the primitive held. This lesson reads the patch-token pipeline end to end and builds it in stdlib Python so the rest of Phase 12 has a concrete mental model for "visual tokens."

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## Learning Objectives

- Convert an H×W×3 image into a sequence of patch tokens with correct positional encoding.
- Compute sequence length, parameter count, and FLOPs for a ViT of a given (patch size, resolution, hidden dim, depth).
- Name the three upgrades that took ViT from 2020 research to 2026 production: self-supervised pretraining (DINO / MAE), register tokens, and native-resolution packing.
- Pick between CLS pooling, mean pooling, and register tokens for a downstream task.

## The Problem

Transformers operate on sequences of vectors. Text is already a sequence (bytes or tokens). An image is a 2D grid of pixels with three color channels — not a sequence. If you flatten every pixel, a 224×224 RGB image becomes 150,528 tokens, and self-attention at that length is a non-starter (quadratic in sequence length).

Pre-2020 approaches bolted a CNN feature extractor onto the front: ResNet produces a 7×7 feature map of 2048-dim vectors, feed those 49 tokens to a transformer. This works but inherits the CNN's biases (translation equivariance, local receptive fields) and loses the transformer's appetite for scale.

Dosovitskiy et al. (2020) asked the blunt question: what if we skip the CNN? Split the image into fixed-size patches (say 16×16 pixels), linearly project each patch into a vector, add a positional embedding, and feed the sequence to a vanilla transformer. At the time this was heresy — vision without convolutions. With enough data (JFT-300M, then LAION) it beat ResNet on ImageNet and kept improving.

By 2026 the ViT primitive is the unquestioned foundation. Every open-weights VLM's vision tower is some descendant (DINOv2, SigLIP 2, CLIP, EVA, InternViT). The question is no longer "should we use patches?" but "what patch size, what resolution schedule, what pretraining objective, what positional encoding."

## The Concept

### Patches as tokens

Given an image `x` of shape `(H, W, 3)` and a patch size `P`, you carve the image into a grid of `(H/P) × (W/P)` non-overlapping patches. Each patch is a `P × P × 3` cube of pixels. Flatten each cube to a `3 P²` vector. Apply a shared linear projection `W_E` of shape `(3 P², D)` to map each patch into the model's hidden dimension `D`.

For the ViT-B/16 canonical config:
- Resolution 224, patch size 16 → grid 14×14 → 196 patch tokens.
- Each patch is `16 × 16 × 3 = 768` pixel values, projected to `D = 768`.
- Add a learnable `[CLS]` token → sequence length 197.

The patch projection is mathematically identical to a 2D convolution with kernel size `P`, stride `P`, and `D` output channels. That is how production code actually implements it — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. The "linear projection" framing is conceptual; the kernel framing is efficient.

### Positional embeddings

Patches have no inherent order — the transformer sees them as a bag. Early ViTs added a learnable 1D positional embedding (one 768-dim vector per position, 197 of them). Works, but ties the model to the training resolution: at inference you have to interpolate the position table if you change the grid.

Modern vision backbones use 2D-RoPE (Qwen2-VL's M-RoPE, SigLIP 2's default) or factorized 2D positions. 2D-RoPE rotates the query and key vectors based on the patch's (row, column) index, so the model infers relative 2D position from the rotation angle. No position table. The model handles arbitrary grid sizes at inference.

### CLS token, pooled output, and register tokens

What is the image-level representation? Three choices coexist:

1. `[CLS]` token. Prepend a learnable vector to the patch sequence. After all transformer blocks, the CLS token's hidden state is the image representation. Inherited from BERT. Used by original ViT, CLIP.
2. Mean pool. Average the patch tokens' output hidden states. Used by SigLIP, DINOv2, most modern VLMs.
3. Register tokens. Darcet et al. (2023) observed that ViTs trained without an explicit sink token develop high-norm "artifact" patches that hijack self-attention. Adding 4–16 learnable register tokens absorbs this load and improves dense-prediction quality (segmentation, depth). DINOv2 and SigLIP 2 both ship with registers.

The choice matters for downstream tasks. CLS is fine for classification. For VLMs that feed patch tokens into an LLM, you skip pooling entirely — every patch becomes an LLM input token. Registers get discarded before handoff (they are scaffolding, not content).

### Pretraining: supervised, contrastive, masked, self-distilled

The 2020 ViT was pretrained with supervised classification on JFT-300M. Quickly supplanted by:

- CLIP (2021): contrastive image-text on 400M pairs. Lesson 12.02.
- MAE (2021, He et al.): mask 75% of patches, reconstruct pixels. Self-supervised, works on pure images.
- DINO (2021) / DINOv2 (2023): self-distillation with student-teacher, no labels, no captions. The 2023 DINOv2 ViT-g/14 is the strongest purely-visual backbone and the default for "dense features" use cases.
- SigLIP / SigLIP 2 (2023, 2025): CLIP with a sigmoid loss and NaFlex for native aspect ratio. The dominant vision tower in 2026 open VLMs (Qwen, Idefics2, LLaVA-OneVision).

Your choice of pretraining determines what the backbone is good for: CLIP/SigLIP for semantic matching with text, DINOv2 for dense visual features, MAE as a starting point for downstream finetuning.

### Scaling laws

ViT scaling (Zhai et al. 2022) established that a ViT's quality obeys predictable laws in model size, data size, and compute. At fixed compute:
- Bigger model + more data → better quality.
- Patch size is a lever on sequence length vs fidelity. Patch 14 (typical for DINOv2/SigLIP SO400m) gives more tokens per image than patch 16; better for OCR and dense tasks, worse for speed.
- Resolution is the other big lever. Going from 224 to 384 to 512 almost always helps, at quadratic cost in FLOPs.

ViT-g/14 (1B params, patch 14, resolution 224 → 256 tokens) and SigLIP SO400m/14 (400M params, patch 14) are the two workhorse encoders for 2026 open VLMs.

### Parameter count for a ViT

The full calculation lives in `code/main.py`. For ViT-B/16 at 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768² (QKVO) + 2 * 4 * 768² (MLP) + 2 * 2*768 (LN)
             = 12 * 768² + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Ball-park every ViT this way before you load the checkpoint. The backbone size sets your VRAM floor in any downstream VLM.

### 2026 production config

The encoder most open VLMs ship with in 2026 is SigLIP 2 SO400m/14 at native resolution (NaFlex). It has:
- 400M parameters.
- Patch size 14, default resolution 384 → 729 patch tokens per image.
- Mean pool for image-level tasks; all 729 patches flow into the LLM for VQA.
- 4 register tokens, discarded before LLM handoff.
- 2D-RoPE with image-level scaling for native aspect ratio.

Every decision in that config traces back to a paper you can read.

```figure
image-patch-tokens
```

## Use It

`code/main.py` is a patch tokenizer and geometry calculator. It takes (image H, W, patch P, hidden D, depth L) and reports:

- Grid shape and sequence length after patching.
- Token sequence for a synthetic 8×8 pixel toy image (walk through the flatten + project path).
- Parameter count broken down by patch embed, position embed, transformer blocks, and head.
- FLOPs per forward pass at the target resolution.
- A comparison table across ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Run it. Match the parameter counts to the published numbers. Play with patch size and resolution to feel the token-count cost.

## Ship It

This lesson produces `outputs/skill-patch-geometry-reader.md`. Given a ViT config (patch size, resolution, hidden dim, depth), it produces a token-count, parameter-count, and VRAM estimate with justifications. Use this skill whenever you pick a vision backbone for a VLM — it prevents "the tokens exploded and my LLM context filled up" surprises.

## Exercises

1. Compute the patch-token sequence length for Qwen2.5-VL at native 1280×720 input with patch size 14. How does that compare to a CLS-only representation?

2. A 1080p frame (1920×1080) at patch 14 produces how many tokens? At 30 FPS over a 5-minute video, how many total visual tokens? Which cost saves you most: pooling, frame sampling, or token merging?

3. Implement mean pooling over patch tokens in pure Python. Verify that mean-pool over 196 tokens of a DINOv2 output matches what the model's `forward` returns when you ask for a pooled embedding.

4. Read Section 3 of "Vision Transformers Need Registers" (arXiv:2309.16588). Describe in two sentences what artifact the registers absorb and why it matters for downstream dense prediction.

5. Modify `code/main.py` to support patch-n'-pack: given a list of images of different resolutions, produce a single packed sequence and the block-diagonal attention mask. Verify against Lesson 12.06 when you reach it.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Patch | "16×16 pixel square" | A fixed-size non-overlapping region of the input image; becomes one token |
| Patch embedding | "Linear projection" | A shared learned matrix (or Conv2d with stride=P) mapping flattened patch pixels to D-dim vectors |
| CLS token | "Class token" | Prepended learnable vector whose final hidden state represents the whole image; optional in 2026 |
| Register token | "Sink token" | Extra learnable tokens that absorb the high-norm attention artifacts ViTs develop during pretraining |
| Position embedding | "Positional info" | Per-position vector or rotation making the sequence-order-aware; 2D-RoPE is the modern default |
| Grid | "Patch grid" | The (H/P) × (W/P) 2D array of patches for a given resolution and patch size |
| NaFlex | "Native flexible resolution" | SigLIP 2 feature: single model serves multiple aspect ratios and resolutions without retraining |
| Backbone | "Vision tower" | The pretrained image encoder whose patch-token outputs feed the LLM in a VLM |
| Pooling | "Image-level summary" | Strategy to turn patch tokens into one vector: CLS, mean, attention pool, or register-based |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14 produces more tokens per image, better fidelity for OCR, slower; patch 16 is the classic default |

## Further Reading

- [Dosovitskiy et al. — An Image is Worth 16×16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — original ViT.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, self-supervised pretraining.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — self-distillation at scale, no labels.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens and artifact analysis.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — the 2026 default vision tower.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — empirical scaling laws.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/01-vision-transformer-patch-tokens)
