# Vision Encoder Patches

> A vision model that reads pixels needs a tokenizer for pixels. Patch embedding is that tokenizer.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37
**Time:** ~90 minutes

## Learning Objectives

- Tokenize an image into a fixed-length sequence of patch embeddings.
- Implement a Conv2d-based patch projection.
- Build a deterministic 2D sinusoidal position embedding.
- Verify patch count, embedding shape, and Conv2d/unfold equivalence.

## The Problem

A 224x224 RGB image is 150,528 tokens if read pixel-by-pixel. Patch embedding with 16x16 patches produces 196 tokens. Each patch is flattened and linearly projected to the model's hidden dimension.

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### The Conv2d trick

A `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` gives the same numerical result as unfold-then-linear.

### Position embeddings

Half the embedding dimension encodes row position with sin/cos; the other half encodes column position. Deterministic and interpolates cleanly to new resolutions.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection | (hidden, 3, patch, patch) | 3 * P * P * hidden + hidden |
| Position embedding | (num_patches, hidden) | 0 (fixed) |
| CLS token | (1, hidden) | hidden |

## Build It

`code/main.py` implements: `PatchEmbed`, `sinusoidal_2d`, `VisionFrontEnd`, `synthesize_image`.

## Key Terms

| Term | What it means |
|------|---------------|
| Patch | Square sub-region of the image, typically 14x14 or 16x16 |
| Patch embedding | Linear projection of one flattened patch to the hidden dim |
| Sequence length | Number of tokens after patch tokenization, usually plus CLS |
| Sinusoidal position | Fixed sin/cos signal encoding 2D grid coordinates |
| CLS token | Learned vector prepended as the pooling head |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/58-vision-encoder-patches)
