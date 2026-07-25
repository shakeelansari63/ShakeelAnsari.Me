# Vision Transformer Encoder

> Patches alone do not see. A 12-layer pre-LN transformer with 12 attention heads turns patch tokens into contextual tokens.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37
**Time:** ~90 minutes

## Learning Objectives

- Implement a pre-LN transformer block with multi-head self-attention and feed-forward sub-layer.
- Stack 12 blocks with 12 heads to form a ViT-Base encoder.
- Wire the patch front end from lesson 58 into the encoder.
- Verify that the CLS token aggregates information from every patch.

## The Concept

```mermaid
flowchart TB
  Input[token sequence B x 197 x 768] --> B1[Block 1]
  B1 --> B2[Block 2]
  B2 --> Dots[...]
  Dots --> B12[Block 12]
  B12 --> LN[Final LayerNorm]
  LN --> Out[contextual tokens B x 197 x 768]
```

```mermaid
flowchart LR
  Token[token x] --> N1[LayerNorm]
  N1 --> Attn[multi-head self-attention]
  Attn --> R1[+ residual]
  R1 --> N2[LayerNorm]
  N2 --> MLP[FFN: linear -> GELU -> linear]
  MLP --> R2[+ residual]
  R2 --> Out[token x']
```

### Parameter count at ViT-Base

| Component | Parameters |
|-----------|------------|
| qkv projection per block | 1.77M |
| output projection per block | 590K |
| FFN per block (4x) | 4.72M |
| Total | ~86M |

## Build It

`code/main.py` implements: `MultiHeadSelfAttention`, `FeedForward`, `Block`, `ViT`, `VisionEncoder`.

## Key Terms

| Term | What it means |
|------|---------------|
| Pre-LN | LayerNorm before each sub-layer instead of after |
| Self-attention | Each token attends to every other token |
| Multi-head | Hidden dim split across H independent heads |
| FFN expansion | Feed-forward widens to 4 * hidden before contracting |
| CLS pooling | First token's final state as image summary |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/59-vit-transformer)
