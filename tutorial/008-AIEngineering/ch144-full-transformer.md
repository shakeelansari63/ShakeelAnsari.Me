# The Full Transformer — Encoder + Decoder

> Attention is the star. Everything else — residuals, normalization, feed-forward, cross-attention — is the scaffolding that lets you stack it deep.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## The Problem

A single attention layer is a feature extractor, not a model. One matmul per layer is not enough capacity for language. You need depth — and depth breaks without the right plumbing.

The 2017 Vaswani paper packaged six design decisions that turned one attention layer into a stackable block. Every transformer since — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — inherits the same skeleton. In 2026 the blocks have been refined (RMSNorm, SwiGLU, pre-norm, RoPE) but the skeleton is identical.

## The Concept

### The six pieces

1. **Embedding + positional signal.** Tokens → vectors. Position injected via RoPE (modern) or sinusoidal (classic).
2. **Self-attention.** Every position attends to every other. Masked in decoders.
3. **Feed-forward network (FFN).** Position-wise two-layer MLP: `W_2 · activation(W_1 · x)`. Expansion ratio 4× by default.
4. **Residual connection.** `x + sublayer(x)`. Without this, gradients vanish past ~6 layers.
5. **Layer normalization.** `LayerNorm` or `RMSNorm` (modern). Stabilizes the residual stream.
6. **Cross-attention (decoder only).** Queries come from the decoder, keys and values from the encoder output.

### Encoder block (used by BERT, T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder is bidirectional. No masking. All positions see all positions.

### Decoder block (used by GPT, T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder has three sublayers per block. The middle one — cross-attention — is the only place information flows from encoder to decoder.

### Pre-norm vs post-norm

Original paper: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm lost favor around 2019 — it is harder to train deeply without careful warmup. Pre-norm (`LN` *before* sublayer) is the 2026 default.

### The 2026 modernized block

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm drops the mean-centering of LayerNorm. SwiGLU (`Swish(W1 x) ⊙ W3 x`) consistently outperforms ReLU/GELU FFN.

### Parameter count

For one block with `d_model = d` and FFN expansion `r`:

- MHA: `4 · d²` (Q, K, V, O projections)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: negligible

At `d = 4096, r = 2.6, layers = 32` (roughly Llama 3 8B): `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`.

## Build It

### Step 1: the building blocks

Using a tiny `Matrix` class:

- `layer_norm(x, eps=1e-5)` — subtract mean, divide by std
- `rms_norm(x, eps=1e-6)` — divide by RMS, no mean subtraction
- `gelu(x)` and `silu(x) * W3 x` (SwiGLU)
- `ffn_swiglu(x, W1, W2, W3)` and `ffn_relu(x, W1, W2)`

```python
def silu(x):
    return x / (1.0 + math.exp(-x))

def ffn_swiglu(X, W1, W2, W3):
    h1 = matmul(X, W1)
    h3 = matmul(X, W3)
    gated = Matrix(h1.rows, h1.cols)
    for i in range(len(h1.data)):
        gated.data[i] = silu(h1.data[i]) * h3.data[i]
    return matmul(gated, W2)
```

### Step 2: wire a 2-layer encoder and decoder

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: encoder_block and decoder_block

```python
def encoder_block(x, p):
    h = rms_norm(x)
    a = multi_head_attention(h, p.Wq, p.Wk, p.Wv, p.Wo, p.n_heads)
    x = add(x, a)
    h = rms_norm(x)
    f = ffn_swiglu(h, p.W1, p.W2, p.W3)
    return add(x, f)

def decoder_block(x, enc_out, p):
    h = rms_norm(x)
    a = multi_head_attention(h, p.Wq, p.Wk, p.Wv, p.Wo, p.n_heads, causal=True)
    x = add(x, a)
    h = rms_norm(x)
    a = multi_head_attention(h, p.Wq_x, p.Wk_x, p.Wv_x, p.Wo_x, p.n_heads, kv_source=enc_out)
    x = add(x, a)
    h = rms_norm(x)
    f = ffn_swiglu(h, p.W1, p.W2, p.W3)
    return add(x, f)
```

### Step 4: swap in RMSNorm + SwiGLU

Replace LayerNorm and ReLU-FFN with RMSNorm and SwiGLU. Confirm shapes still match.

## Use It

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output | Encoder-decoder | T5, BART, Whisper |

Decoder-only won language because it scales cleanest. Encoder-decoder is still best when the input has a clear "source sequence" identity.

## Ship It

See `outputs/skill-transformer-block-reviewer.md`. The skill reviews a transformer block implementation against 2026 defaults.

## Exercises

1. **Easy.** Count the parameters in your encoder_block at `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`.
2. **Medium.** Switch from post-norm to pre-norm. Initialize both and measure activation norm after 12 stacked layers.
3. **Hard.** Implement a 4-layer encoder-decoder on a toy copy task. Swap in RMSNorm + SwiGLU + RoPE — does loss drop?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residuals |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))` |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same stability |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2` |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers |

## Further Reading

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/05-full-transformer)
