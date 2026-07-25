# Positional Encoding — Sinusoidal, RoPE, ALiBi

> Attention is permutation-invariant. "The cat sat on the mat" and "mat the on sat cat the" produce the same output without positional signal. Three algorithms fix it — each with a different bet on what "position" means.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## The Problem

Scaled dot-product attention is order-blind. The attention matrix `softmax(Q K^T / √d) V` is computed from pairwise similarities. Shuffle the rows of `X`, get the rows of the output shuffled the same way. Nothing inside attention cares about position.

The fix is to inject position into the embeddings somehow. Three eras of answers:

1. **Absolute sinusoidal** (Vaswani 2017). Add `sin/cos` of position to the embedding. Simple, learnable-free, extrapolates poorly beyond trained lengths.
2. **RoPE — Rotary Position Embeddings** (Su 2021). Rotate Q and K vectors by an angle proportional to position. Encodes *relative* position directly in the dot product. Dominant in 2026.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Skip embeddings entirely; add a per-head linear penalty to attention scores based on distance. Excellent length extrapolation.

As of 2026, essentially every frontier open model uses RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi.

## The Concept

### Absolute sinusoidal

Pre-compute a fixed matrix `PE` of shape `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Then `X' = X + PE[:N]` before attention. Each dimension is a sinusoid at a different frequency. Fails beyond `max_len`: nothing told the model what happens at position 2048 when it only saw positions 0–2047.

### RoPE

Rotate the Q and K vectors (not embeddings). For a pair of dimensions `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Apply the same rotation to keys with position `pos_k`. The dot product `q'_m · k'_n` becomes a function of `(m - n)` alone. That is: **the attention score depends only on the relative distance**, even though the rotation was keyed off absolute positions.

Extending RoPE: `base` can be scaled (NTK-aware, YaRN, LongRoPE) to extrapolate to longer contexts without retraining. Llama 3 extended from 8K to 128K context this way.

### ALiBi

Skip the embedding trick. Bias the attention scores directly:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Where `m_h` is a head-specific slope (e.g. `1 / 2^(8·h/H)`). Closer tokens get boosted; far tokens get penalized. No training-time cost.

### What to pick in 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3 |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

## Build It

### Step 1: sinusoidal encoding

```python
def sinusoidal_pe(n, d, base=10000.0):
    pe = [[0.0] * d for _ in range(n)]
    for pos in range(n):
        for i in range(d // 2):
            theta = pos / (base ** (2 * i / d))
            pe[pos][2 * i] = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Add this to the embedding matrix before the first attention layer.

### Step 2: RoPE applied to Q, K

```python
def apply_rope(x, pos, base=10000.0):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Crucial: apply the same function to Q at position `m` and K at position `n`. Their dot product picks up a `cos((m-n)·θ_i)` factor on every coordinate pair.

### Step 3: ALiBi slopes and bias

```python
def alibi_slopes(n_heads):
    return [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]

def alibi_bias(n_heads, seq_len, causal=True):
    slopes = alibi_slopes(n_heads)
    out = []
    for m in slopes:
        head_bias = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if causal and j > i:
                    row.append(float("-inf"))
                else:
                    row.append(-m * abs(i - j))
            head_bias.append(row)
        out.append(head_bias)
    return out
```

Add `bias[h]` to the attention score matrix of head `h`, then softmax.

### Step 4: verify relative-distance property of RoPE

```python
def demo_rope_relative():
    rng = random.Random(0)
    d = 16
    q = [rng.gauss(0, 1) for _ in range(d)]
    k = [rng.gauss(0, 1) for _ in range(d)]
    pairs = [(3, 5), (7, 9), (100, 102), (1024, 1026)]
    for pq, pk in pairs:
        q_rot = apply_rope(q, pq)
        k_rot = apply_rope(k, pk)
        d_prod = sum(qi * ki for qi, ki in zip(q_rot, k_rot))
        print(f"gap={pk - pq:>4}  score={d_prod:>18.6f}")
```

Pick two random vectors `a, b`. Rotate by `(pos_a, pos_b)`. Then by `(pos_a + k, pos_b + k)`. Both dot products must match.

## Use It

PyTorch 2.5+ ships RoPE utilities. Most production code uses `flash_attn` or `xformers`:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, ...}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.** Rescale `base` to `base * (scale_factor)^(d/(d-2))` when extending from 4K to 16K+.
- **YaRN.** Smarter interpolation that preserves attention entropy on long contexts.
- **LongRoPE.** Evolutionary search to pick per-dimension scale factors.
- **Position interpolation + fine-tuning.** Shrink positions by the extension factor and fine-tune.

## Ship It

See `outputs/skill-positional-encoding-picker.md`. The skill picks an encoding strategy for a new model.

## Exercises

1. **Easy.** Plot the sinusoidal PE matrix as a heatmap for `max_len=512, d=128`. Confirm the "stripes get wider as dimension index grows" pattern.
2. **Medium.** Implement NTK-aware RoPE scaling. Train a tiny LM on sequences of length 256, then test on length 1024 with and without scaling.
3. **Hard.** Implement ALiBi and RoPE in the same attention module. Compare degradation on a copy task when extrapolating from 512 to 2048.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position |
| Sinusoidal | "The original one" | sin/cos at geometric frequencies added to embeddings |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation preserving attention entropy |

## Further Reading

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762)
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071)
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/04-positional-encoding)
