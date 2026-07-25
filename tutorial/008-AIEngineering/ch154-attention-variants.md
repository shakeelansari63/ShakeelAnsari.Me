# Attention Variants — Sliding Window, Sparse, Differential

> Full attention is a circle. Every token sees every token, and memory pays the price. Four variants bend the shape of the circle and recover half the cost.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## The Problem

Full attention costs `O(N²)` memory and `O(N²)` compute in sequence length. For a 128K-context Llama 3 70B that is 16 billion attention entries per layer, times 80 layers. Flash Attention hides the `O(N²)` activation memory but does not change the arithmetic cost.

Three classes of variants change the topology of the attention matrix itself:

1. **Sliding window attention (SWA).** Each token attends to a fixed window of neighbors, not the full prefix. Memory and compute drop to `O(N · W)`.
2. **Sparse / block attention.** Only selected pairs `(i, j)` get scored; the rest are forced to zero weight.
3. **Differential attention.** Compute two attention maps with separate Q/K projections, subtract one from the other. Kills the "attention sink."

## The Concept

### Sliding Window Attention (SWA)

Each query at position `i` attends only to positions in `[i - W, i]` (causal SWA). Tokens outside the window get `-inf`.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

**KV cache shrinks with SWA.** Only the last `W` tokens of K and V need to be kept per layer.

### Sparse / Block Attention

Three canonical shapes:

- **Local + strided (OpenAI sparse transformer).** Attend to the last `W` tokens plus every `stride`-th token before.
- **Longformer / BigBird.** Local window + global tokens + random-sparse links.
- **Native Sparse Attention (DeepSeek, 2025).** Learn which blocks matter; skip zero blocks at kernel level.

### Differential Attention (DIFF Transformer, 2024)

Regular attention has an "attention sink" problem: softmax forces every row to sum to 1, so uninformative queries dump weight on the first token. Differential attention computes **two** attention maps and subtracts:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

### Variant Comparison

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl | Gemma 2/3, Phi-3-Long |
| Local + strided | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer |
| BigBird | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse | O(N · active) | O(N) | within 0.05 ppl | DeepSeek-V3.2 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer |

## Build It

### Step 1: full causal mask (baseline)

```python
NEG_INF = float("-inf")

def causal_mask(n):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            M[i][j] = 0.0
    return M
```

### Step 2: sliding window causal mask

```python
def swa_mask(n, window):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

For `window >= n`, you recover full causal attention.

### Step 3: local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

### Step 4: differential attention

```python
def diff_attention_row(q1, q2, K1, K2, V, mask_row, lam):
    _, w1 = attention_row(q1, K1, V, mask_row)
    _, w2 = attention_row(q2, K2, V, mask_row)
    diff = [a - lam * b for a, b in zip(w1, w2)]
    d_v = len(V[0])
    out = [0.0] * d_v
    for w, v in zip(diff, V):
        for j in range(d_v):
            out[j] += w * v[j]
    return out, diff
```

Two attention passes, subtract with a learned mixing coefficient.

### Step 5: KV cache sizes

```python
def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * dtype_bytes
```

At 128K context, Llama-3-70B-ish (80 layers, 8 KV heads, d_head=128):

- Full: ~10.5 GB
- SWA window=1024: ~82 MB (128× shrink)
- Gemma-3 5:1 mix: ~1.9 GB (5.6× shrink)
- Differential: ~21 GB (2× cost)

## Use It

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention in PyTorch 2.5+:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

**When to pick each:**
- **Pure full attention** — every layer up to ~16K context, retrieval-critical
- **SWA + global mix** — long context (>32K), memory-bound. 2026 default above 32K
- **Sparse block attention** — custom kernel, specialized workloads
- **Differential attention** — attention-sink contamination hurts (long-context RAG)

## Ship It

See `outputs/skill-attention-variant-picker.md`. The skill picks an attention topology for a new model.

## Exercises

1. **Easy.** Verify SWA at `window=4` zeroes everything outside the last 4 tokens. Verify `window=n` reproduces full causal attention.
2. **Medium.** Implement causal SWA with `window=1024` on the capstone model. Compare val loss and peak memory vs full attention.
3. **Hard.** Implement a Gemma-3-style 5:1 layer mix in the capstone model. Compare loss, memory, and generation quality.
4. **Hard.** Implement differential attention with a learned `λ` per head. Train on a synthetic retrieval task.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Each query attends to its last `W` tokens |
| Effective receptive field | "How far back the model sees" | In an L-layer SWA stack, up to `L × W` tokens |
| Longformer / BigBird | "Local + global + random" | Sparse patterns with global tokens |
| Native Sparse Attention | "DeepSeek's kernel trick" | Learn block-level sparsity; skip zero blocks |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer: subtract second attention map to cancel sinks |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization forces weight on position 0 |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API that compiles mask functions into kernels |
| Layer type mix | "5:1 SWA-to-global" | Interleave sparse and full attention layers |

## Further Reading

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786)
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258)
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/15-attention-variants)
