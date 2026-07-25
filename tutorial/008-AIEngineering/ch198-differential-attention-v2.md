# Differential Attention (V2)

> Softmax attention spreads a small amount of probability over every non-matching token. Over 100k tokens that noise adds up. Differential Transformer computes attention as the difference of two softmaxes, subtracting the shared noise floor.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 10 · 14 (architecture walkthroughs)
**Time:** ~60 minutes

## Learning Objectives

- State why softmax attention has a noise floor and why it grows with context length
- Derive the differential attention formula and explain why subtraction cancels shared noise
- Walk the V1-to-V2 diff and explain each change
- Implement differential attention and verify noise cancellation on synthetic data

## The Problem

Softmax can never produce exact zeros. Every non-matching token gets some positive mass. At 128k tokens, even 0.001% per token sums to ~12% of the total probability. The model learns to route around a noise floor that grows with context.

Empirically: hallucinated citations in long-context RAG, lost-in-the-middle failures on 100k-token retrieval, subtle accuracy degradation past 32k.

DIFF V1 (ICLR 2025) measured the gap: Differential Transformers hit lower perplexity and higher long-context accuracy than same-size baselines. DIFF V2 (January 2026) made it production-ready.

## The Concept

### The Noise Floor

For query `q` and keys `K = [k_1, ..., k_N]`:

`w_i = exp(q·k_i / √d) / Σ_j exp(q·k_j / √d)`

No `w_i` is ever zero. Each unrelated token contributes `O(1/N)`. Total noise: `O(1)`.

### The Differential Idea

Split each head's Q and K into two: `Q = (Q_1, Q_2)`, `K = (K_1, K_2)`. Compute two attention maps:

```
A1 = softmax(Q_1 K_1^T / √d)
A2 = softmax(Q_2 K_2^T / √d)
DiffAttn = (A1 - λ * A2) V
```

The subtraction cancels whatever noise the two maps share (the 127k unrelated tokens). Signal -- peaked weight on relevant tokens -- only cancels if it appears in both at the same magnitude, which training prevents.

`λ` is a learnable per-head scalar: `λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init`. Can be negative.

### V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Head dim | Halved (parameter neutral) | Same as baseline |
| Q heads | Same as baseline | Doubled |
| KV cache | Loaded twice per decode | Loaded once |
| FlashAttention | Custom kernel needed | Compatible |
| Per-head RMSNorm | Required for stability | Removed (simpler init) |
| Decode speed | Slower than baseline | Matches baseline |

V2 doubles Q heads while keeping KV heads the same, borrowing parameters from the up-projection. After subtraction, the extra dimension projects back down.

### When to Reach For It

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Cleaner attention, fewer hallucinations |
| Needle-in-haystack (32k+) | Substantial accuracy lift |
| Short chat (< 4k) | Indistinguishable from baseline |

## Build It

### Step 1: Standard Softmax Attention

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2: Split Q, K into Two Halves

V1 style: halve head dimension. V2 style: double Q heads. The toy uses V1 for clarity.

### Step 3: Two Softmax Branches + Subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Output weights can be negative -- V projection absorbs the sign.

### Step 4: Noise Cancellation Measurement

Build synthetic sequence of length 1024. Place signal at known position, rest is noise. Compare standard vs differential attention weight on the signal position. DIFF produces 3-10x higher signal-to-noise ratio.

### Step 5: V1 vs V2 Parameter Accounting

For hidden=4096, heads=32, d_head=128: V2 adds roughly `hidden * hidden` extra per attention block (doubling Q size).

## Use It

DIFF V2 integration is underway in vLLM and SGLang as of April 2026. Reach for it when training a new model targeting 64k+ context. Apply LoRA on Q projections to approximate DIFF on existing weights.

## Ship It

This lesson produces `outputs/skill-diff-attention-integrator.md` -- integration plan for adding DIFF attention to pre-training or fine-tuning.

## Exercises

1. Run the implementation and verify signal-to-noise is higher for DIFF. Vary noise amplitude to find crossover.
2. Compute parameter-count delta from baseline to V1 and V2 for 7B-class model.
3. Read DIFF V1 Section 3 and V2 Section 2. Explain why V1 needed per-head RMSNorm and V2 removed it.
4. Sweep λ from 0 to 1. Measure signal-to-noise. Find the optimal λ.
5. Extend to GQA + DIFF V2: pick 8 KV heads, 32 Q heads. Show KV cache matches baseline.

## Further Reading

- Ye et al., "Differential Transformer" (arXiv:2410.05258, ICLR 2025)
- Microsoft unilm, "Differential Transformer V2" (HuggingFace blog, January 2026)
- Liu et al., "Lost in the Middle" (arXiv:2307.03172)
