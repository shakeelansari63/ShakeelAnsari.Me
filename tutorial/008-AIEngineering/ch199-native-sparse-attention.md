# Native Sparse Attention (DeepSeek NSA)

> At 64k tokens, attention eats 70-80% of decode latency. DeepSeek's NSA (ACL 2025 Best Paper) runs three parallel attention branches -- compressed coarse-grained, selectively retained fine-grained, and sliding windows -- combined through a learned gate.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 14 (architecture walkthroughs)
**Time:** ~60 minutes

## Learning Objectives

- State the three NSA attention branches and what each captures
- Explain why NSA is "natively trainable" where prior sparse-attention methods were inference-only
- Compute the attention savings vs full attention at 64k context
- Implement the three-branch combination and verify gating weights behave

## The Problem

Full attention at sequence N costs `O(N^2)` time and `O(N)` KV cache. At 64k tokens, attention accounts for 70-80% of decode latency.

Prior sparse attention falls into two buckets: fixed-pattern (sliding window, strided) that fails on long-range recall, and inference-time pruning (H2O, StreamingLLM) that recovers only partial speedup because the model was not trained to route through the sparse pattern.

NSA is both: a sparsity pattern learned during pre-training, implemented as a kernel-aligned algorithm.

## The Concept

### Three Parallel Branches

For each query, NSA runs attention three times against different views:

1. **Compressed branch.** Tokens grouped into blocks of size `l` (typically 32-64). Each block compressed to one summary token via a learned MLP. Query attends over compressed tokens for a coarse-grained view.

2. **Selected branch.** Using compressed-branch attention scores, top-k most relevant blocks are identified. Fine-grained (uncompressed) tokens from those blocks are loaded and attended.

3. **Sliding-window branch.** Query attends to the most recent `W` tokens (typically 512) for local context.

Output = `g_cmp * out_cmp + g_sel * out_sel + g_win * out_win`, gates from a small MLP on the query.

### Why It Is Natively Trainable

The selection step (top-k blocks) is discrete, breaking gradient flow. NSA sidesteps this: the compressed-branch attention is a differentiable coarse-grained attention on the whole sequence. The top-k just reuses top attention scores to pick which fine-grained blocks to load. Gradients flow through compressed-branch scores. Top-k is a no-op on the forward graph -- it only controls memory loading.

### Compute Budget

With `N=64k, l=64, k=16, b=64, w=512`:

- Compressed: `N/l = 1000` keys per query
- Selected: `k*b = 1024` keys per query
- Sliding: `w = 512` keys per query
- Total: 2536 vs 64000 for full attention -- 25x reduction

At 128k: 3536 vs 128000 -- 36x reduction. Benefit grows with sequence length.

### Hardware Alignment

Loads queries by GQA groups (outer loop), fetches sparse KV blocks per group (inner loop), runs attention on SRAM. KV loads amortized across the group. Reported 9x faster than FlashAttention on 64k decodes.

### Comparison

| Method | Differentiable | Real speedup | Long-range recall |
|--------|---------------|-------------|-------------------|
| Sliding window only | yes | yes | fails |
| KV pruning | N/A (inference) | yes | partial |
| NSA | natively | 9x at 64k | matches full attention |

## Build It

### Step 1: Compress Tokens into Blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2-4: Three Branch Attentions

Compressed attention against compressed keys. Top-k selection based on compressed scores. Slide window attention on last W tokens.

### Step 5: Gate + Combine

Small MLP on query produces three gate weights. Weighted sum of branch outputs.

### Step 6: Compute Counting

On 1024-token synthetic with l=32, k=4, w=128: NSA sees 32 + 128 + 128 = 288 keys per query vs 1024 for full -- 3.5x fewer.

## Use It

NSA ships in DeepSeek's own long-context pre-training. Integration in vLLM/SGLang is experimental as of April 2026.

When to reach: pre-training targeting 64k+ context with a serious compute budget. Inference of DeepSeek's own long-context checkpoints.

When not: serving existing dense-attention models (cannot retrofit). Context under 16k.

## Ship It

This lesson produces `outputs/skill-nsa-integrator.md` -- NSA integration plan for long-context pre-training.

## Exercises

1. Sweep (l, k, w) across presets and find the lowest compute that maintains 95% recall on needle-in-haystack.
2. Replace mean-pool compressor with a trained MLP and measure perplexity gap.
3. Implement the gate MLP and show it behaves sensibly.
4. Compute KV cache for NSA-enabled 70B at 128k vs full attention and MLA.
5. Read NSA Section 4 and explain why compressed-branch scores are reused for selection.

## Further Reading

- Yuan et al., "Native Sparse Attention" (arXiv:2502.11089, ACL 2025 Best Paper)
- DeepSeek-V3 Technical Report (arXiv:2412.19437)
- Moonshot AI, "MoBA: Mixture of Block Attention" (arXiv:2502.13189)
- Dao et al., "FlashAttention-2" (arXiv:2307.08691)
