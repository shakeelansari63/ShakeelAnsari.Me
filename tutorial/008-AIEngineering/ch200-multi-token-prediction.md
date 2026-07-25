# Multi-Token Prediction (MTP)

> Every autoregressive LLM trains on one loss per position. DeepSeek-V3 added a second: predict the token after that. The extra parameters got distilled back through gradient flow, and the trained heads were repurposed as speculative-decoding drafters.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Learning Objectives

- State the MTP training objective and derive the joint loss
- Explain the difference between parallel MTP (Gloeckle) and sequential MTP (DeepSeek)
- Compute the parameter and memory overhead of MTP modules
- Implement one MTP module from scratch

## The Problem

Next-token prediction supervises every hidden state to predict exactly one thing. Most of a sequence's information extends beyond one token -- structure, coherence, factuality. The model learns these by accumulating many one-token signals over trillions of tokens.

MTP asks: what if every hidden state were supervised to predict multiple future tokens at once?

Gloeckle et al. (Meta, 2024) put independent output heads on top of the backbone, each predicting a different offset. Parallel, simple, but heads saw the same hidden state without hierarchical refinement, and predictions did not chain causally.

DeepSeek-V3 re-designed MTP as sequential modules preserving the causal chain, making them usable as speculative-decoding drafters.

## The Concept

### The Sequential MTP Recipe

DeepSeek-V3 adds `D` MTP modules. Each module `k` predicts `t_{i+k}` given prefix through position `i`.

Module `k` consists of: transformer block `T_k`, projection `M_k`, shared embedding `E`, shared output head `Out`.

At training for prefix through position `i`:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k(M_k · concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))))   for k >= 1
logits_{i+k} = Out(h_i^(k-1))
L_k = CE(logits_{i+k}, t_{i+k})
L_MTP = (λ / D) · Σ_{k=1..D} L_k
```

`λ` is a small weighting factor: 0.3 for first 10% of training, 0.1 after.

### Why Sequential

Parallel MTP has D heads on the same hidden state, each firing independently. Sequential builds `h_i^(k)` from `h_i^(k-1)` plus the actual next-token embedding, preserving the causal chain. This makes MTP modules directly usable as speculative-decoding drafters.

### Parameter Accounting

For hidden `h` and vocab `V`:

- Shared output head: reuse main model's head. Zero extra.
- Shared embedding: reuse main model's table. Zero extra.
- Per MTP module: projection `M_k` (2h²) + transformer block (~12h²) = ~14h².

DeepSeek-V3 with h=7168, D=1: ~720M paper estimate, 14B reported (MoE in MTP modules too).

### Speculative-Decoding Payoff

- Training: 10% slowdown (more compute, extra loss)
- Inference: free draft with 80%+ acceptance, ~1.8x throughput

The 10% training cost pays back the first time you run inference.

### MTP vs EAGLE

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible | Yes | No (re-train needed) |
| Accept rate | 0.88-0.92 | 0.80+ |
| Extra benefit | Speedup only | Denser signal + speedup |

## Build It

### Step 1: Shared Embedding

Single `vocab_size x hidden` table used by main model and every MTP module.

### Step 2: Per-Depth Combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)
    projected = matvec(M_k, concat)
    return projected
```

Real DeepSeek-V3 concatenates to [2h] and projects with h x 2h matrix.

### Step 3: Transformer Block at Depth k

Self-attention + MLP. In the toy: one-layer linear attention + SwiGLU MLP.

### Step 4-6: Shared Output Head, Per-Depth Loss, Parameter Accounting

Reuse main model's output projection. Cross-entropy per depth, aggregated with λ/D scaling. Print parameter breakdown.

## Use It

MTP integrated in DeepSeek-V3 and R1 series. vLLM and SGLang have integration paths as of April 2026.

When to use: you control the full pre-training pipeline and want denser training signal + free speculative decoding.

When not: fine-tuning existing pre-trained models (MTP not trained). Research baselines.

## Ship It

This lesson produces `outputs/skill-mtp-planner.md` -- plan for integrating MTP into a pre-training run.

## Exercises

1. Run the implementation and show per-depth loss decreases monotonically.
2. Compute parameter overhead for dense 70B (hidden 8192) with D=1. Compare to DeepSeek-V3's 14B.
3. Implement D=2. Verify joint loss matches DeepSeek equations 19-21.
4. Switch to parallel MTP (Gloeckle-style) and compare losses per depth.
5. Use trained MTP module as EAGLE-style draft and measure acceptance rate.

## Further Reading

- DeepSeek-AI, "DeepSeek-V3 Technical Report" (arXiv:2412.19437)
- Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction" (arXiv:2404.19737)
- Leviathan et al., "Fast Inference via Speculative Decoding" (arXiv:2211.17192)
- Li et al., "EAGLE-3" (arXiv:2503.01840)
