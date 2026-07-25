# Speculative Decoding and EAGLE-3

> The Leviathan rejection rule preserves the verifier's distribution exactly. EAGLE-3 turned the draft model into a purpose-built network trained on the verifier's own hidden states. Result: 3x to 6.5x speedup, acceptance rates above 0.9, no distributional tradeoff.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## Learning Objectives

- State the Leviathan theorem and prove the speculative loop produces identical distributions
- Walk the progression from vanilla spec-decoding through EAGLE-1/2/3 and name each limitation removed
- Compute expected speedup from acceptance rate and cost ratio
- Implement the full speculative loop: draft, verify, reject-sample, KV rollback, bonus token

## The Problem

Autoregressive decoding on a 70B model runs at ~35 tokens/sec on H100. Memory bandwidth is the ceiling: every token loads 70B of weights from HBM. Speculative decoding: a cheap draft proposes N tokens, the verifier runs once on all N, accepting or rejecting each.

The Leviathan theorem: the output distribution is identical to direct sampling from the verifier. Not approximate. Identical.

## The Concept

### The Leviathan Rejection Rule

Let `p(t)` be the draft's distribution, `q(t)` be the verifier's. Sample `d ~ p`. Accept with probability `min(1, q(d) / p(d))`. On reject, sample from `(q - p)_+ / ||(q - p)_+||_1`. Result: exact samples from `q`.

Stack N calls: verifier processes `prefix + d_1 + ... + d_N` in one pass. Walk left to right. On first rejection at position j, sample from residual and stop. On full acceptance, sample bonus token from `q_{N+1}`.

### Speedup Math

Let `α` = expected acceptance rate, `c = cost(draft) / cost(verifier)`. Expected accepted tokens per verifier forward: `(1 - α^(N+1)) / (1 - α)`. Speedup = `E[accepted] / (N*c + 1)`.

For `α = 0.8, c = 0.05`: optimal N ~5-7, speedup 3.2x. For `α = 0.95`: speedup pushes 5x.

### The Progression

| Strategy | Draft type | α | Speedup |
|----------|-----------|-----|---------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3x |
| Medusa | Extra LM heads | 0.65-0.75 | 2-3x |
| EAGLE-1 | 1-layer on hidden states | 0.70-0.80 | 2.5-3x |
| EAGLE-2 | + dynamic draft tree | 0.80-0.88 | 3-4x |
| EAGLE-3 | + training-time test | 0.88-0.92 | 3.5-6.5x |

**EAGLE-1 (2024):** Draft is a tiny transformer taking the verifier's last-layer hidden state as input. Because it sees the verifier's features, distribution is much closer.

**EAGLE-2:** Dynamic draft tree instead of linear chain. Propose a tree of candidates, score all with tree attention in one verifier pass.

**EAGLE-3 (NeurIPS 2025):** Drops feature-prediction loss, trains on direct token prediction. Training-time test (TTT): feed draft's own predictions back during training, aligning train and inference distributions.

### KV Cache Rollback

Verification extends the KV cache by N entries. On rejection at position j, truncate to `prefix_length + j + 1`. Write to scratch buffer and commit on acceptance, or maintain logical length and truncate on reject.

## Build It

### Step 1: Rejection Rule

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0: return True
    return u < min(1.0, q_prob / p_prob)

def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw] if s > 0 else list(q)
```

### Step 2: Full Speculative Step

Draft N tokens from `p`. Verify all in one parallel `q` evaluation. Walk left to right applying rejection rule. On first rejection, sample correction from residual. If all accept, emit bonus token from `q_{N+1}`.

### Step 3: Leviathan Check

Run 50,000 speculative steps. Compare empirical distribution to 50,000 direct samples from `q`. Chi-square should be well under critical value.

### Step 4: Speedup vs α Sweep

Perturb `p` away from `q` at different amplitudes. Measure α, plot expected tokens per verifier call. EAGLE-3 class quality (α ≈ 0.9) unlocks 4-5 tokens per verifier call.

## Use It

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{"model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5, "method": "eagle3"}'
```

When to reach for it: interactive chat where p50 latency matters. Code generation where α is above 0.9. Long-form generation.

When not: very small models (< 3B). Tiny batch-1 CPU deployments. Very high temperature creative sampling.

## Ship It

This lesson produces `outputs/skill-eagle3-tuner.md` -- recommends spec-decoding strategy and tuning for a given workload.

## Exercises

1. Run the speculative loop and confirm chi-square passes on 50,000 samples.
2. Sweep N from 1 to 10 with α=0.9, c=0.04. Plot expected tokens per verifier call. Find optimal N.
3. Modify to simulate EAGLE-2 tree search with shape [2,2,2]. Compare to linear chain.
4. Implement batched KV rollback for two concurrent sequences.
5. Read EAGLE-3 Section 4 on TTT. Explain in two sentences why naive draft training suffers from exposure bias.

## Further Reading

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
- Li et al., "EAGLE-2: Faster Inference with Dynamic Draft Trees" (2024)
- Li et al., "EAGLE-3: Scaling up Inference Acceleration via TTT" (NeurIPS 2025)
- Cai et al., "Medusa: Multiple Decoding Heads" (2024)
