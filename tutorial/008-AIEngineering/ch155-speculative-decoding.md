# Speculative Decoding — Draft, Verify, Repeat

> Autoregressive decoding is serial. Each token waits for the previous one. Speculative decoding breaks the chain: a cheap model drafts N tokens, the expensive model verifies all N in one forward pass. When the draft is right you paid one big forward for N generations.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## The Problem

A 70B LLM sampling one token takes ~30 ms on an H100. A 3B draft model takes ~3 ms. If we let the 3B draft 5 tokens ahead, then run the 70B *once* to verify all 5, the total is `5×3 + 30 = 45 ms` for up to 5 accepted tokens — versus `5×30 = 150 ms` for straight-line generation. That is the full speculative-decoding pitch: trade a small amount of extra GPU memory (draft model) for 2–4× lower decode latency.

The trick preserves the distribution. Speculative sampling guarantees that the output sequence is **identically distributed** to what the big model would have produced on its own. No quality tradeoff. Just faster.

Four families of draft-verifier pairs dominate 2026 inference:

1. **Vanilla speculative (Leviathan 2023).** Separate draft model + verifier.
2. **Medusa (Cai 2024).** Multiple decoding heads on the verifier predict `t+1..t+k` in parallel.
3. **EAGLE family (Li 2024, 2025).** Lightweight draft that reuses the verifier's hidden states.
4. **Lookahead decoding (Fu 2024).** Jacobi iteration; no draft model required.

## The Concept

### The core algorithm

Given a verifier `M_q` and a cheaper draft `M_p`:

1. Let `x_1..x_k` be the prefix already decoded.
2. **Draft**: use `M_p` to autoregressively propose `d_{k+1}, ..., d_{k+N}` with probabilities `p_1..p_N`.
3. **Verify in parallel**: run `M_q` once on `x_1..x_k, d_{k+1}, ..., d_{k+N}`, getting probabilities `q_1..q_{N+1}`.
4. **Accept/reject**: for each `i`, accept with probability `min(1, q_i(d_i) / p_i(d_i))`.
5. On first rejection at position `j`: sample from residual `(q_j - p_j)_+` normalized.
6. On accepting all `N`: sample one bonus token from `q_{N+1}`.

### What determines speedup

Let `α` = expected acceptance rate. Per step:

- Naive: 1 big-model call per token.
- Speculative: 1 big-model call per `(1 - α^{N+1}) / (1 - α)` tokens.

At `α = 0.75` and `N = 5`: ~3× fewer big-model calls.

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft | Fast prototype, no training | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, max speed | 3–4× |
| Lookahead | No draft, no training | 1.3–1.6× |

## Build It

### Step 1: the rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` is a uniform random number. This Bernoulli decision preserves the verifier's distribution exactly.

### Step 2: residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0.0:
        return list(q)
    return [r / s for r in raw]
```

Subtract `p` from `q` element-wise, clamp negatives to zero, renormalize.

### Step 3: one speculative step (N draft tokens)

```python
def spec_step_n(q, p, N, rng):
    accepted = 0
    for _ in range(N):
        d = sample(p, rng)
        p_prob = p[d]
        q_prob = q[d]
        u = rng.random()
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            accepted += 1
        else:
            return sample(residual(q, p), rng), accepted
    bonus = sample(q, rng)
    return bonus, accepted + 1
```

Five accepted → one bonus → six tokens produced in one verifier pass.

### Step 4: measure acceptance rate

```python
def acceptance_rate(q, p, n_samples, rng):
    hits = 0
    for _ in range(n_samples):
        d = sample(p, rng)
        u = rng.random()
        q_prob = q[d]
        p_prob = p[d]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            hits += 1
    return hits / n_samples
```

### Step 5: verify distribution equivalence

```python
def run_distribution_check(q, p, n_samples, rng):
    spec_counts = [0] * len(q)
    direct_counts = [0] * len(q)
    for _ in range(n_samples):
        d, _ = spec_step_one_token(q, p, rng)
        spec_counts[d] += 1
        direct_counts[sample(q, rng)] += 1
    return spec_counts, direct_counts
```

A chi-square test confirms the speculative histogram matches the direct sample within sampling error.

### Step 6: expected tokens per verify call

```python
def expected_tokens_per_verify(alpha, N):
    if alpha >= 1.0:
        return N + 1
    if alpha == 0:
        return 1
    return (1 - alpha ** (N + 1)) / (1 - alpha)
```

At `α=0.85` and `N=5`: ~4.1 tokens per verifier call ≈ 4× fewer big-model forwards.

## Use It

Production:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --num-speculative-tokens 5

# vLLM with vanilla draft
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

**When NOT to spec-decode:**
- Single-sequence generation of 1–5 tokens (overhead dominates).
- Wildly creative / high-temperature sampling (α drops).
- Memory-constrained deployments (draft model adds VRAM).

## Ship It

See `outputs/skill-spec-decode-picker.md`. The skill picks a speculative decoding strategy and tuning parameters.

## Exercises

1. **Easy.** Confirm the speculative token distribution matches the verifier's direct-sample distribution on 50,000 tokens within chi-square p > 0.05.
2. **Medium.** Plot speedup as a function of `N` for `α = 0.5, 0.7, 0.85`. Identify optimal `N` for each.
3. **Hard.** Implement a tiny Medusa: take the capstone GPT, add 3 extra LM heads predicting t+2, t+3, t+4.
4. **Hard.** Implement KV rollback: feed 5 draft tokens, simulate rejection at position 3, verify cache reads correctly.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | Smaller model proposing candidate tokens |
| Verifier | "The big one" | Target model whose distribution we preserve |
| Acceptance rate (α) | "How often the draft is right" | Per-token acceptance probability, 0.7–0.9 typical |
| Residual distribution | "The rejection fallback" | `(q - p)_+` normalized; preserves verifier distribution |
| Bonus token | "The free one" | Extra token sampled when all N drafts are accepted |
| Medusa | "Draft-less speculative" | Multiple LM heads predicting future tokens |
| EAGLE | "Hidden-state draft" | Tiny draft conditioned on verifier's hidden states |
| Lookahead decoding | "Jacobi iteration" | Self-speculation with no draft model |
| Tree attention | "Verify many candidates at once" | Branching verification for multiple continuations |
| KV rollback | "Undo rejected drafts" | Scratch buffer; commit on acceptance, discard on reject |

## Further Reading

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al. (2023). Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774)
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [Li et al. (2025). EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference](https://arxiv.org/abs/2402.02057)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/16-speculative-decoding)
