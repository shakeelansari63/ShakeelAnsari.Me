# Mixture of Experts (MoE)

> A dense 70B transformer activates every parameter for every token. A 671B MoE activates only 37B per token and beats it on every benchmark. Sparsity is the most important scaling idea of the decade.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## The Problem

A dense transformer's FLOPs at inference equal its parameter count (times 2 for forward pass). Scale up a dense model and every token pays the full bill. By 2024 the frontier was hitting a compute wall: to be meaningfully smarter, you needed exponentially more FLOPs per token.

Mixture of Experts breaks this link. Replace each FFN with `E` independent experts + a router that picks `k` experts per token. Total parameters = `E × FFN_size`. Active parameters per token = `k × FFN_size`. Typical 2026 configuration: `E=256`, `k=8`. Storage scales with `E`, compute scales with `k`.

The 2026 frontier is almost entirely MoE: DeepSeek-V3 (671B total / 37B active), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2.

## The Concept

### The FFN swap

Dense transformer block:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block:

```
h = x + attn(norm(x))
scores = router(norm(h))
top_k = argmax_k(scores)
h = h + sum_{e in top_k}( gate(scores[e]) * Expert_e(norm(h)) )
```

### The load-balancing problem

If the router puts 90% of tokens through expert 3, the other experts starve. Three fixes:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Add a penalty proportional to the variance in expert usage.
2. **Expert capacity + token dropping** (early Switch). Each expert processes at most `C × N/E` tokens; overflow skips the layer.
3. **Auxiliary-loss-free balancing** (DeepSeek-V3). Add a learned per-expert bias that shifts selection. No penalty on the main objective.

### Shared experts

DeepSeek-V2/V3 splits experts into *shared* and *routed*. Every token passes through all shared experts. Routed experts are picked via top-k.

### The cost profile

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

## Build It

### Step 1: the router

```python
def route(hidden, W_router, top_k, bias):
    E = len(W_router)
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(E)]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(E), key=lambda i: -biased[i])[:top_k]
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias affects selection, not gate weight. That is the DeepSeek-V3 trick.

### Step 2: auxiliary-loss-free balancing

```python
def update_bias(bias, usage_counts, target, gamma):
    for e in range(len(bias)):
        if usage_counts[e] > target:
            bias[e] -= gamma
        elif usage_counts[e] < target:
            bias[e] += gamma
    return bias
```

### Step 3: run 100 tokens through the router

```python
def run_epoch(tokens, experts, W_router, top_k, bias):
    usage = [0] * len(experts)
    for x in tokens:
        out = [0.0] * d_hidden
        top_idx, gates = route(x, W_router, top_k, bias)
        for e_idx, gate in zip(top_idx, gates):
            h = apply_expert(x, experts[e_idx])
            for j in range(d_hidden):
                out[j] += gate * h[j]
        for e in top_idx:
            usage[e] += 1
    return usage
```

Track which experts fire how often. Without the bias, usage is skewed. With bias update, usage converges to uniform.

### Step 4: param count comparison

```python
def dense_active_params(n_experts, expert_params, top_k, d_model):
    total = n_experts * expert_params
    active = top_k * expert_params
    return total, active
```

DeepSeek-V3-shaped: 256 routed + 1 shared, 8 active, d_model=7168. The total parameter count is eye-watering. The active count is a seventh of a dense Llama 3 70B.

## Use It

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

**When to pick MoE:** You want frontier quality at lower inference cost per token, have VRAM / expert-parallel infrastructure, and your workload is token-heavy.

**When NOT to pick MoE:** Edge deployment, latency-critical single-user serving, small models (<7B).

## Ship It

See `outputs/skill-moe-configurator.md`. The skill picks E, k, and shared-expert layout for a new MoE.

## Exercises

1. **Easy.** Watch how the auxiliary-loss-free bias update evens out expert usage over 50 iterations.
2. **Medium.** Replace the learned router with a hash-based router. Compare quality and balance.
3. **Hard.** Implement GRPO-style "rollout-matched routing": log which experts fire during inference, force same routing during gradient computation.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | "One FFN among many" | An independent feed-forward network |
| Router | "The gate" | Tiny linear layer that scores each token against each expert |
| Top-k routing | "k active experts per token" | Each token goes through exactly k experts |
| Auxiliary loss | "Load-balance penalty" | Extra loss term penalizing skewed expert usage |
| Auxiliary-loss-free | "DeepSeek-V3's trick" | Balance via per-expert bias on selection only |
| Shared expert | "Always on" | Expert through which every token passes |
| Expert parallelism | "Shard by expert" | Distribute experts to different GPUs |
| Sparsity | "Active params < total params" | The ratio `k × expert_size / (E × expert_size)` |

## Further Reading

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Fedus, Zoph, Shazeer (2022). Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/11-mixture-of-experts)
