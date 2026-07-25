# Scaling Laws

> The 2020 Kaplan paper said: bigger model, lower loss. The 2022 Hoffmann paper said: you were under-training. Compute goes into two buckets — parameters and tokens — and the split is not obvious.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## The Problem

When you have C FLOPs of training compute and want the best model, you face two knobs:

1. **How many parameters (N)?** Bigger model, higher capacity.
2. **How many training tokens (D)?** More data, better use of capacity.

FLOPs scale approximately as `6 × N × D`. You can push N up and D down, or D up and N down. Which is better?

Before 2022, the answer was "push N hard." GPT-3 (2020) was 175B parameters trained on ~300B tokens. A ratio of about 1.7 tokens per parameter. The Kaplan scaling laws backed this up.

Hoffmann et al. (2022), training a small family of models called Chinchilla, found something different: optimal ratio is closer to **20 tokens per parameter**. GPT-3 was 10× undertrained. Chinchilla (70B params, 1.4T tokens) beat GPT-3 (175B, 300B tokens) on every benchmark at 2.5× less inference cost.

2026 is Chinchilla's world — with one important twist. Llama 3 8B was trained on 15 trillion tokens, a ratio of 1,875 tokens per parameter. Inference cost matters more than training cost for models that will be used at scale.

## The Concept

### The Hoffmann law

From the Chinchilla paper, loss follows:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters (non-embedding).
- `D` = training tokens.
- `α ≈ 0.34`, `β ≈ 0.28` (roughly symmetric).
- `E ≈ 1.69`, the irreducible loss ceiling.
- `A ≈ 406`, `B ≈ 411`.

Two terms trade against each other as you scale. Take the derivative w.r.t. `N` at fixed compute (C = 6ND) and solve:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal: 20 tokens per parameter.

### Why over-training anyway

Chinchilla-optimal minimizes training loss per training FLOP. But you pay training cost once; inference cost forever.

For a chatbot that serves a trillion tokens per month, inference dominates total cost. Llama's approach: train smaller, longer. 8B at 15T tokens is deeply inference-optimized.

### The 2026 picture

| Factor | Changed how |
|--------|-------------|
| Data quality | Curating "good" tokens shifts curves by >2× effective compute |
| MoE | Total params decouple from active FLOPs |
| Post-training | Some capabilities shift with SFT+RLHF more than pretraining |
| Multimodality | Image + text tokens scale together |
| Synthetic data | Models generate training data; effective compute can compound |

## Build It

### Step 1: Chinchilla loss

```python
A = 406.4
B_CONST = 410.7
ALPHA = 0.34
BETA = 0.28
E_CONST = 1.69

def chinchilla_loss(N, D, A=A, B=B_CONST, alpha=ALPHA, beta=BETA, E=E_CONST):
    return A / N ** alpha + B / D ** beta + E
```

### Step 2: compute-optimal frontier

```python
def compute_optimal(C_flops, n_grid=200):
    log_N_min = math.log10(1e5)
    log_N_max = math.log10(1e13)
    best = (None, None, float("inf"))
    for i in range(n_grid):
        log_N = log_N_min + (log_N_max - log_N_min) * i / (n_grid - 1)
        N = 10 ** log_N
        D = C_flops / (6 * N)
        if D < 1e6:
            continue
        loss = chinchilla_loss(N, D)
        if loss < best[2]:
            best = (N, D, loss)
    return best
```

For compute budgets from `1e17` to `1e25` FLOPs, verify the ratio `D/N ≈ 20`.

### Step 3: over-training cost

```python
C = 1e24
N_opt, D_opt, L_opt = compute_optimal(C)
N_under = N_opt / 10
D_over = D_opt * 10
L_over = chinchilla_loss(N_under, D_over)
print(f"chinchilla optimal:  loss={L_opt:.3f}")
print(f"over-trained:        loss={L_over:.3f}")
print(f"inference savings:   {N_opt / N_under:.0f}x")
```

### Step 4: compare to real models

```python
models = [
    ("GPT-3 175B",          175e9,  300e9),
    ("Chinchilla 70B",       70e9, 1400e9),
    ("Llama 2 70B",          70e9, 2000e9),
    ("Llama 3 8B",            8e9, 15_000e9),
    ("Llama 3 70B",          70e9, 15_000e9),
    ("DeepSeek-V3 (active)", 37e9, 14_800e9),
    ("Qwen 2.5 72B",         72e9,  18_000e9),
]
for name, N, D in models:
    L = chinchilla_loss(N, D)
    print(f"  {name:<22}  D/N={D/N:>6.1f}  loss={L:>6.3f}")
```

## Use It

Scaling laws tell you:

1. **Whether your fine-tune has enough data.** If your task-specific data is below 20 tokens per param of the base model, expect saturation.
2. **Whether to pick a bigger base model.** If spending all budget on inference, prefer a smaller, longer-trained model.
3. **Where the returns diminish.** Beyond 1000× Chinchilla-optimal, log-loss changes become noise.

## Ship It

See `outputs/skill-training-budget-estimator.md`. The skill picks `(N, D, hours, GPU)` for a new training run.

## Exercises

1. **Easy.** Print Chinchilla-optimal `(N, D)` for compute budgets `1e20`, `1e22`, `1e24`. Compare to the real model table.
2. **Medium.** Implement the Hoffmann loss-as-function-of-compute curve. Identify when `>10^28` FLOPs are needed for the next 0.1 reduction in cross-entropy.
3. **Hard.** Fit your own scaling law on 5 tiny models (100K to 10M params) trained on the same dataset.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Parameters (N) | "Model size" | Non-embedding weight count; determines capacity |
| Tokens (D) | "Training data" | Number of training tokens seen |
| Compute (C) | "FLOPs spent" | Approximately `6 × N × D` |
| Chinchilla-optimal | "D/N ≈ 20" | Ratio minimizing loss per FLOP of pretraining |
| Over-training | "Past Chinchilla" | Spend extra training FLOPs to save inference FLOPs |
| Irreducible loss | "The floor" | The `E` term; the entropy of the data itself |
| Emergent capability | "Sudden jumps at scale" | Often a scorer artifact; continuous loss is smooth |
| Effective compute | "Training-efficiency multiplier" | Better data/optimizer/architecture multiplies a FLOP |

## Further Reading

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/13-scaling-laws)
