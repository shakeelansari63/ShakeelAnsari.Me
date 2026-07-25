# Jamba -- Hybrid SSM-Transformer

> AI21's Jamba puts Transformer and Mamba layers in the same model: 1 Transformer layer for every 7 Mamba layers, MoE on every other block, and a 256k context window that fits on a single 80GB GPU.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (NSA)
**Time:** ~60 minutes

## Learning Objectives

- Explain the three primitives in a Jamba block and the 1:7:even interleaving recipe
- State what an SSM's recurrence looks like and why it enables constant-memory inference
- Compute the KV cache footprint of a Jamba model at 256k vs a pure Transformer
- Name the three Mamba-3 innovations and the problem each targets

## The Problem

Attention is quadratic. SSMs are linear. At 256k tokens, a Transformer attention map is 65B entries per head; an SSM's recurrent state is fixed-size.

Pure-SSM models match Transformer perplexity at small scales but lag on state-tracking and in-context retrieval. SSMs compress history into a fixed state, and when history is long, information leaks.

The fix: use both. Jamba is the first production-grade hybrid at scale (52B total, 12B active, 256k context).

## The Concept

### An SSM in One Page

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Computing `y_t` needs only `h_{t-1}` and `x_t`. Memory is constant. Inference is O(1) per token.

S4 used structured A matrix. Mamba made A, B, C data-dependent ("selective"). Mamba-2 simplified further. Mamba-3 adds complexity in specific places.

### The Jamba Block

Interleaves layers by two numbers:
- `l = 8`: 1 Transformer for every 7 Mamba layers
- `e = 2`: every other layer applies MoE

Layer sequence within a block:
```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (| marks MoE)
```

At 4 blocks (32 layers): 28 Mamba + 4 Attention. 16 use MoE.

### Why 1:7

AI21 ran ablations. Too much attention (1:1): quality up but memory degrades. Too little (1:15): memory great but retrieval fails. Sweet spot: 1:7 or 1:8. Transformer layers handle exact recall and state tracking. Mamba layers handle cheap bulk processing.

### The Memory Budget

For Jamba-1 (32 layers: 28 Mamba + 4 Attention, hidden 4096, 32 heads):
- KV cache (attention only): 2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB
- SSM state: 28 * 4096 * 16 * 2 = 3.7 MB (fixed, does not grow with sequence)

Compare pure Transformer (32 layers, full MHA): 2 * 32 * 32 * 128 * 256k * 2 = 128 GB. Even vs GQA(8): 32 GB. Jamba's hybrid is 8x to 16x smaller.

### Mamba-3 (ICLR 2026)

Three innovations:

1. **Exponential-trapezoidal discretization.** More expressive recurrence than Mamba-2's Euler method.
2. **Complex-valued state update.** Re-adds complex values (removed in Mamba-2). Equivalent to data-dependent rotary embedding on the state. Restores state-tracking.
3. **Multi-input multi-output (MIMO) projections.** Matrix-valued projections instead of per-feature scalar. Improves modeling and hardware utilization.

At 1.5B: +0.6 points over Gated DeltaNet. MIMO variant adds +1.2 more.

### When to Reach for Hybrid

Win when: context is 64k+, tasks mix short-range structure with long-range recall, single-GPU memory where Transformer KV cache alone would not fit.

Lose when: context under 16k, tasks needing everywhere-to-everywhere attention (cross-document references), scaling to trillion-parameter frontier (pure Transformer + MLA + MoE currently winning).

## Build It

The code is a memory calculator for hybrid architectures. Given SSM-Transformer ratio, hidden size, and layer count, it computes KV cache at target context, SSM state memory, and total memory for pure Transformer, Jamba 1:7 hybrid, and pure SSM.

## Use It

Run the calculator to reproduce Jamba's 8x memory reduction claim. Most production servers (vLLM, SGLang) support Jamba. At 256k, Jamba's memory advantage shows in concurrent-request throughput.

## Ship It

This lesson produces `outputs/skill-hybrid-picker.md` -- recommends between pure Transformer, Jamba hybrid, and pure SSM based on workload.

## Exercises

1. Run calculator for 32-layer pure Transformer (hidden 4096, 32 heads) and Jamba hybrid at 256k. Verify 8x reduction.
2. Modify calculator for 1:3 and 1:15 ratios. Plot KV cache vs ratio. Find where KV cache equals SSM state.
3. Read Section 3 of Jamba paper. Explain why AI21 chose Mamba-1 over Mamba-2.
4. Compute parameter overhead of MoE-every-other-layer in Jamba 1.5 Large. Compare active ratio to DeepSeek-V3.
5. Read Mamba-3 Section 3. Explain why complex-valued state update is equivalent to data-dependent rotary embedding.

## Further Reading

- Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model" (arXiv:2403.19887)
- AI21, "Jamba 1.5: Hybrid Transformer-Mamba at Scale" (arXiv:2408.12570)
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling" (arXiv:2312.00752)
- Gu, Dao, "Mamba-2" (arXiv:2405.21060)
- Lahoti et al., "Mamba-3" (arXiv:2603.15569, ICLR 2026)
