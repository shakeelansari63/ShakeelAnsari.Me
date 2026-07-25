# DeepSeek-V3 Architecture Walkthrough

> DeepSeek-V3 turns all six architectural knobs and adds four more: Multi-Head Latent Attention, auxiliary-loss-free load balancing, Multi-Token Prediction, and DualPipe training.

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14, 17, 18, 19
**Time:** ~75 minutes

## Learning Objectives

- Read the DeepSeek-V3 config top to bottom and explain each field
- Derive total (671B) and active (37B) parameter counts
- Compute MLA's KV cache footprint vs GQA baseline
- State the four DeepSeek-specific innovations

## The Problem

DeepSeek-V3 is the first frontier open model whose architecture is meaningfully different from Llama. Llama 3 405B is "GPT-2 with six knobs turned." DeepSeek-V3 is GPT-2 with all six plus four more. Understanding it is table stakes for frontier LLM work.

## The Concept

### The Invariant Core

Autoregressive. Decoder blocks. Attention + MLP + two RMSNorms. SwiGLU. RoPE. Pre-norm. Weight-tied embeddings. Same as every Llama or Mistral.

### MLA Instead of GQA

K and V are compressed into a shared low-rank latent (`kv_lora_rank = 512`), then decompressed per head on the fly. KV cache stores only the latent.

At 128k context:
- DeepSeek-V3 MLA: 61 * 512 * 131072 * 2 = 7.6 GB
- Hypothetical GQA (8 KV heads): 2 * 61 * 8 * 128 * 131072 * 2 = 30.5 GB

MLA is 4x smaller.

### Auxiliary-Loss-Free Load Balancing

MoE routers concentrate too much work on a few experts. Standard fix: auxiliary loss term penalizing imbalance. DeepSeek-V3 uses per-expert bias terms adjusted during training: if expert is overloaded, decrease bias; underloaded, increase. No extra loss term. No hyperparameter to tune.

### MTP Module

D=1 module predicting token two positions ahead. 14B extra parameters (2.1% overhead). Provides denser training signal and free speculative-decoding draft with 80%+ acceptance.

### DualPipe

Bidirectional pipeline overlapping forward/backward compute with cross-node all-to-all. Recovers ~245k GPU-hours at 2,048-H800 scale.

### The Config

```
hidden_size: 7168
intermediate_size: 18432
moe_intermediate_size: 2048
num_hidden_layers: 61
first_k_dense_layers: 3          # first 3 layers dense MLP
num_attention_heads: 128
kv_lora_rank: 512                # MLA latent dimension
num_experts: 256                  # MoE experts per block
num_experts_per_tok: 8            # top-8 routing
shared_experts: 1                 # always-on shared expert
vocab_size: 129280
mtp_module: 1
```

### Parameter Accounting

- Embedding: 129280 * 7168 = ~0.93B
- 3 dense blocks: ~1.2B total
- 58 MoE blocks: ~461B total (attention ~144M + 256 experts each ~30M + 1 shared)
- MTP module: 14B
- Grand total: ~671B

Active per forward: ~26B core + 14B MTP (sometimes not run at inference) ≈ 37B.

18x sparsity ratio (5.5% active). DeepSeek-V3 is the sparsest frontier open MoE model.

### Where It Sits

| Model | Total | Active | Ratio | Attention |
|-------|------|-------|-------|-----------|
| Llama 3 70B | 70B | 70B | 100% | GQA |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA |
| Mixtral 8x22B | 141B | 39B | 27% | GQA |

## Build It

The code is a parameter calculator specialized to DeepSeek-V3's shape. Run it, compare to paper numbers, and experiment with variants (256 vs 512 experts, top-8 vs top-16, MLA rank 512 vs 1024).

## Use It

Run the calculator and examine:
- Total vs published 671B
- Active vs published 37B
- KV cache at 128k -- MLA vs GQA comparison
- Per-layer breakdown

## Ship It

This lesson produces `outputs/skill-deepseek-v3-reader.md` -- reads any DeepSeek-family model and produces component-by-component analysis.

## Exercises

1. Run the calculator and compare total parameter estimate to published 671B. Identify delta using paper Section 2.
2. Modify MLA rank to 256. Compute KV cache at 128k. What percentage reduction?
3. Compare (256 experts, top-8) to (512 experts, top-8). Total params grow, active stays same. What does extra capacity buy?
4. Read Section 2.1 on MLA. Explain in 3 sentences why K/V decompression matrices can be absorbed into subsequent matmul.
5. Compute memory savings of FP8 vs BF16 for 671B weights.

## Further Reading

- DeepSeek-AI, "DeepSeek-V3 Technical Report" (arXiv:2412.19437)
- DeepSeek-V3 HuggingFace model card
- DeepSeek-V2 paper (arXiv:2405.04434, introduced MLA)
- DeepSeek-R1 paper (arXiv:2501.12948)
- Native Sparse Attention (arXiv:2502.11089)
