# Open Models: Architecture Walkthroughs

> You built a GPT-2 Small from scratch. Frontier open models are the same family with five or six concrete changes. The math you already know covers 95% of them.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10, Lessons 04, 05, 12
**Time:** ~45 minutes

## Learning Objectives

- Read any model's config.json and explain every field
- Name the specific architectural change each model made vs GPT-2 and why
- Compute parameter count, KV cache size, and activation memory from config alone
- Pick the right model for a deployment target

## The Problem

You wrote 350 lines of numpy and had a GPT-2-shaped model. Llama 3 405B has a 200-page report. The skeleton -- embedding, transformer blocks, attention, MLP, norm, head -- is unchanged. A diff. This lesson shows exactly what changed from GPT-2, why, and what it cost.

## The Concept

### The Invariant Core

All autoregressive open models share: token embedding matrix, stack of N decoder blocks, final norm and linear head, causal mask, next-token cross-entropy loss.

### The Six Knobs

1. **RMSNorm.** LayerNorm subtracts mean and divides by std. RMSNorm keeps only the scale: `x / sqrt(mean(x^2) + eps) * gamma`. ~10% faster, matches quality. Every modern open model uses it.

2. **RoPE.** Learned position embeddings cannot extrapolate beyond training length. Rotary Position Embedding rotates Q and K vectors by an angle that is a deterministic function of position. With NTK/YaRN scaling, 8k-trained models stretch to 128k at inference.

3. **SwiGLU.** GPT-2's `gelu(xW1)W2` becomes `(xW1) * sigmoid(xW1) * xV`. Two parallel projections gated by Swish. Stronger perplexity per parameter. MLP hidden size adjusted to `8/3 * hidden`.

4. **Attention Head Sharing.** MHA -> GQA -> MQA -> MLA. GQA (Grouped-Query Attention) shares K,V across groups of Q heads. Llama 3 8B uses 32 Q heads, 8 KV heads (4x KV cache reduction). MLA (DeepSeek) compresses to a low-rank latent.

5. **Mixture of Experts.** K experts per block, router picks top-k per token. More total params, same active params. Mixtral 8x7B: 47B total, 13B active. DeepSeek-V3: 671B total, 37B active.

6. **Pre-norm.** Norm before each sublayer, not after. Strictly easier to train at depth.

### Model-by-Model Diff

| Model | Params | Active | Norm | Activation | Position | Attention | MoE |
|-------|--------|--------|------|-----------|----------|-----------|-----|
| GPT-2 Small | 124M | 124M | LayerNorm | GELU | Learned | MHA 12 | no |
| Llama 3 8B | 8B | 8B | RMSNorm | SwiGLU | RoPE | GQA 32/8 | no |
| Llama 3 70B | 70B | 70B | RMSNorm | SwiGLU | RoPE | GQA 64/8 | no |
| Mistral 7B | 7.2B | 7.2B | RMSNorm | SwiGLU | RoPE | GQA | no |
| Mixtral 8x7B | 47B | 13B | RMSNorm | SwiGLU | RoPE | GQA | 8 top-2 |
| Gemma 2 9B | 9B | 9B | RMSNorm | GeGLU | RoPE+slide | GQA | no |
| DeepSeek V3 | 671B | 37B | RMSNorm | SwiGLU | RoPE | MLA | 256 top-8 |

### Reading a config.json

```json
{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 131072,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5,
  "vocab_size": 128256
}
```

`hidden_size`: embedding dim. `intermediate_size`: MLP hidden (SwiGLU: ~3.5x hidden). `num_key_value_heads`: KV heads for GQA. `rope_theta`: RoPE base frequency (500k for long-context).

### Memory Budget

KV cache at max context: `2 * num_layers * num_kv_heads * head_dim * seq_len * 2` (BF16). Llama 3 8B at 128k: 17.2 GB -- larger than the 16 GB weights.

### When Each Model Wins

- Single 80GB GPU: Llama 3 8B, Mistral 7B
- Single node big capacity: Llama 3 70B, Qwen 2.5 72B
- Biggest open capability: DeepSeek V3
- Long-context: Llama 3, DeepSeek (MLA advantage)

## Build It

The code is a parameter calculator. Given any config.json, it prints parameter count by component, KV cache at max context, SwiGLU MLP ratio, and architecture verdict.

## Use It

Run on Llama 3 8B, Mistral 7B, Mixtral 8x7B, DeepSeek V3 configs. Compare breakdowns. Notice MoE models have total params dwarfing dense but active params often smaller.

## Ship It

This lesson produces `outputs/skill-open-model-picker.md` -- recommends model + quantization + inference stack for a deployment target.

## Exercises

1. Read Qwen 2.5 72B config from HuggingFace, compute total parameters, compare to HF-reported value.
2. Compute KV cache for Llama 3 405B at 128k in FP8 and BF16; calculate concurrent sequences on 8xH100.
3. Gemma 2 alternates full and sliding-window attention. Write KV cache math when half the layers use sliding window.
4. Find a recent frontier open model released after this lesson. Identify which knobs it picked.

## Further Reading

- Dubey et al., "The Llama 3 Herd of Models" (2024)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
- Su et al., "RoFormer: Enhanced Transformer with RoPE" (2021)
- Shazeer, "GLU Variants Improve Transformer" (2020)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
