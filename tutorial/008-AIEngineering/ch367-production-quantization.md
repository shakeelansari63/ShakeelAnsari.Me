# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Quantization format is not a universal choice — it is a function of hardware, serving engine, and workload. GGUF Q4_K_M or Q5_K_M owns CPU and edge. GPTQ wins inside vLLM when you need multi-LoRA on the same base. AWQ with Marlin-AWQ kernels delivers ~741 tok/s on a 7B class model with the best Pass@1 at INT4 — the 2026 default for datacenter production. FP8 stays the middle ground on Hopper, Ada, and Blackwell. NVFP4 and MXFP4 are aggressive and require per-block validation.

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- Name the six production quantization formats and their sweet spots in 2026.
- Pick a format given hardware, engine, and workload.
- Compute the weight memory saved and the KV cache left untouched for a chosen format.
- Name the calibration-dataset pitfall that degrades quantized models on domain traffic.

## The Problem

An FP16 70B model is 140 GB of weights. Quantize to INT4 (AWQ or GPTQ) and the model is 35 GB — fits in one H100 with room for KV cache. But quantization is not free. Aggressive quantization degrades quality, especially on reasoning-heavy tasks.

## The Concept

### The six formats

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptops | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA on vLLM | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### The calibration trap

AWQ and GPTQ require a calibration dataset. For domain models (code, medical, legal), calibrating on generic web text leads to wrong decisions about which weights to protect. Calibrate on in-domain data.

### The KV cache trap

AWQ shrinks weights to 4 bits. KV cache is separate. For a 70B model at 128 concurrent × 2k context: weights ~35 GB, KV cache ~20 GB, activations ~5 GB. Total ~60 GB — fits on H100 80GB.

### Picking guide

- CPU/edge: GGUF Q4_K_M.
- GPU, routine chat, no LoRA: AWQ.
- GPU, multi-LoRA: GPTQ with Marlin.
- Reasoning workload: FP8.
- Blackwell, validated quality: NVFP4 + FP8 KV.

## Use It

`code/main.py` computes memory footprint and relative throughput across six formats.

```python
"""Toy quantization memory and throughput calculator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Format:
    name: str
    weight_bits: float
    kv_bits: float
    engine: str
    notes: str

FORMATS = [
    Format("BF16 baseline (vLLM)",       16, 16, "vLLM",     "reference"),
    Format("GGUF Q5_K_M (llama.cpp)",     5, 16, "llama.cpp", "CPU/edge"),
    Format("GGUF Q4_K_M (llama.cpp)",     4, 16, "llama.cpp", "CPU/edge, default"),
    Format("GPTQ-Int4 + Marlin (vLLM)",   4, 16, "vLLM",     "multi-LoRA support"),
    Format("AWQ-Int4 + Marlin (vLLM)",    4, 16, "vLLM",     "best Pass@1 at INT4"),
    Format("FP8 (vLLM / TRT-LLM)",        8,  8, "multi",    "safe default reasoning"),
    Format("NVFP4 + FP8 KV (TRT-LLM)",    4,  8, "TRT-LLM",  "Blackwell aggressive"),
]

def memory_breakdown(params_b: float, fmt: Format, concurrency: int = 128, ctx: int = 2048) -> dict:
    weight_gb = params_b * fmt.weight_bits / 8
    layers = 64 * (params_b / 70.0)**0.5
    kv_heads = 8
    head_dim = 128
    per_seq_kv_gb = layers * 2 * kv_heads * head_dim * ctx * (fmt.kv_bits / 8) / 1e9
    kv_total = per_seq_kv_gb * concurrency
    activations_gb = 0.05 * params_b
    return {"weight": weight_gb, "kv": kv_total, "act": activations_gb, "total": weight_gb + kv_total + activations_gb}

def relative_throughput(fmt: Format) -> float:
    return 16 / fmt.weight_bits

def gpu_check(total_gb: float) -> str:
    if total_gb <= 80: return "H100 80GB"
    if total_gb <= 141: return "H200 141GB"
    if total_gb <= 192: return "B200 192GB"
    return "MULTI-GPU"

def print_scenario(params_b: float, concurrency: int, ctx: int) -> None:
    print(f"Model: {params_b}B  |  concurrency {concurrency}  |  ctx {ctx}")
    print("-" * 98)
    print(f"{'format':36} {'W GB':>7} {'KV GB':>7} {'Act GB':>7} {'Total':>7} {'fits on':>14} {'rel tput':>10}")
    for f in FORMATS:
        m = memory_breakdown(params_b, f, concurrency, ctx)
        tput = relative_throughput(f)
        print(f"{f.name:36} {m['weight']:7.1f} {m['kv']:7.1f} {m['act']:7.1f} {m['total']:7.1f} {gpu_check(m['total']):>14} {tput:10.2f}x")
    print()

def main() -> None:
    print("=" * 98)
    print("TOY QUANTIZATION CALCULATOR — memory and relative throughput by format")
    print("=" * 98)
    print_scenario(params_b=7, concurrency=128, ctx=2048)
    print_scenario(params_b=70, concurrency=128, ctx=2048)
    print_scenario(params_b=70, concurrency=256, ctx=8192)
    print_scenario(params_b=405, concurrency=128, ctx=2048)

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-quantization-picker.md`. Given hardware, model size, workload type, and quality tolerance, picks a format.

## Exercises

1. For 70B at 128 concurrent with 2k context, compute total HBM for each format. Which fits on one H100 80GB?
2. You have a 7B coding model. Pick a format and justify.
3. Compute calibration-dataset size needed for AWQ on a medical domain model.
4. Read the Marlin-AWQ kernel paper. Why does AWQ hit 741 tok/s on 7B while GPTQ hits ~712?
5. When does it make sense to combine AWQ weights with FP8 KV cache vs keeping KV at BF16?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GGUF | "llama.cpp format" | File format bundling K-quant variants; CPU/edge default |
| GPTQ | "gee pee tee q" | Post-train INT4 with calibration; supports LoRA in vLLM |
| AWQ | "a w q" | Activation-aware INT4; Marlin kernels; best Pass@1 at INT4 |
| FP8 | "eight-bit float" | Safe precision default on Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "microscaling four" | Blackwell 4-bit FP with per-block scale factors |
| Calibration dataset | "cal data" | Input text used to pick quantization parameters; must match domain |

## Further Reading

- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978)
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/09-production-quantization)
