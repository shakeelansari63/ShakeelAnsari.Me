# TensorRT-LLM on Blackwell with FP8 and NVFP4

> TensorRT-LLM is NVIDIA-only but it wins on Blackwell. On GB200 NVL72 with Dynamo orchestration, SemiAnalysis InferenceX measured $0.012 per million tokens on a 120B model in Q1-Q2 2026, against $0.09/M on H100 + vLLM — a 7x economic gap. The stack is three floating-point regimes compounded: FP8 stays critical for KV cache and attention kernels because it has the dynamic range they need; NVFP4 handles weights and activations; multi-token prediction and disaggregated prefill/decode add another 2-3x on top.

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## Learning Objectives

- Explain why FP8 stays critical for KV cache and attention even when weights are in NVFP4.
- Compute the HBM footprint of a frontier model under BF16, FP8, and NVFP4 and reason about where the savings come from.
- Name the Blackwell-specific features TRT-LLM exploits (day-0 FP4, MTP, disaggregated serving, all-to-all primitives).
- Decide when TRT-LLM's NVIDIA-lock is worth the 7x cost gap vs vLLM on Hopper.

## The Problem

On Hopper with vLLM, a 120B MoE runs at ~$0.09 per million tokens. On Blackwell with TRT-LLM + Dynamo, the same model runs at ~$0.012 — 7x cheaper. Some of that gap is hardware (Blackwell is 11-15x per-GPU LLM throughput vs Hopper). Some is the stack: FP4 weights, MTP draft, disaggregated prefill/decode, and NVLink 5 all-to-all.

## The Concept

### Why FP8 is still the floor for KV cache

KV cache needs FP8 because it stores attention keys and values that span a wide dynamic range. Quantizing KV to FP4 causes catastrophic accuracy loss. NVFP4 applies to weights and activations. The typical Blackwell config:

- Weights: NVFP4
- Activations: NVFP4
- KV cache: FP8
- Attention accumulator: FP32

### Blackwell-specific primitives TRT-LLM uses

- **Day-0 FP4 weights**: model providers ship FP4 weights directly; TRT-LLM loads without post-training conversion.
- **Multi-token prediction (MTP)**: integrated speculative-decoding draft.
- **Disaggregated serving**: prefill and decode on separate GPU pools.
- **All-to-all communication**: NVLink 5 cuts MoE expert communication latency by 3x.
- **NVFP4 + MXFP8 microscaling**: hardware-accelerated on Blackwell Tensor Cores.

### Numbers

- HGX B200: $0.02/M tokens on GPT-OSS-120B via TRT-LLM.
- GB200 NVL72: $0.012/M tokens via Dynamo.
- H100 + vLLM: ≈ $0.09/M tokens.
- 11-15x per-GPU LLM throughput, Blackwell vs Hopper.

### What FP4 costs in quality

NVFP4 degrades on reasoning-heavy workloads. Per-block calibration mitigates but does not eliminate. Always validate on your eval set.

### 2026 practical recipe

Migrate cost-dominant workloads to Blackwell + TRT-LLM + Dynamo. Keep experimentation tier on H100 + vLLM.

## Use It

`code/main.py` computes HBM footprint and $/M-tokens across three stacks.

```python
"""Toy Blackwell + TRT-LLM economics calculator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Stack:
    name: str
    hbm_gb: int
    hbm_bw_tbs: float
    weight_bits: float
    kv_bits: float
    mtp_factor: float
    disagg_factor: float
    price_per_gpu_hour: float

STACKS = [
    Stack("H100 + BF16 + vLLM",           80, 3.35,  16, 16, 1.0,  1.0,  2.50),
    Stack("H100 + FP8 + vLLM",            80, 3.35,   8,  8, 1.0,  1.0,  2.50),
    Stack("H200 + FP8 + vLLM",           141, 4.80,   8,  8, 1.0,  1.0,  3.50),
    Stack("B200 + NVFP4 + FP8 + TRT-LLM", 192, 8.00,   4,  8, 1.8,  1.6,  4.80),
    Stack("GB200 NVL72 + TRT-LLM + Dyn", 192, 8.00,   4,  8, 1.8,  2.5,  6.20),
]

def hbm_footprint_gb(params_b: float, active_b: float, seq_len: int, stack: Stack) -> tuple[float, float]:
    weight_gb = params_b * stack.weight_bits / 8
    layers = 64 * (active_b / 35.0)**0.5
    kv_heads = 8
    head_dim = 128
    kv_gb = layers * 2 * kv_heads * head_dim * seq_len * (stack.kv_bits / 8) / 1e9
    return weight_gb, kv_gb

def decode_throughput(active_b: float, stack: Stack) -> float:
    bytes_per_token = active_b * 1e9 * stack.weight_bits / 8
    raw_tokens_per_s = stack.hbm_bw_tbs * 1e12 / bytes_per_token
    return raw_tokens_per_s * stack.mtp_factor * stack.disagg_factor

def cost_per_million_tokens(active_b: float, stack: Stack) -> float:
    tps = decode_throughput(active_b, stack)
    tokens_per_hour = tps * 3600
    return stack.price_per_gpu_hour / tokens_per_hour * 1e6

def print_stack(params_b: float, active_b: float, seq_len: int = 8192) -> None:
    print(f"Model: {params_b}B total, {active_b}B active, {seq_len:,} tokens context")
    print("-" * 90)
    print(f"{'stack':40} {'W GB':>7} {'KV GB':>7} {'tok/s':>9} {'$/M tok':>10}")
    for s in STACKS:
        w, kv = hbm_footprint_gb(params_b, active_b, seq_len, s)
        tps = decode_throughput(active_b, s)
        cost = cost_per_million_tokens(active_b, s)
        print(f"{s.name:40} {w:7.1f} {kv:7.2f} {tps:9.0f} {cost:10.4f}")

def main() -> None:
    print("=" * 90)
    print("TOY BLACKWELL + TRT-LLM ECONOMICS — memory-bandwidth-limited decode")
    print("=" * 90)
    print_stack(70, 70)
    print_stack(120, 36)
    print_stack(405, 405)
    print_stack(671, 37)

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-trtllm-blackwell-advisor.md`. Given workload, model size, and annual token volume, decides if Blackwell + TRT-LLM is worth the NVIDIA-lock.

## Exercises

1. Run `code/main.py`. On a 120B MoE with 30% active parameters, compute throughput on each stack.
2. Customer spends $2M/year on H100 + vLLM. Break-even for Blackwell migration in 12 months?
3. Accuracy drops 3 points on MATH after NVFP4 conversion. Name two recovery paths.
4. Read MLPerf v6.0 inference results. Which task has the smallest Blackwell-over-Hopper gap?
5. Compute HBM needed for 405B at NVFP4 + FP8 KV at 128k context.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| FP8 | "eight-bit float" | 8-bit floating point; used for KV cache and attention |
| NVFP4 | "four-bit micro" | NVIDIA's 4-bit microscaling FP format; weights and activations on Blackwell |
| MTP | "multi-token prediction" | TRT-LLM's integrated speculative-decoding draft |
| Disaggregated serving | "split prefill/decode" | Prefill and decode on separate GPU pools |
| All-to-all | "MoE expert comm" | Communication pattern routing tokens to expert GPUs |

## Further Reading

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/)
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/)
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/07-tensorrt-llm-blackwell)
