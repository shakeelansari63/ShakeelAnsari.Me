# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> The core edge constraint is memory bandwidth, not compute. Mobile DRAM sits at 50-90 GB/s; datacenter HBM3 clears 2-3 TB/s — a 30-50x gap. In 2026 the landscape splits four ways. Apple M4/A18 Neural Engine peaks at 38 TOPS with unified memory. Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon hits 45 TOPS. WebGPU + WebLLM runs Llama 3.1 8B (Q4) at ~41 tok/s on M3 Max. NVIDIA Jetson Orin Nano Super fits Llama 3.2 3B / Phi-3.

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## Learning Objectives

- Explain why mobile LLM inference is memory-bandwidth-bound and compute is secondary.
- Enumerate the four edge targets and match each to a use case.
- Name the 2026 WebGPU coverage gap and the Safari iOS 26 landing.
- Pick a quantization format per target.

## The Problem

On a MacBook Pro M3 Max, Llama 3.1 8B Q4 runs at ~55 tok/s. On an iPhone 16 Pro, the same model runs at 3 tok/s. The throughput variance is the bandwidth gap times the quantization format times whether the NPU is accessible.

## The Concept

### Bandwidth is the real ceiling

One 7B model in Q4 is 3.5 GB. Reading 3.5 GB at 50 GB/s takes 70 ms — ceiling of ~14 tok/s. Datacenter HBM3 at 3 TB/s clears the same 3.5 GB in 1.2 ms — ceiling is 830 tok/s.

### Apple Neural Engine

Up to 38 TOPS. Unified memory. Access via Core ML + `.mlmodel` compiled models. Best practical path: Core ML with INT4 weights + FP16 activations.

### Qualcomm Hexagon

Up to 45 TOPS. QNN SDK and AI Hub provide conversion from PyTorch/ONNX.

### WebGPU + WebLLM

Llama 3.1 8B Q4 at ~41 tok/s on M3 Max. 2026 coverage: ~70-75% mobile. Chrome Android v121+, Safari iOS 26 GA.

### NVIDIA Jetson

Orin Nano Super (8GB): Llama 3.2 3B, Phi-3. AGX Orin: gpt-oss-20b at ~40 tok/s via vLLM. Thor / T4000: 2x AGX Orin, EAGLE-3 and NVFP4 supported.

### Quantization per target

| Target | Format |
|--------|--------|
| Apple ANE | INT4 weights + FP16 activations |
| Qualcomm Hexagon | QNN INT8 / INT4 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 |
| Jetson AGX / Thor | NVFP4 + FP8 KV |

## Use It

`code/main.py` computes theoretical decode throughput ceilings from bandwidth-bound math.

```python
"""Edge-inference bandwidth-bound decode simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Target:
    name: str
    bandwidth_gb_s: float
    observed_toks_per_s_llama8b_q4: float | None
    notes: str

TARGETS = [
    Target("Datacenter H100 HBM3",  3350, 170,  "reference ceiling"),
    Target("Jetson AGX Orin",        205,  45,  "edge-datacenter bridge"),
    Target("Apple M3 Max",           400,  55,  "unified memory MPS"),
    Target("Apple M4 (MacBook Air)", 120,  25,  "consumer laptop"),
    Target("Apple A18 (iPhone 16)",   60,   8,  "phone with ANE"),
    Target("Snapdragon 8 Gen 3",      77,   7,  "mid/high Android"),
    Target("Snapdragon X Elite",     135,  22,  "Windows ARM laptop"),
    Target("WebGPU on M3 Max",       400,  41,  "browser penalty ~25%"),
    Target("WebGPU on Pixel 9",       77,   6,  "mobile browser Chrome 121+"),
]

def ceiling(target: Target, model_gb: float) -> float:
    return 1 / (model_gb / target.bandwidth_gb_s)

def main() -> None:
    model_name = "Llama 3.1 8B Q4"
    model_gb = 4.7
    print("=" * 95)
    print(f"EDGE DECODE CEILING — {model_name} ({model_gb:.1f} GB)")
    print("=" * 95)
    for t in TARGETS:
        c = ceiling(t, model_gb)
        obs = t.observed_toks_per_s_llama8b_q4
        eff = f"{obs / c * 100:3.0f}%" if obs else "   -"
        obs_display = f"{obs:>8.0f}" if obs else "        -"
        print(f"{t.name:26}  {t.bandwidth_gb_s:8.0f} GB/s  ceiling={c:10.1f}  observed={obs_display}  eff={eff}  {t.notes}")
    print("\nQuantization impact on iPhone 16:")
    for name, size in [("BF16", 18.8), ("INT8", 9.4), ("Q4 GGUF", 4.7), ("Q3 GGUF", 3.6)]:
        c = 1 / (size / 60.0)
        print(f"  {name:8}  model={size:5.1f} GB  ceiling={c:6.1f} tok/s")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-edge-target-picker.md`. Given platform, model, and latency/memory budget, picks a quantization format and conversion pipeline.

## Exercises

1. For a 7B model in Q4 on Snapdragon 8 Gen 3 (~77 GB/s), compute the decode ceiling.
2. WebGPU on Android requires Chrome v121+. Design fallback for older browsers.
3. iOS app needs 4K-context streaming. Which model/format keeps under 4 GB on iPhone 16?
4. Jetson AGX Orin runs gpt-oss-20b at 40 tok/s. Jetson Nano fits only 3B. Unify the inference stack?
5. Argue whether WebLLM is production-ready in 2026.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | On-device NPU in M-series and A-series; unified memory |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU; QNN SDK for access |
| WebGPU | "browser GPU" | W3C-standardized browser GPU API |
| WebLLM | "browser LLM runtime" | MLC-LLM project; Apache 2.0; OpenAI-compatible JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | "edge TensorRT" | 2026 edge port of TensorRT-LLM; EAGLE-3 + NVFP4 |

## Further Reading

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/)
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/)
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2)
- [Apple Core ML](https://developer.apple.com/documentation/coreml)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/12-edge-inference)
