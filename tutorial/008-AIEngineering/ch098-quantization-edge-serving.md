# Production Quantization, Cold Start & Edge

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch367): Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

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

---

## Part 2 (ch368): Cold Start Mitigation for Serverless LLMs

> A 20 GB model image takes 5-10 minutes (7B) to 20+ minutes (70B) to go from cold to serving. In a true serverless world, that is not a warm-up — it is an outage. Mitigations operate at five layers: pre-seeded node images, model streaming, GPU memory snapshots, warm pools, tiered loading, and live migration. Modal publishes 2-4s cold starts as a floor; Baseten 5-10s default, sub-second with pre-warming.

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## Learning Objectives

- Enumerate the five layers of cold-start mitigation and name one tool or pattern at each layer.
- Compute total cold-start time as a sum of (node provision) + (weights download) + (weights load into HBM) + (engine init) for a 70B model.
- Explain why live migration transfers input tokens (KB) not KV cache (GB) and what the penalty is.
- Name the warm-pool trade-off (pay for idle GPU or accept cold-start tail) and the SLA threshold at which `min_workers > 0` becomes mandatory.

## The Problem

Your serverless LLM endpoint scales to zero overnight. At 8 a.m. traffic spikes. The first request waits while: Karpenter provisions a GPU node (45-60s), container pulls 30 GB image (120-300s), engine loads weights into HBM (45-120s), vLLM initializes CUDA graphs (10-30s). Total: 220-510s before one token comes back.

## The Concept

### Layer 1 — pre-seeded node images (Bottlerocket)

On AWS, Bottlerocket's dual-volume architecture separates OS from data. Snapshot the data volume with your container image pre-pulled. New nodes boot with weights already on local NVMe.

### Layer 2 — model streaming (Run:ai Model Streamer)

Stream weights into GPU memory layer-by-layer and start processing as soon as the first transformer block is resident. Cuts weight-load time roughly in half.

### Layer 3 — GPU memory snapshots (Modal)

Take a checkpoint of GPU state after first load. Subsequent restarts deserialize directly into HBM — 10x faster.

### Layer 4 — warm pools (min_workers=1)

Keep one replica always ready. Cost is one GPU's hourly rate 24x7.

### Layer 5 — tiered loading (ServerlessLLM)

Storage hierarchy: NVMe → DRAM → HBM. 10-200x latency reduction versus naive disk-to-HBM.

### Live migration

Move input tokens (KB) to a destination that has the model loaded and recompute KV cache. Recomputation is cheaper than transferring GB of KV cache.

### Cold start anatomy for 70B

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total mitigated** | **~15s** | 22x reduction |

## Use It

`code/main.py` models cold-start with and without each mitigation.

```python
"""Cold-start mitigation path simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Phase:
    name: str
    raw_sec: float
    pre_seeded_sec: float
    streamer_sec: float
    snapshot_sec: float

PHASES_70B = [
    Phase("node provision",   50.0, 50.0,  50.0,  0.5),
    Phase("image pull",      180.0,  0.0, 180.0,  0.0),
    Phase("weights to HBM",   75.0, 75.0,  35.0,  0.0),
    Phase("engine init",      20.0, 20.0,  20.0,  2.0),
    Phase("first forward",     3.0,  3.0,   3.0,  0.5),
]

def total_for_stack(stack: set[str]) -> float:
    seconds = 0.0
    for phase in PHASES_70B:
        if "gpu_snapshot" in stack:
            seconds += phase.snapshot_sec
        elif "streamer" in stack and "pre_seeded" in stack:
            used = phase.pre_seeded_sec
            if phase.name == "weights to HBM":
                used = phase.streamer_sec
            seconds += used
        elif "pre_seeded" in stack:
            seconds += phase.pre_seeded_sec
        elif "streamer" in stack:
            seconds += phase.streamer_sec if phase.name == "weights to HBM" else phase.raw_sec
        else:
            seconds += phase.raw_sec
    return seconds

def report_stack(label: str, stack: set[str]) -> None:
    total = total_for_stack(stack)
    print(f"{label:20}  {total:6.1f} s  ({total/60:4.1f} min)  stack={sorted(stack) if stack else '{baseline}'}")

def main() -> None:
    print("=" * 80)
    print("COLD START MITIGATION — 70B model on fresh H100 node")
    print("=" * 80)
    report_stack("RAW", set())
    report_stack("+ PRE_SEEDED", {"pre_seeded"})
    report_stack("+ STREAMER", {"streamer"})
    report_stack("+ PRE_SEEDED + STREAMER", {"pre_seeded", "streamer"})
    report_stack("+ GPU_SNAPSHOT", {"gpu_snapshot"})

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-cold-start-planner.md`. Given SLA, model size, and traffic shape, picks which mitigations to stack.

## Exercises

1. Compute the break-even request rate above which a warm replica is cheaper than cold-start tax.
2. Deploy a 13B model with P99 TTFT SLA of 3s. Pick the minimum mitigation stack.
3. Bottlerocket pre-seeding eliminates image pull. Compute wall-clock for 70B if NVMe reads at 7 GB/s.
4. Argue both sides of snapshot PII risk and the mitigation (ephemeral snapshots, encryption, isolation).
5. Design a tiered warm-pool policy: paid users, trial users, batch workloads.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cold start | "the big pause" | Time from request to first token on a fresh replica |
| Warm pool | "always-on minimum" | `min_workers >= 1` to keep at least one replica ready |
| Pre-seeded image | "baked AMI" | Node image with container weights pre-resident |
| Model streamer | "streaming load" | Overlap weights I/O with compute setup |
| GPU snapshot | "checkpoint to HBM" | Serialize post-load GPU state; deserialize on restart |
| Tiered loading | "NVMe + DRAM + HBM" | Hierarchy of storage tiers; load on demand |
| Live migration | "move tokens" | Transfer input (KB), recompute KV on destination |

## Further Reading

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start)
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket)
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/10-cold-start-mitigation)

---

## Part 3 (ch369): Multi-Region LLM Serving and KV Cache Locality

> Round-robin load balancing is actively harmful for cached LLM inference. A request that does not land on the node holding its prefix pays full prefill cost — roughly 800 ms at P50 on a long prompt versus ~80 ms with a cache hit. In 2026 the production pattern is a cache-aware router (vLLM Router in Rust, llm-d router) that consumes KV-cache events and routes on prefix-hash match.

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Explain why round-robin load balancing breaks cached inference and quantify the TTFT penalty.
- Diagram a cache-aware router: inputs (KV-cache events), algorithm (prefix-hash match), tie-breaker (GPU utilization).
- Name the 32% DR failure driver for LLMs (missing tokenizer files / quantization configs) and state a three-file DR checklist.
- Distinguish commercial cross-region offerings (Bedrock CRI, GKE Multi-Cluster Gateway) from KV-aware routing.

## The Problem

Your service runs in us-east-1, us-west-2, and eu-west-1. You put an ALB in front with round-robin. Prefix cache hit rate drops to 8%. TTFT P50 triples. Round-robin is optimal for stateless services. LLM inference is stateful by design.

## The Concept

### Cache-aware routing

Request arrives with a prompt. Router hashes the prefix; it asks each replica "do you have this prefix cached?" Replicas publish KV-cache events. Router picks the replica with the match, falls through to GPU-util-based tie-breaker.

**vLLM Router** (Rust): subscribes to `kv.cache.block_added` events, maintains prefix-hash → replica index, routes with O(1) lookup.

### Numbers

TTFT P50 on a 2K-token prompt, Llama 3.3 70B FP8, H100:
- Cache hit: ~80 ms.
- Cache miss (cold prefill): ~800 ms.

10x gap.

### Cross-region has a new constraint — network latency

Inter-region RTT:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

GORGO makes `prefill_time + network_latency` the explicit objective.

### DR hygiene — the 32% missing-files problem

32% of LLM DR failures happen because teams forgot:
- `tokenizer.json` or `tokenizer.model`
- Quantization configs
- Model-specific configs (RoPE scaling, attention masks, chat templates)
- Engine config

Fix: three-file minimum DR manifest — HF model repo + engine config + deployment manifest.

## Use It

`code/main.py` simulates three routing strategies.

```python
"""Cache-aware multi-region router simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass, field
import random
import statistics

REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
REPLICAS_PER_REGION = 4
CACHE_HIT_MS = 80
CACHE_MISS_MS = 800
CROSSREGION_RTT = {
    ("us-east-1", "us-west-2"): 65,
    ("us-east-1", "eu-west-1"): 75,
    ("us-west-2", "eu-west-1"): 130,
}

def rtt(a: str, b: str) -> int:
    if a == b: return 0
    key = (a, b) if (a, b) in CROSSREGION_RTT else (b, a)
    return CROSSREGION_RTT.get(key, 200)

@dataclass
class Replica:
    region: str
    idx: int
    prefix_cache: set = field(default_factory=set)
    queue_depth: int = 0

@dataclass
class Request:
    origin_region: str
    prefix_hash: str
    served_by: Replica | None = None
    ttft_ms: float = 0
    crossregion: bool = False

def make_workload(n: int = 1000, seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    hot_prefixes = [f"prefix_{i}" for i in range(40)]
    return [Request(rng.choice(REGIONS), rng.choice(hot_prefixes)) for _ in range(n)]

def simulate(strategy: str, reqs: list[Request]) -> dict:
    replicas = [Replica(r, i) for r in REGIONS for i in range(REPLICAS_PER_REGION)]
    rng = random.Random(11)
    hits = 0
    ttfts: list[float] = []
    crossregion_count = 0
    for i, r in enumerate(reqs):
        chosen: Replica | None = None
        if strategy == "ROUND_ROBIN":
            chosen = replicas[i % len(replicas)]
        elif strategy == "REGIONAL":
            local = [rep for rep in replicas if rep.region == r.origin_region]
            matches = [rep for rep in local if r.prefix_hash in rep.prefix_cache]
            chosen = min(matches, key=lambda x: x.queue_depth) if matches else min(local, key=lambda x: x.queue_depth)
        elif strategy == "GLOBAL":
            matches = [rep for rep in replicas if r.prefix_hash in rep.prefix_cache]
            best_cost = float("inf")
            for rep in matches:
                c = CACHE_HIT_MS + rtt(r.origin_region, rep.region)
                if c < best_cost:
                    best_cost = c
                    chosen = rep
            if chosen is None or best_cost > CACHE_MISS_MS:
                local = [rep for rep in replicas if rep.region == r.origin_region]
                chosen = min(local, key=lambda x: x.queue_depth)
        hit = r.prefix_hash in chosen.prefix_cache
        if hit:
            hits += 1
            r.ttft_ms = CACHE_HIT_MS + rtt(r.origin_region, chosen.region)
        else:
            r.ttft_ms = CACHE_MISS_MS + rtt(r.origin_region, chosen.region)
            chosen.prefix_cache.add(r.prefix_hash)
            if len(chosen.prefix_cache) > 12:
                chosen.prefix_cache.pop()
        ttfts.append(r.ttft_ms)
        if chosen.region != r.origin_region:
            crossregion_count += 1
    ttfts.sort()
    return {"strategy": strategy, "hit_rate": hits / len(reqs), "mean_ttft": statistics.mean(ttfts),
            "p50_ttft": ttfts[len(ttfts) // 2], "p99_ttft": ttfts[int(len(ttfts) * 0.99) - 1],
            "crossregion": crossregion_count}

def main() -> None:
    print("=" * 80)
    print("MULTI-REGION LLM ROUTING — three strategies, 1000 requests")
    print("=" * 80)
    base = make_workload()
    for strategy in ("ROUND_ROBIN", "REGIONAL", "GLOBAL"):
        reqs = [Request(origin_region=r.origin_region, prefix_hash=r.prefix_hash) for r in base]
        r = simulate(strategy, reqs)
        print(f"{r['strategy']:13}  hit={r['hit_rate']*100:5.1f}%  mean={r['mean_ttft']:5.0f}ms  P50={r['p50_ttft']:5.0f}ms  P99={r['p99_ttft']:5.0f}ms  cross={r['crossregion']:4}")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-multi-region-router.md`. Given regions, residency constraints, and SLA, designs a routing plan.

## Exercises

1. At what prompt length does cross-region routing beat local-only, given 75 ms RTT?
2. Cache hit rate drops from 70% to 12%. Diagnose three possible causes.
3. Design a DR manifest for a 70B AWQ model in vLLM with 5 LoRA adapters.
4. Argue whether Bedrock cross-region inference is enough for fintech with strict TTFT SLOs.
5. Paris-origin request matches a prefix in us-east-1. Do you route it? Write the policy.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cache-aware routing | "smart LB" | Route on prefix-hash match to KV-cache-holding replica |
| GORGO | "cross-region routing research" | arXiv 2602.11688; network latency as explicit term |
| Cross-region inference | "Bedrock CRI" | AWS product; availability failover, not TTFT awareness |
| DR manifest | "the backup list" | Every file needed to restore — not just weights |
| Data residency | "GDPR boundary" | Legal constraint on which region sees user data |

## Further Reading

- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/11-multi-region-kv-locality)

---

## Part 4 (ch370): Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

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
