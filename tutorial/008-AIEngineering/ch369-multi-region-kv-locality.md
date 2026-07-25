# Multi-Region LLM Serving and KV Cache Locality

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
