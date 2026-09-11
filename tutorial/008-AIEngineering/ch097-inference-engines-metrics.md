# EAGLE-3, SGLang, TensorRT & Goodput Metrics

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch363): EAGLE-3 Speculative Decoding in Production

> Speculative decoding pairs a fast draft model with the target model. The draft proposes K tokens; the target verifies in a single forward; accepted tokens are free. In 2026, EAGLE-3 is the production-grade variant — it trains a draft head on the target model's hidden states rather than on raw tokens, pushing acceptance rate alpha into the 0.6-0.8 band on general chat. If alpha drops below ~0.55, speculative decoding is net negative at high concurrency.

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## Learning Objectives

- Name the three generations of speculative decoding and explain what EAGLE-3 changes from EAGLE-2 and from a classic draft model.
- Define acceptance rate alpha, compute expected speedup from alpha and K (draft length), and identify the break-even alpha for your target concurrency.
- Explain why speculative decoding is opt-in (not default) in vLLM 2026 and why turning it on without measuring alpha is a production anti-pattern.
- Write a measurement plan: which benchmark, which prompt distribution, which concurrency point, which metric to gate on.

## The Problem

Decode is memory-bound. On an H100 running Llama 3.3 70B FP8, each decoded token reads ~140 GB/s of weights and emits one token. The GPU compute is almost idle during decode — the bottleneck is HBM bandwidth.

Speculative decoding exploits the gap. Generate K candidate tokens with a cheap draft model, then ask the target model to verify all K in a single forward pass. Each verified token is effectively free.

The classic draft-model approach uses a smaller model of the same family (Llama 3.2 1B drafting for Llama 3.3 70B). Acceptance rate is mediocre — the smaller model distribution diverges from the target. EAGLE-3 trains a light draft head directly on the target model's internal states, pushing alpha from 0.4 with draft-model to 0.6-0.8.

## The Concept

### What speculative decoding actually buys

Without spec decode, per-token cost is one target forward. With spec decode at draft length K and acceptance alpha, expected tokens per target forward is `1 + K * alpha`. The speedup is `(1 + K * alpha) / (1 + epsilon)` where epsilon is draft-plus-verify overhead. For K=5, alpha=0.7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`.

### Why alpha is the only metric that matters

Rejected tokens do not disappear — they force a second target forward for the first rejected token. At high concurrency (say 256 concurrent), the decode batch is already large enough that the memory-bandwidth gap shrinks. Below alpha 0.55 on most 2026 hardware, spec decode is net negative.

Alpha varies by workload. On ShareGPT-style general chat, EAGLE-3 hits 0.6-0.8. On domain-specific traffic (code, medical, legal) the draft head trained on general data drops to 0.4-0.6.

### EAGLE generations at a glance

- **Classic draft model**: small model of same family. Alpha 0.3-0.5.
- **EAGLE-1 (2024)**: single draft head on target hidden states. Alpha ~0.5-0.6.
- **EAGLE-2 (2025)**: adaptive draft length and tree-based drafts. Alpha ~0.6-0.7.
- **EAGLE-3 (2025-2026)**: draft head on multiple target layers. Alpha ~0.6-0.8.

### The 2026 production recipe

1. Ship target model plain. Measure baseline.
2. Enable EAGLE-3 draft via vLLM `speculative_config`.
3. Log acceptance rate alpha (`spec_decode_metrics.accepted_tokens_per_request`).
4. If alpha < 0.55, disable or train a domain-specific draft.
5. Confirm P99 ITL did not get worse.

### Break-even math

Expected speedup: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. At high concurrency effective alpha_breakeven climbs to ~0.45-0.55.

## Use It

`code/main.py` simulates decode with and without speculative decoding across alpha values and draft lengths K.

```python
"""Toy speculative-decoding analyzer — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass
import random
import statistics

@dataclass
class SpecPoint:
    alpha: float
    k: int
    verify_overhead: float
    concurrency: int

def expected_speedup(p: SpecPoint) -> float:
    effective_overhead = p.verify_overhead * (1 + p.concurrency / 256)
    tokens_per_target = 1 + p.k * p.alpha
    cost_per_target = 1 + effective_overhead
    return tokens_per_target / cost_per_target

def breakeven_alpha(k: int, verify_overhead: float, concurrency: int) -> float:
    effective_overhead = verify_overhead * (1 + concurrency / 256)
    return effective_overhead / k

def simulate_tail(p: SpecPoint, n_tokens: int = 1000, seed: int = 3) -> tuple[float, float]:
    rng = random.Random(seed)
    base_target_ms = 8.0
    effective_overhead = p.verify_overhead * (1 + p.concurrency / 256)
    verify_ms = base_target_ms * (1 + effective_overhead)
    reroll_ms = base_target_ms
    latencies: list[float] = []
    tokens_emitted = 0
    while tokens_emitted < n_tokens:
        accepted = 0
        for _ in range(p.k):
            if rng.random() < p.alpha:
                accepted += 1
            else:
                break
        batch_lat = verify_ms + (reroll_ms if accepted < p.k else 0)
        batch_tokens = max(1, accepted + 1)
        per_tok = batch_lat / batch_tokens
        for _ in range(batch_tokens):
            jitter = rng.gauss(0, per_tok * 0.1)
            latencies.append(max(0.1, per_tok + jitter))
            tokens_emitted += 1
            if tokens_emitted >= n_tokens:
                break
    latencies.sort()
    p99 = latencies[int(0.99 * len(latencies)) - 1]
    return statistics.mean(latencies), p99

def plain_tail(concurrency: int, n_tokens: int = 1000, seed: int = 5) -> tuple[float, float]:
    rng = random.Random(seed)
    base = 8.0 * (1 + concurrency / 512)
    lats = [max(0.1, base + rng.gauss(0, base * 0.08)) for _ in range(n_tokens)]
    lats.sort()
    return statistics.mean(lats), lats[int(0.99 * len(lats)) - 1]

def main() -> None:
    print("=" * 80)
    print("TOY EAGLE-3 SPECULATIVE-DECODING ANALYZER")
    print("=" * 80)
    base_overhead = 0.15
    k = 5
    for concurrency in [32, 128, 256]:
        be = breakeven_alpha(k, base_overhead, concurrency)
        plain_mean, plain_p99 = plain_tail(concurrency)
        rows = []
        for alpha in [0.30, 0.45, 0.55, 0.70, 0.80]:
            p = SpecPoint(alpha=alpha, k=k, verify_overhead=base_overhead, concurrency=concurrency)
            s = expected_speedup(p)
            mean_ms, p99_ms = simulate_tail(p)
            delta = p99_ms - plain_p99
            rows.append((f"alpha={alpha:.2f} conc={concurrency}", s, be, mean_ms, p99_ms, delta))
        print(f"  --- concurrency {concurrency} ---  plain P99 = {plain_p99:.2f} ms")
        print(f"{'config':28} {'speedup':>8} {'be_alpha':>10} {'mean_ms':>10} {'p99_ms':>10}")
        for label, speedup, be_alpha, mean, p99, delta_p99 in rows:
            tag = "  OK" if delta_p99 <= 0 else "  TAIL"
            print(f"{label:28} {speedup:8.2f} {be_alpha:10.3f} {mean:10.2f} {p99:10.2f}{tag}")
        print()

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-eagle3-rollout.md`. Given target model and traffic, produces a staged EAGLE-3 rollout plan.

## Exercises

1. Run `code/main.py`. At K=5, what alpha do you need for a 2x speedup?
2. Blended alpha 70% general chat (alpha 0.7) + 30% code (alpha 0.4). Is spec decode net-positive?
3. Read vLLM `speculative_config` docs. Name the three modes and which is compatible with chunked prefill.
4. Mean ITL drops 25% but P99 ITL goes up 15%. Diagnose and propose a mitigation.
5. Compute the memory cost of EAGLE-3 draft head for Llama 3.3 70B vs Llama 3.2 1B classic draft.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speculative decoding | "draft plus verify" | Propose K tokens with a cheap model, verify all K in one target forward |
| Acceptance rate alpha | "spec accept rate" | Fraction of draft tokens accepted by the target |
| Draft length K | "spec k" | How many tokens the draft proposes per target forward; typical 4-8 |
| EAGLE-3 | "latest EAGLE" | 2025-2026 variant; trains draft head on multiple target layers; alpha 0.6-0.8 |
| Break-even alpha | "no-op alpha" | Alpha at which spec decode gives zero speedup |
| Rejected-draft two-pass | "reroll cost" | Two target forwards when drafts reject; drives P99 tail |

## Further Reading

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077)
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/05-eagle3-speculative-decoding)

---

## Part 2 (ch364): SGLang and RadixAttention for Prefix-Heavy Workloads

> SGLang treats the KV cache as a first-class, reusable resource stored in a radix tree. Where vLLM schedules requests FCFS, SGLang's cache-aware scheduler prioritizes requests with longer shared prefixes — effectively a depth-first radix traversal so hot branches stay resident in HBM. On Llama 3.1 8B with ShareGPT-like 1K prompts, SGLang hits ~16,200 tok/s to vLLM's ~12,500, a ~29% edge. On prefix-heavy RAG workloads the advantage reaches 6.4x.

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## Learning Objectives

- Diagram RadixAttention: how prefixes are stored in a radix tree and how KV blocks are shared across sequences rooted at the same branch.
- Explain cache-aware scheduling and why FCFS is wrong for prefix-heavy traffic.
- Compute expected speedup for a workload given prefix-cache hit rate and prompt length distribution.
- Name the prompt-ordering discipline that makes the 6.4x number real vs a lost upside.

## The Problem

Classic serving treats each request's prompt as opaque. Even when 5,000 RAG requests all start with the same 2,000-token system prompt plus same retrieval preamble, vLLM prefills that 2,000-token prefix 5,000 times. The GPU does the same work over and over.

RadixAttention does exactly this. Tokens are indexed in a radix tree; each node owns KV blocks for the token sequence on its path from root. A new request walks the tree: any node whose token matches re-uses that node's KV blocks. Prefill cost becomes proportional to the "new" suffix, not the full prompt.

## The Concept

### The radix tree as a KV index

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

A new request with system prompt + "Context: <doc A>" + "Question: Carol" walks: system prefix matches (124 blocks reused), doc-A branch matches (31 blocks reused), then allocates fresh blocks only for "Question: Carol" (4 blocks). ~40x savings on prefill.

### Cache-aware scheduling

1. **Depth-first dispatch** — prefer requests rooted at the same branch as the current running set.
2. **LRU at branch level, not block level** — evict whole branches rather than individual blocks.

### Benchmark numbers

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% edge).
- Prefix-heavy RAG: up to 6.4x on SGLang.
- Voice cloning workloads: 86.4% prefix-cache hit rate.
- Deployed on 400,000+ GPUs in 2026.

### The ordering gotcha

The 6.4x number relies on consistent prompt-template ordering. If your client constructs prompts with dynamic interleaved content, the tree cannot find the shared prefix. Fix the order: immutable first (system, tools), then retrieval context, then user question.

## Use It

`code/main.py` implements a toy radix-tree KV cache plus a scheduler with two policies.

```python
"""Toy RadixAttention scheduler — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import random

KV_BUDGET_BLOCKS = 160
BLOCK_TOKENS = 16

def token_count(seg: str) -> int:
    if seg == "SYSTEM": return 2000
    if seg.startswith("DOC_"): return 500
    if seg.startswith("Q_"): return 60
    if seg == "TOOLS": return 300
    return 100

@dataclass
class Request:
    rid: int
    segments: list[str]

class RadixCache:
    def __init__(self, budget_blocks: int = KV_BUDGET_BLOCKS):
        self.budget = budget_blocks
        self.used = 0
        self.time = 0
        self.nodes: dict[tuple[str, ...], list[int]] = {}

    def walk(self, segments: list[str]) -> int:
        reused = 0
        self.time += 1
        for i in range(1, len(segments) + 1):
            key = tuple(segments[:i])
            if key in self.nodes:
                reused += token_count(segments[i - 1])
                self.nodes[key][1] = self.time
            else:
                break
        return reused

    def insert(self, segments: list[str]) -> None:
        for i in range(1, len(segments) + 1):
            key = tuple(segments[:i])
            if key in self.nodes:
                continue
            blocks = (token_count(segments[i - 1]) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            while self.used + blocks > self.budget and self._evict_one():
                pass
            self.nodes[key] = [blocks, self.time]
            self.used += blocks

    def _evict_one(self) -> bool:
        leaves = [k for k in self.nodes if not any(
            other != k and other[:len(k)] == k for other in self.nodes)]
        if not leaves:
            return False
        victim = min(leaves, key=lambda k: self.nodes[k][1])
        self.used -= self.nodes.pop(victim)[0]
        return True

def simulate(requests: list[Request], scheduler: str) -> dict:
    cache = RadixCache()
    if scheduler == "CACHE_AWARE":
        branch_count: dict[tuple[str, ...], int] = defaultdict(int)
        for r in requests:
            for i in range(1, len(r.segments) + 1):
                branch_count[tuple(r.segments[:i])] += 1
        def score(r: Request) -> int:
            return max(branch_count[tuple(r.segments[:i])] * sum(
                token_count(s) for s in r.segments[:i]) for i in range(1, len(r.segments) + 1))
        order = sorted(requests, key=score, reverse=True)
    else:
        order = list(requests)
    saved = 0
    total = 0
    for r in order:
        prompt_tokens = sum(token_count(s) for s in r.segments)
        total += prompt_tokens
        reused = cache.walk(r.segments)
        saved += reused
        cache.insert(r.segments)
    return {"hit_rate": saved / total if total else 0, "saved": saved, "total": total}

def workload_rag(n: int = 80, docs: int = 4, seed: int = 1) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        doc = f"DOC_{rng.randrange(docs)}"
        q = f"Q_{i}"
        reqs.append(Request(i, ["SYSTEM", "TOOLS", doc, q]))
    rng.shuffle(reqs)
    return reqs

def workload_scrambled(n: int = 80, docs: int = 4, seed: int = 1) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        doc = f"DOC_{rng.randrange(docs)}"
        q = f"Q_{i}"
        prefix = ["SYSTEM", "TOOLS", doc]
        rng.shuffle(prefix)
        reqs.append(Request(i, prefix + [q]))
    rng.shuffle(reqs)
    return reqs

def main() -> None:
    print("=" * 88)
    print("TOY RADIX CACHE — cache hit rate across schedulers and orderings")
    print("=" * 88)
    report("RAG workload | FCFS", simulate(workload_rag(), "FCFS"))
    report("RAG workload | CACHE_AWARE", simulate(workload_rag(), "CACHE_AWARE"))
    report("RAG scrambled prefix | FCFS", simulate(workload_scrambled(), "FCFS"))
    report("RAG scrambled prefix | CACHE_AWARE", simulate(workload_scrambled(), "CACHE_AWARE"))

def report(label: str, res: dict) -> None:
    print(f"{label:44}  hit_rate={res['hit_rate']:6.1%}   saved={res['saved']:>6}/{res['total']:<6} tok")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-radix-scheduler-advisor.md`. Given workload description, produces a prompt-ordering prescription and go/no-go for SGLang adoption.

## Exercises

1. Run `code/main.py`. Compare FCFS and cache-aware — where does the delta come from?
2. Modify prompts to randomly permute [system, tools, context]. What happens to hit rate?
3. Compute HBM cost of a 2,000-token system prompt as one radix branch on Llama 3.1 8B.
4. Read the SGLang RadixAttention paper. Explain tree-shaped LRU vs block-shaped LRU.
5. Customer reports 8% cache hit rate. Name three likely causes.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| RadixAttention | "the SGLang thing" | KV cache indexed as a radix tree so shared prefixes reuse blocks |
| Radix tree | "compact trie" | Tree where each node owns a token range and its KV blocks |
| Cache-aware scheduler | "hot-branch-first" | Scheduler that prefers requests sharing the resident branch |
| Prefix-cache hit rate | "how much of your prompt was free" | Fraction of prompt tokens served from reused KV blocks |
| FCFS | "first-come first-served" | Default scheduling that breaks prefix locality |
| Prompt template ordering | "the cache key" | The prompt's component order determines what the tree can share |

## Further Reading

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang documentation](https://sgl-project.github.io/)
- [SGLang paper (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/06-sglang-radixattention)

---

## Part 3 (ch365): TensorRT-LLM on Blackwell with FP8 and NVFP4

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

---

## Part 4 (ch366): Inference Metrics — TTFT, TPOT, ITL, Goodput, P99

> Four metrics decide whether an inference deployment is working. TTFT is prefill plus queue plus network. TPOT (equivalently ITL) is the memory-bound decode cost per token. End-to-end latency is TTFT plus TPOT times output length. Throughput is tokens per second aggregated across the fleet. But the one that matters for product is goodput — the fraction of requests that met every SLO simultaneously. Reference numbers for Llama-3.1-8B-Instruct on TRT-LLM in 2026: mean TTFT 162 ms, mean TPOT 7.33 ms, mean E2E 1,093 ms.

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- Define TTFT, TPOT, ITL, E2E, throughput, and goodput precisely and name the component each one measures.
- Explain why mean is the wrong statistic for LLM serving and how to read P50/P90/P99.
- Construct an SLO multi-constraint (e.g. TTFT<500 ms AND TPOT<15 ms AND E2E<2 s) and compute goodput against it.
- Name two benchmark tools that disagree on TPOT for the same run and explain why.

## The Problem

"Our throughput is 15,000 tokens per second." So what? If 40% of requests blew past 2 seconds end-to-end, users abandoned the session. Throughput alone does not tell you whether the product works.

## The Concept

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

### TPOT / ITL — inter-token latency

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

### Throughput

`throughput = total_output_tokens / elapsed_time`

### Goodput — the metric you actually care about

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

### Why mean is wrong

LLM latency distributions are right-skewed. Always report (P50, P90, P99).

### Reference numbers — Llama-3.1-8B-Instruct on TRT-LLM

- mean TTFT: 162 ms
- mean TPOT: 7.33 ms
- mean E2E: 1,093 ms

### The measurement trap

- **NVIDIA GenAI-Perf**: excludes TTFT from ITL calculation. ITL starts from token 2.
- **LLMPerf**: includes TTFT. ITL starts from token 1.

### Constructing an SLO

Consumer-facing 70B chat model SLO:
- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s for <300-token outputs.
- Goodput target >= 99%.

## Use It

`code/main.py` is a toy goodput calculator.

```python
"""Toy goodput calculator — stdlib Python."""

from __future__ import annotations
import random
import statistics
from dataclasses import dataclass

@dataclass
class RequestTrace:
    queue_ms: float
    prefill_ms: float
    decode_ms_per_token: list[float]
    output_tokens: int

    @property
    def ttft_ms(self) -> float:
        return self.queue_ms + self.prefill_ms

    @property
    def e2e_ms(self) -> float:
        return self.ttft_ms + sum(self.decode_ms_per_token)

    def tpot_llmperf(self) -> float:
        return self.e2e_ms / self.output_tokens

    def tpot_genaiperf(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return sum(self.decode_ms_per_token) / (self.output_tokens - 1)

def synth_workload(n: int = 1000, seed: int = 7, tail_spike_rate: float = 0.02) -> list[RequestTrace]:
    rng = random.Random(seed)
    traces = []
    for _ in range(n):
        prompt_len = rng.choice([128, 256, 512, 2048, 8192])
        output_tokens = rng.randint(50, 300)
        queue = rng.expovariate(1 / 40.0)
        prefill = prompt_len * 0.05
        decode_base = 7.0
        decodes = []
        for _ in range(output_tokens):
            t = max(1.5, rng.gauss(decode_base, decode_base * 0.15))
            if rng.random() < tail_spike_rate:
                t *= rng.uniform(3, 8)
            decodes.append(t)
        traces.append(RequestTrace(queue, prefill, decodes, output_tokens))
    return traces

def percentiles(values: list[float], ps: list[float]) -> list[float]:
    s = sorted(values)
    return [s[min(len(s) - 1, int(p * len(s)))] for p in ps]

def goodput(traces: list[RequestTrace], slo_ttft: float, slo_tpot: float, slo_e2e: float) -> float:
    good = 0
    for t in traces:
        if t.ttft_ms <= slo_ttft and t.tpot_genaiperf() <= slo_tpot and t.e2e_ms <= slo_e2e:
            good += 1
    return good / len(traces)

def main() -> None:
    print("=" * 78)
    print("TOY GOODPUT CALCULATOR — inference SLOs and the measurement trap")
    print("=" * 78)
    traces = synth_workload(n=2000)
    ttft = [t.ttft_ms for t in traces]
    tpot_nv = [t.tpot_genaiperf() for t in traces]
    tpot_llm = [t.tpot_llmperf() for t in traces]
    e2e = [t.e2e_ms for t in traces]
    p50_ttft, p90_ttft, p99_ttft = percentiles(ttft, [0.5, 0.9, 0.99])
    p50_tpot, p90_tpot, p99_tpot = percentiles(tpot_nv, [0.5, 0.9, 0.99])
    p50_e2e, p90_e2e, p99_e2e = percentiles(e2e, [0.5, 0.9, 0.99])
    print(f"  TTFT (ms)     P50={p50_ttft:7.1f}  P90={p90_ttft:7.1f}  P99={p99_ttft:7.1f}  mean={statistics.mean(ttft):7.1f}")
    print(f"  TPOT (ms)     P50={p50_tpot:7.2f}  P90={p90_tpot:7.2f}  P99={p99_tpot:7.2f}  mean={statistics.mean(tpot_nv):7.2f}")
    print(f"  E2E  (ms)     P50={p50_e2e:7.1f}  P90={p90_e2e:7.1f}  P99={p99_e2e:7.1f}")
    print(f"  Tool trap     GenAI-Perf mean TPOT={statistics.mean(tpot_nv):6.2f}  LLMPerf mean TPOT={statistics.mean(tpot_llm):6.2f}  delta={statistics.mean(tpot_llm) - statistics.mean(tpot_nv):+5.2f} ms")
    for label, t1, t2, t3 in [("loose   TTFT<800 TPOT<25 E2E<3000", 800, 25, 3000),
                               ("target  TTFT<500 TPOT<15 E2E<2000", 500, 15, 2000),
                               ("tight   TTFT<300 TPOT<10 E2E<1500", 300, 10, 1500)]:
        g = goodput(traces, t1, t2, t3)
        tag = "  SHIPPABLE" if g >= 0.99 else ("  DEGRADED" if g >= 0.95 else "  FAILING")
        print(f"  {label}  goodput={g:6.2%}{tag}")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-slo-goodput-gate.md`. Given a workload and SLO, produces a CI/CD-ready benchmark recipe that gates deploys on goodput.

## Exercises

1. Generate a distribution with 1% tail spike. How does goodput change tightening P99 TPOT from 30 ms to 15 ms?
2. Vendor quotes "15,000 tok/s on Llama 3.3 70B H100". Name three questions to ask before trusting it.
3. Why does chunked prefill protect P99 TPOT but not mean TPOT?
4. Construct a consumer SLO for a voice assistant. Which metric is most user-visible?
5. Read LLMPerf and GenAI-Perf docs. Identify three other metrics where the tools disagree.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| TTFT | "time to first token" | Queue + network + prefill; dominated by prefill at long prompts |
| TPOT | "time per output token" | Memory-bound decode cost per token after first |
| ITL | "inter-token latency" | Same as TPOT in most tools (not all — see GenAI-Perf) |
| Goodput | "SLO-met rate" | Fraction of requests meeting every SLO constraint simultaneously |
| P99 | "tail" | 1-in-100 worst-case latency; the user experience metric |
| SLO multi-constraint | "the joint" | AND of all three latency bounds |
| GenAI-Perf vs LLMPerf | "the tool trap" | Tools disagree on whether ITL includes TTFT |

## Further Reading

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html)
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/08-inference-metrics-goodput)
