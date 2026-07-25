# EAGLE-3 Speculative Decoding in Production

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
