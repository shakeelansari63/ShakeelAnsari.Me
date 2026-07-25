# Prompt Caching and Semantic Caching Economics

> Caching happens at two layers. L2 (provider-level) prompt/prefix caching reuses attention KV for repeated prefixes — Anthropic's prompt-caching docs advertise up to 90% cost reduction and 85% latency reduction; for Claude 3.5 Sonnet cache reads are $0.30/M vs $3.00/M fresh with a 5-minute TTL and a 2x write premium for the 1-hour TTL option. L1 (app-level) semantic caching skips the LLM entirely on embedding similarity hits.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish L2 prompt/prefix caching from L1 semantic caching.
- Explain Anthropic's `cache_control` explicit marking and the two TTL options with their price multipliers.
- Compute expected monthly savings given hit rate, prompt/response mix, and token prices.
- Name the parallelization anti-pattern that inflates bills by 5-10x and the dynamic-content anti-pattern that collapses hit rate.

## The Problem

You add prompt caching to your RAG service. The bill stays flat. Your prompts look static but they are not — the system prompt includes the current date, a request ID, and randomized example reorder. Every request writes a new cache entry, reads zero.

## The Concept

### L2 — provider prompt/prefix caching

**Anthropic**: explicit `cache_control` marker. TTL: 5-minute (1.25x write premium) or 1-hour (2x write premium). Cache reads: $0.30/M vs $3.00/M fresh.

**OpenAI**: automatic caching for prompts ≥1024 tokens. Cached input is roughly 10x cheaper.

**Google (Gemini)**: context caching via explicit API.

### L1 — app-level semantic caching

Before calling the LLM, hash the prompt, embed it, and look for similar cached request (cosine similarity > 0.95). Production hit rates: open-ended chat 10-15%, structured FAQ 40-70%.

### The parallelization anti-pattern

10 parallel tool calls with same system prompt. First cache-write completes ~300 ms later. Requests 2-10 arrive in the same millisecond window — each sees cache miss. Fix: serialize first request, then fire the rest.

### The dynamic content anti-pattern

`"Current time is 14:32:17. User ID: abc123."` — every request is unique. Fix: move static content to cacheable prefix; append dynamic after the cache boundary.

## Use It

`code/main.py` simulates L1 + L2 caching on mixed workloads.

```python
"""Two-layer caching simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass
import random

BASE_INPUT = 3.00
BASE_OUTPUT = 15.00
CACHED_INPUT = 0.30
CACHE_WRITE_5MIN = 1.25 * BASE_INPUT
CACHE_WRITE_1HR = 2.00 * BASE_INPUT

@dataclass
class Request:
    prompt_tokens: int
    prefix_hash: str
    is_parallel_wave: bool
    arrived_at: float

@dataclass
class Config:
    l1_enabled: bool
    l2_enabled: bool
    parallel_penalty: bool
    l1_hit_prob: float
    ttl: str

def make_workload(n: int = 500, seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    prefixes = [f"prefix_{i}" for i in range(12)]
    now = 0.0
    reqs = []
    for i in range(n):
        if rng.random() < 0.4:
            for _ in range(5):
                reqs.append(Request(rng.choice([2000, 4000, 8000]), rng.choice(prefixes), True, now))
        else:
            reqs.append(Request(rng.choice([2000, 4000, 8000]), rng.choice(prefixes), False, now))
        now += rng.uniform(0.1, 2.0)
    return reqs

def simulate(reqs: list[Request], cfg: Config) -> dict:
    l2_cache: set[str] = set()
    l2_writes = 0; l2_reads = 0; l1_hits = 0; cost = 0.0
    rng = random.Random(11)
    for r in reqs:
        if cfg.l1_enabled and rng.random() < cfg.l1_hit_prob:
            l1_hits += 1
            continue
        if cfg.l2_enabled:
            if r.prefix_hash in l2_cache:
                l2_reads += 1
                cost += (r.prompt_tokens / 1e6) * CACHED_INPUT
            else:
                write_cost = CACHE_WRITE_5MIN if cfg.ttl == "5min" else CACHE_WRITE_1HR
                if cfg.parallel_penalty and r.is_parallel_wave:
                    cost += (r.prompt_tokens / 1e6) * write_cost
                    l2_writes += 1
                else:
                    cost += (r.prompt_tokens / 1e6) * write_cost
                    l2_cache.add(r.prefix_hash)
                    l2_writes += 1
        else:
            cost += (r.prompt_tokens / 1e6) * BASE_INPUT
        cost += (200 / 1e6) * BASE_OUTPUT
    return {"cost": cost, "l1_hits": l1_hits, "l2_reads": l2_reads, "l2_writes": l2_writes}

def main() -> None:
    print("=" * 95)
    print("PROMPT + SEMANTIC CACHING — 500 requests")
    print("=" * 95)
    base = make_workload()
    for label, cfg in [
        ("NO CACHING", Config(False, False, True, 0.0, "5min")),
        ("L2 5-min, parallel penalty", Config(False, True, True, 0.0, "5min")),
        ("L2 5-min, parallel fixed", Config(False, True, False, 0.0, "5min")),
        ("L2 1hr + L1 semantic 30%", Config(True, True, False, 0.30, "1hr")),
        ("L2 1hr + L1 semantic 70%", Config(True, True, False, 0.70, "1hr")),
    ]:
        r = simulate([Request(x.prompt_tokens, x.prefix_hash, x.is_parallel_wave, x.arrived_at) for x in base], cfg)
        print(f"{label:45}  cost=${r['cost']:7.2f}  L1={r['l1_hits']:4}  L2_reads={r['l2_reads']:4}  L2_writes={r['l2_writes']:4}")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-cache-auditor.md`. Given prompt template and traffic, audits cacheability and recommends restructure.

## Exercises

1. Toggle the parallelization flag. How much does the bill change?
2. System prompt has a date. Move it out. Show before/after hit rate math.
3. Calculate break-even for 1-hour TTL vs 5-minute TTL given your arrival rate.
4. Semantic cache at 0.95 threshold hits 20%. At 0.85 it hits 50% but incorrect responses. Pick the right threshold.
5. Rewrite 10 parallel sub-queries per user question for cache-friendliness.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |

## Further Reading

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/14-prompt-semantic-caching)
