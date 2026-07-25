# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> Provider lock-in is expensive. Different tool-calling workloads suit different models. Routing gateways give one API surface, retries, failover, cost tracking, and guardrails.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02, 17
**Time:** ~45 minutes

## Learning Objectives
- Distinguish self-hosted, managed, and production-grade routing options
- Implement a fallback chain that retries on provider failures in priority order
- Track per-request cost and token usage across providers
- Decide between LiteLLM, OpenRouter, and Portkey for production constraints

## The Problem

Scenarios: cost (Sonnet 3x Haiku — route accordingly), failover (OpenAI down → auto fallback to Anthropic), latency (live chat needs fast TTFt, batch doesn't), compliance (EU users stay in EU regions), experimentation (A/B test models).

## The Concept

### OpenAI-compatible proxy

Everyone speaks OpenAI shape. The gateway exposes `/v1/chat/completions` and internally proxies to any backend. The client doesn't care.

### Model aliases

```python
ROUTES = {
    "our_smart_model": [
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet"},
        {"provider": "google", "model": "gemini-1.5-pro"},
    ],
    "our_fast_model": [
        {"provider": "anthropic", "model": "claude-3-5-haiku"},
        {"provider": "openai", "model": "gpt-4o-mini"},
    ]
}
```

### Fallback chains

```python
def route(alias):
    for entry in ROUTES[alias]:
        try:
            return call_provider(entry["provider"], entry["model"], ...)
        except ProviderError:
            continue  # fall back to next in chain
    raise NoProviderAvailable
```

### Guardrails

Gateway-level: PII redaction (regex or ML before sending prompts), policy violation rejection, output filters for leak scrubbing.

### Self-hosted vs managed

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source Python | Managed SaaS | Open source + managed |
| Setup | Deploy proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your keys | Credits | Your keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Full control | Rapid prototyping | Compliance |

### Cost tracking

```python
PRICING = {
    ("openai", "gpt-4o"): {"input": 2.50/1M, "output": 10.00/1M},
    ("anthropic", "claude-3-5-sonnet"): {"input": 3.00/1M, "output": 15.00/1M},
}
cost = tokens_in * PRICING[provider, model]["input"] + tokens_out * PRICING[provider, model]["output"]
```

### Routing strategies

- Static priority: first in list, fall back on error
- Load balancing: round-robin or weighted
- Cost-aware: pick cheapest model meeting SLA
- Latency-aware: pick fastest in last N minutes
- Task-aware: prompt classifier routes by category

## Use It

`code/main.py` implements a routing gateway in ~150 lines: accepts OpenAI-shaped requests, runs priority fallback chain, tracks per-request cost, and applies PII redaction on inputs.

## Exercises

1. Trigger the outage scenario and confirm fallback with correct cost attribution.
2. Add semantic caching: SHA256 prompt key with instant cache hits.
3. Add a prompt classifier routing "code" vs "summarize" to different aliases.
4. Design per-team budgets with monthly spend caps.
5. Read LiteLLM, OpenRouter, and Portkey docs; name one unique feature each ships.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Routing gateway | One-API-surface layer in front of many providers |
| OpenAI-compatible | Accepts `/v1/chat/completions`, translates to any backend |
| Model alias | Name your code uses; gateway maps to concrete model |
| Fallback chain | Ordered provider list attempted on failure |
| Semantic caching | Embedding-based cache; near-duplicates share a hit |
| Guardrails | Input/output filters: PII redaction, policy enforcement |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/21-llm-routing-layer)
