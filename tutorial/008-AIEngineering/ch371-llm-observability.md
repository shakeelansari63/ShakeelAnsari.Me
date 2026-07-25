# LLM Observability Stack Selection

> The 2026 observability market splits into two categories. Development platforms (LangSmith, Langfuse, Comet Opik) bundle monitoring with evals, prompt management, session replays. Gateway/instrumentation tools (Helicone, SigNoz, OpenLLMetry, Phoenix) focus on telemetry. Common production pattern: Gateway (Helicone/Portkey) + eval platform (Phoenix/TruLens) glued by OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish development platforms from gateway/telemetry tools.
- Map six major tools to their licensing, pricing, and sweet-spot use cases.
- Explain the OpenTelemetry-glue pattern that lets you combine a gateway tool with a separate eval platform.
- Name the 2026 cost differentiator (Arize AX's zero-copy approach vs monolithic ingest).

## The Problem

You shipped an LLM feature. It works. You have no visibility into prompt failures, tool loops, latency regressions, cost spikes, or prompt-cache hit rate. Picking a tool involves four axes: stack, license tolerance, budget, and self-host need.

## The Concept

### Two categories

**Development platforms** bundle observability with evals, prompt management, dataset versioning, session replay. LangSmith, Langfuse, Comet Opik.

**Gateway/telemetry tools** instrument inference calls — prompt, response, tokens, latency, model, cost. Helicone, SigNoz, OpenLLMetry, Phoenix.

### Langfuse — OSS balance

Core Apache/MIT licensed; self-host via Docker. Cloud free: 50K events/month. Paid: $29/mo for team.

### Phoenix (Arize) — telemetry-first

Elastic License 2.0. Excellent RAG and drift visualization. Primarily development-time observability.

### Arize AX — the scale play

Commercial. Zero-copy Iceberg/Parquet integration. Claims ~100x cheaper than monolithic at scale.

### LangSmith — LangChain/LangGraph first

Commercial, $39/user/month. Self-host only on Enterprise.

### Helicone — proxy-based

15-30 minute setup by swapping `OPENAI_API_BASE`. MIT licensed; 100K req/mo free.

### The glue: OpenTelemetry + GenAI semantic conventions

1. Emit OTel with GenAI conventions from every LLM call.
2. Route to gateway (Helicone / Portkey) for day-to-day.
3. Dual-ship to eval platform (Phoenix / Langfuse) for regressions.
4. Archive in data lake (Iceberg) for long-term analysis.

### Sampling

At >1M requests/day, full-trace retention costs more than the LLM calls. Sample: 100% errors, 100% high-cost, 5% success.

## Use It

`code/main.py` simulates a 1M-trace day across retention strategies.

```python
"""Observability sampling and cost simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass
import random

BYTES_PER_TRACE = 4_500
COST_PER_GB_MONTH = 0.023
OBSERVABILITY_INGEST_PER_GB = 0.50
ARIZE_AX_PER_GB = 0.005

@dataclass
class Strategy:
    name: str
    sample_rate: float
    keep_errors: bool
    keep_highcost: bool

STRATEGIES = [
    Strategy("100% retain",                1.00, True, True),
    Strategy("10% random sample",          0.10, False, False),
    Strategy("5% success + 100% errors",   0.05, True, False),
    Strategy("5% success + errors + $$$",  0.05, True, True),
    Strategy("1% aggregates only",         0.01, True, True),
]

def simulate_day(strategy: Strategy, traces_per_day: int = 1_000_000) -> dict:
    rng = random.Random(7)
    retained = 0
    for i in range(traces_per_day):
        is_error = rng.random() < 0.02
        is_highcost = rng.random() < 0.01
        keep = rng.random() < strategy.sample_rate
        if (strategy.keep_errors and is_error) or (strategy.keep_highcost and is_highcost):
            keep = True
        if keep: retained += 1
    gb = retained * BYTES_PER_TRACE / 1e9
    return {"name": strategy.name, "retained": retained, "gb_per_day": gb,
            "monolithic_month": gb * 30 * OBSERVABILITY_INGEST_PER_GB,
            "arize_month": gb * 30 * ARIZE_AX_PER_GB}

def main() -> None:
    print("=" * 120)
    print("OBSERVABILITY SAMPLING — 1M traces/day")
    print("=" * 120)
    for s in STRATEGIES:
        r = simulate_day(s)
        print(f"{r['name']:30}  retained={r['retained']:7}  {r['gb_per_day']:6.2f} GB/day  mono=${r['monolithic_month']:8.2f}  arize=${r['arize_month']:6.2f}")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-observability-stack.md`. Given stack, scale, budget, license posture, picks the tool(s).

## Exercises

1. Your team on LangChain wants OSS self-hosted observability. Pick Langfuse or Opik.
2. At 5M traces/day with Datadog at $150K/month, compute break-even for Arize AX.
3. Design an OpenTelemetry GenAI attribute set your org should mandate.
4. Argue whether Phoenix alone is sufficient for production.
5. Helicone is 20ms proxy overhead. At P99 TTFT 300 ms, is that acceptable?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |

## Further Reading

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/13-llm-observability)
