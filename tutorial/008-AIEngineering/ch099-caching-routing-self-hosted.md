# Observability, Caching, Batch & Self-Hosted

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch371): LLM Observability Stack Selection

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

---

## Part 2 (ch372): Prompt Caching and Semantic Caching Economics

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

---

## Part 3 (ch373): Batch APIs — the 50% Discount as Industry Standard

> Every major provider ships an async batch API with a 50% discount and ~24-hour turnaround. OpenAI, Anthropic, Google, and most of the inference platforms (Fireworks batch tier, Together batch) implement the same pattern. Stack batch with prompt caching and overnight pipelines drop to ~10% of synchronous-uncached cost. The rule is brutally simple: if it is not interactive, it belongs on batch. Content generation pipelines, document classification, data extraction, report generation, bulk labeling, catalog tagging — anything tolerant of 24-hour latency is money left on the table until it moves to batch. The 2026 production pattern is to triage every new LLM workload into three lanes: interactive (synchronous with caching), semi-interactive (async queue with fallback), batch (overnight, cached input stacked). Workloads that pretend to be interactive but tolerate minutes of latency waste most.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Learning Objectives

- Name the three provider batch APIs (OpenAI, Anthropic, Google) and the common 50% discount + 24h turnaround guarantees.
- Compute the cost for stacking batch + cached-input on an overnight classification workload and compare to synchronous-uncached baseline.
- Triage a workload into interactive / semi-interactive / batch and justify the lane.
- Name the two traps: partial interactivity (user expects faster than 24h) and output-schema drift (batch file format differs per provider).

## The Problem

Your team ships a nightly report generation pipeline. 50,000 documents, summarize each, cluster the summaries, draft an executive brief. Running synchronously it takes 4 hours at $2,000/night. You hear about batch APIs.

The batch gets you 50% off. You also enable prompt caching on the system prompt (shared across all 50k calls). Stacked, the bill drops to $180/night — ~9% of baseline. Same pipeline, three config changes.

Batch is the cheapest lever in the LLM cost toolkit that nobody pulls. The reason is mostly organizational: teams think "real-time" when the SLA actually is "by morning." This lesson is about not leaving 90% of the bill on the table.

## The Concept

### The three batch APIs

**OpenAI Batch API**: JSONL file upload with a list of requests. Promised 24-hour turnaround (usually ~2-8 hours in practice). 50% discount on input and output tokens. `/v1/batches` endpoint. Cache-eligible inputs also get cached-input pricing on top.

**Anthropic Message Batches**: JSONL upload. 24-hour turnaround. 50% discount. Supports `cache_control` — cache writes are explicit, reads happen automatically within the batch.

**Google Vertex AI Batch Prediction**: BigQuery or GCS input. Similar 50% discount for Gemini. Integrates with Vertex pipelines.

### Semantic: asynchronous, not slow

Batch is "I promise to return within 24 hours" — not "this will take 24 hours." Typical P50 is 2-6 hours. Provider schedules your batch during off-peak windows when GPU inventory is underutilized.

### Stack with caching

A 50k-document summarization with the same 4K-token system prompt:

- Synchronous uncached: 50000 × ($input × 4000 + $output × 200) at full rates.
- Synchronous cached: system prompt cached after first write; remaining 49999 get 10x cheaper input.
- Batch cached: all of the above plus 50% discount on both read and write.

The stack: batch + cache = ~10% of sync uncached bill. Any workload that runs overnight and has a shared system prompt should use this.

### Workload triage

**Interactive** — user waits for the response. TTFT matters. Synchronous call with prompt caching. Cannot batch.

**Semi-interactive** — user submits a task, checks back in minutes. Async queue with fallback to sync if batch not available. Think moderate-volume RAG indexing.

**Batch** — user expects results "by morning" or "next hour." Content pipelines, classification at scale, offline analysis. Always batch, always stack caching.

Common mistake: classifying everything as interactive because the pipeline is production. Production is not a latency spec — SLA is.

### The partial-interactivity trap

Some features look interactive but tolerate 5-10 minutes. Example: a nightly customer health report with "refresh" button. User clicks refresh; wait 10 minutes is fine. Team ships it as synchronous. 50 concurrent refreshes cost 10x what batched-and-delivered-via-email would cost.

The question to ask: "What does 24-hour mean for this user?" If the answer is "they wouldn't notice," batch it.

### The output-schema trap

Batch file formats differ per provider:

- OpenAI: JSONL, one request per line.
- Anthropic: JSONL, one message per line; response format embedded.
- Vertex: BigQuery table or GCS prefix with TFRecord.

Writing "one batch client" across providers means adapter code per provider. Gateways that advertise multi-provider batch (Portkey, LiteLLM some tiers) still thin-wrap the raw format.

### Numbers you should remember

- Batch discount across providers: 50% flat on input + output.
- Turnaround SLA: 24 hours guaranteed, 2-6 hours typical P50.
- Stacked batch + cached input: ~10% of sync uncached cost.
- Workload triage rule: if 24h latency acceptable, always batch.

## Use It

`code/main.py` computes costs across sync, sync+cache, batch, and batch+cache for a 50k-document workload. Reports savings in $ and percent.

## Ship It

This lesson produces `outputs/skill-batch-triager.md`. Given workload characteristics, triages into interactive/semi/batch and estimates savings.

## Exercises

1. Run `code/main.py`. For a 100k-doc pipeline with 3K-token system prompt and 500-token output, compute the savings of full stack (batch + cache) vs sync baseline.
2. Pick three features in a real product you know. Triage each into interactive/semi/batch.
3. A user complains their report took 3 hours. Was that a batch mis-triage or a legitimate interactive? Write the decision criterion.
4. Your batch API return SLA is 24h but P99 is 20 hours. How do you communicate this to the user — what is the downstream system behavior on the edge case?
5. Compute break-even: at what shared-prefix length does batch + cache become cheaper than running overnight on your own reserved GPU?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Further Reading

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format and `cache_control` interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/15-batch-apis)

---

## Part 4 (ch374): Model Routing as a Cost-Reduction Primitive

> A dynamic broker evaluates every request (task type, token length, embedding similarity, confidence) and sends simple queries to a cheap model, escalating complex ones to a frontier model. Also called model cascading. Production case studies show 20-60% cost reduction at iso-quality across US/UK/EU deployments; a 30% routing efficiency improvement on high-volume SaaS turns into six-figure annual savings. The 2026 context is that LLM inference prices dropped ~10x per year — a GPT-4-class token went from $20/M to ~$0.40/M from late 2022 to 2026. Most of the drop is better serving stacks (Phase 17 · 04-09), not hardware. Routing is how you convert that price drop into margin without product regression. The failure mode is cheap-model drift: the route pushes 40% to a weaker model, quality drops 3-5% on reasoning tasks, no one notices for a quarter. Gate routes by online quality metrics, not just offline eval sets.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Learning Objectives

- Explain model cascading: cheap-first with confidence check, escalate on low confidence.
- Enumerate the four routing signals (task classification, prompt length, embedding similarity to known-hard set, self-confidence from first-pass).
- Compute expected blended cost at target routing split and quality loss tolerance.
- Name the drift-monitoring metric (online quality gate) that catches cheap-model creep.

## The Problem

Your service costs $80k/month on GPT-5. Your analytics show 70% of queries are simple: "what time is it in Paris?" "rephrase this sentence." A Haiku-class model handles those perfectly at 3% of the cost. 30% need GPT-5's reasoning — coding, math, multi-step planning.

If you route the 70% to cheap and 30% to expensive, your bill drops ~65% at the same product quality. This is routing. The trick is building the broker without regressing quality.

## The Concept

### Four routing signals

1. **Task classification**: simple/complex/codegen/math/chat. Can be a rules-based classifier, a small LLM (Haiku-class at $0.25/M), or embedding similarity to labeled buckets. Output: route = cheap / balanced / frontier.

2. **Prompt length**: prompts >4K tokens often need frontier for coherence. Prompts <500 tokens usually don't.

3. **Embedding similarity to known-hard set**: if the query is close (cosine > 0.88) to a known-hard bucket, escalate to frontier directly.

4. **Self-confidence from first-pass**: send to cheap; if model's log-probs show low confidence OR it refuses OR outputs hedging language, retry on frontier. Adds P95 latency on ~10% of traffic but saves 50%+ on the other 90%.

### Three patterns

**Pre-route** (classifier up front): ~5-10ms latency added; fastest overall.

**Cascade** (cheap-first, escalate on low confidence): ~1.2x median latency (cheap run plus verify), ~2x on escalated. Best quality floor.

**Ensemble route** (run cheap and frontier in parallel for a sample, reward-model pick): highest quality, highest cost; use only for critical A/B.

### Implementation

AI gateways (Phase 17 · 19) expose routing. LiteLLM has `router` config with fallback and cost-routing. Portkey has guards + routing. Kong AI Gateway has plugin-based routing. OpenRouter's model marketplace exposes a recommendation API.

Open-source: RouteLLM (LMSYS), Not Diamond (commercial), Prompt Mule.

### The 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

Most of the improvement is serving efficiency — the core lessons in Phase 17 · 04-09 turned into provider-side cost drops. Routing lets you capture those gains at the app layer instead of waiting for all your users to migrate to the cheap tier.

### Drift is the real risk

Your route sends 40% to the cheap model. Over six months, the task distribution shifts (users get more sophisticated, ask longer questions). The router doesn't notice because its classifier was trained on Q1 data. Quality drops silently. Nobody complains loud enough. You find out in a competitor benchmark you lost.

Gate routes by online quality metrics:

- User thumbs-up / thumbs-down per route.
- Automated LLM-judge on a held-out sample (5%) per route.
- Escalation rate: if cascade is kicking up-route >30%, the cheap model is being over-routed.
- Refusal rate per route.

### Numbers you should remember

- 2026 routing savings at iso-quality: 20-60% case studies.
- LLM price drop 2022-2026: ~10x per year aggregate.
- GPT-4-level 2022 vs 2026: ~$20/M → ~$0.40/M.
- Cascade latency impact: ~1.2x median, ~2x escalated (~10% of traffic).

## Use It

`code/main.py` simulates pre-route, cascade, and ensemble on a mixed workload. Reports blended cost, quality loss, and escalation rate.

## Ship It

This lesson produces `outputs/skill-router-plan.md`. Given workload and quality budget, picks a routing pattern and signals.

## Exercises

1. Run `code/main.py`. At what accuracy floor does cascade beat pre-route?
2. Your user base is 30% enterprise (complex queries), 70% free tier (simple). Design the routing split. What online metric gates it?
3. A route drops quality by 2% but saves 40%. Is that a ship? Depends on product — argue both.
4. Implement a confidence check using logprobs from OpenAI / Anthropic APIs. What's the threshold you start with?
5. Over six months, escalation rate climbs from 8% to 22%. Diagnose three causes and the fix for each.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Further Reading

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/16-model-routing)

---

## Part 5 (ch386): Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang

> Four engines dominate self-hosted inference in 2026. Pick based on hardware, scale, and ecosystem. **llama.cpp** is fastest on CPU — widest model support, full control over quantization and threading. **Ollama** is the dev-laptop one-command install, ~15-30% slower than llama.cpp (Go + CGo + HTTP serialization), 3x throughput gap under prod-like load. **TGI entered maintenance mode December 11, 2025** — only bug fixes, ~10% slower raw throughput than vLLM but historically top observability and HF-ecosystem integration. That maintenance status makes it a risky long-term bet — SGLang or vLLM are safer defaults for new projects. **vLLM** is the general-purpose production default — v0.15.1 (February 2026) adds PyTorch 2.10, RTX Blackwell SM120, H200 optimization. **SGLang** is the agentic multi-turn / prefix-heavy specialist — 400,000+ GPUs in production (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Hardware constraints: CPU-only → llama.cpp only. AMD / non-NVIDIA → vLLM only (TRT-LLM is NVIDIA-locked). 2026 pipeline pattern: dev = Ollama, staging = llama.cpp, prod = vLLM or SGLang. Same GGUF/HF weights throughout.

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## Learning Objectives

- Pick an engine given hardware (CPU / AMD / NVIDIA Hopper / Blackwell), scale (1 user / 100 / 10,000), and workload (general chat / agent / long-context).
- Name the 2026 TGI maintenance-mode status (December 11, 2025) and why it biases new projects toward vLLM or SGLang.
- Describe the dev/staging/prod pipeline using the same GGUF or HF weights throughout.
- Explain why "CPU only" forces llama.cpp and "AMD" excludes TRT-LLM.

## The Problem

Your team starts a new self-hosted LLM project. One engineer says Ollama, another says vLLM, a third says "doesn't TGI just work out of the box?" All three are right for different contexts. None is right for all.

In 2026 the choice tree matters: hardware first, scale second, workload third. And one specific 2025 event — TGI entering maintenance mode December 11 — changes the default for new projects.

## The Concept

### The five engines

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / minimal deps / widest model support | Fastest on CPU, full control |
| **Ollama** | Dev laptops, single user, one-command install | 15-30% slower than llama.cpp; 3x prod throughput gap |
| **TGI** | HF ecosystem, regulated industries | **Maintenance mode Dec 11, 2025** |
| **vLLM** | General-purpose production, 100+ users | Broad production default; v0.15.1 Feb 2026 |
| **SGLang** | Agentic multi-turn, prefix-heavy workloads | 400,000+ GPUs in production |

### Hardware-first decision

**CPU only** → llama.cpp. Ollama works too but is slower. No other engine is competitive on CPU.

**AMD GPU** → vLLM (AMD ROCm support). SGLang also works. TRT-LLM is NVIDIA-locked, so it's out.

**NVIDIA Hopper (H100 / H200)** → vLLM or SGLang or TRT-LLM. All three top-tier.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM is the throughput leader (Phase 17 · 07). vLLM and SGLang follow close.

**Apple Silicon (M-series)** → llama.cpp (Metal). Ollama wraps this.

### Scale-second decision

**1 user / local dev** → Ollama. One command, first-token in seconds.

**10-100 users / small team** → vLLM single-GPU.

**100-10k users / production** → vLLM production-stack (Phase 17 · 18) or SGLang.

**10k+ users / enterprise** → vLLM production-stack + disaggregated (Phase 17 · 17) + LMCache (Phase 17 · 18).

### Workload-third decision

**General chat / Q&A** → vLLM wins on broad default.

**Agentic multi-turn (tools, planning, memory)** → SGLang's RadixAttention (Phase 17 · 06) dominates.

**RAG with heavy prefix reuse** → SGLang.

**Code generation** → vLLM fine; SGLang slightly better on cache.

**Long context (128K+)** → vLLM + chunked prefill; SGLang + tiered KV.

### The TGI maintenance trap

Hugging Face TGI entered maintenance mode December 11, 2025 — only bug fixes going forward. Historically: top-tier observability, best-in-class HF-ecosystem integration (model cards, safety tools), slightly behind vLLM on raw throughput.

For new projects in 2026: default away from TGI. Existing TGI deployments can continue but should migrate eventually. SGLang and vLLM are the safer defaults.

### The pipeline pattern

Dev (Ollama) → staging (llama.cpp) → prod (vLLM). Same GGUF or HF weights throughout. Engineers iterate quickly on laptops; staging mirrors production quantization; prod is the serving target.

### Ollama caveat

Ollama is great for dev. It is not great for shared production: Go HTTP serialization adds overhead, concurrency management is simpler than vLLM, OpenTelemetry support lags. Use Ollama where it shines — one user, one command — and switch to vLLM for shared.

### Self-hosted vs managed is a separate decision

Phase 17 · 01 (managed hyperscalers), · 02 (inference platforms) cover managed. This lesson assumes you've already decided to self-host. Reasons to self-host: data residency, custom fine-tune, total cost ownership at scale, domain model not available on hosted.

### Numbers you should remember

- TGI maintenance mode: December 11, 2025.
- vLLM v0.15.1: February 2026; PyTorch 2.10; Blackwell SM120 support.
- SGLang production footprint: 400,000+ GPUs.
- Ollama throughput gap vs llama.cpp: 15-30% slower; 3x under prod load.

## Use It

`code/main.py` is a decision-tree walker: given hardware + scale + workload, picks an engine and explains why.

## Ship It

This lesson produces `outputs/skill-engine-picker.md`. Given constraints, picks an engine and writes the migration plan.

## Exercises

1. Run `code/main.py` with your hardware / scale / workload. Does the output match your intuition?
2. Your infra is 12 H100s and 8 MI300X AMD. What engine? Why is TRT-LLM off the table?
3. A team wants to use TGI in 2026 because "it's what we know." Argue the migration case.
4. Ollama dev to vLLM prod: what changes in quantization, configuration, and observability?
5. RAG product with P99 prefix length 8K and high reuse across tenants. Pick an engine and stack it with Phase 17 · 11 + 18.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| llama.cpp | "the CPU one" | Widest model support, fastest on CPU |
| Ollama | "the laptop one" | One-command install, dev-grade throughput |
| TGI | "HF's serving" | Maintenance mode since Dec 2025 |
| vLLM | "the default" | Broad production baseline 2026 |
| SGLang | "the agentic one" | Prefix-heavy, RadixAttention |
| TRT-LLM | "NVIDIA-locked" | Blackwell throughput leader, NVIDIA only |
| GGUF | "llama.cpp format" | Bundled K-quant variants |
| Production-stack | "vLLM K8s" | Phase 17 · 18 reference deployment |
| Pipeline pattern | "dev→stage→prod" | Ollama → llama.cpp → vLLM on same weights |

## Further Reading

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/28-self-hosted-serving-selection)
