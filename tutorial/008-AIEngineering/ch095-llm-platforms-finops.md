# Managed Platforms, Economics & FinOps

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch359): Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI

> Three hyperscalers, three distinct strategies. AWS Bedrock is a model marketplace — Claude, Llama, Titan, Stability, Cohere behind one API. Azure OpenAI is an exclusive OpenAI partnership plus Provisioned Throughput Units (PTUs) for dedicated capacity. Vertex AI is Gemini-first with the best long-context and multimodal story. In 2026 Artificial Analysis measures Azure OpenAI at ~50 ms median and Bedrock at ~75 ms on Llama 3.1 405B equivalents — PTUs explain the gap because dedicated capacity beats shared on-demand. The decision rule is not "which is fastest" but "which model catalog and FinOps surface match my product."

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## Learning Objectives

- Name the three platform strategies (marketplace vs exclusive vs Gemini-first) and match each to a product use case.
- Explain what Provisioned Throughput Units (PTUs) buy you in Azure OpenAI and why on-demand Bedrock typically reads ~25 ms slower at the 405B scale.
- Diagram the FinOps attribution surface for each platform (Bedrock Application Inference Profiles vs Vertex project-per-team vs Azure scopes + PTU reservations).
- Write down a "two-provider minimum" policy and explain why single-vendor lock-in is the expensive mistake in 2026.

## The Problem

You picked Claude 3.7 Sonnet for your product. Now you need to serve it. You can call the Anthropic API directly, or you can call it through AWS Bedrock, or you can go through a gateway. The direct API is the simplest; Bedrock adds BAAs, VPC endpoints, IAM, and CloudWatch attribution. The gateway adds failover, unified billing, and rate limits across providers.

The deeper question is catalog. If you need Claude and Llama and Gemini in the same product, you cannot buy them all from one place unless that place is Bedrock plus Vertex plus Azure OpenAI simultaneously. The hyperscalers are not interchangeable — they each made a different bet on who owns the model layer.

## The Concept

### Three strategies

**AWS Bedrock** — the marketplace. Claude (Anthropic), Llama (Meta), Titan (AWS first-party), Stability (image), Cohere (embeddings), Mistral, plus image and embedding sub-catalogs. One API, one IAM surface, one CloudWatch export. Bedrock's bet is that customers want optionality more than they want a single model.

**Azure OpenAI** — the exclusive partnership. You get GPT-4 / 4o / 5 / o-series, DALL·E, Whisper, and fine-tuning of OpenAI models in Azure datacenters. No non-OpenAI models in the "Azure OpenAI Service" catalog — those go to Azure AI Foundry (separate product). Azure's bet is that OpenAI remains the frontier and customers want enterprise controls on that specific relationship.

**Vertex AI** — Gemini first, everything else second. Gemini 1.5 / 2.0 / 2.5 Flash and Pro, plus Model Garden (third-party). Vertex's bet is multimodal long-context — 1M-token Gemini context is the differentiator.

### Latency gap at scale

Artificial Analysis runs continuous benchmarks. On equivalent Llama 3.1 405B deployments (shared on-demand), Azure OpenAI median first-token latency is around 50 ms; Bedrock is around 75 ms. The gap is not an AWS failure — it is a capacity model difference. Azure sells PTUs (Provisioned Throughput Units), which reserve GPU capacity for your tenant. Bedrock's equivalent (Provisioned Throughput) exists but starts around $21/hour per unit, and most customers stay on shared on-demand.

On-demand shared capacity competes with every other customer's traffic. Dedicated capacity does not. If your product SLA is TTFT < 100 ms at P99, you either buy PTUs on Azure, buy Bedrock Provisioned Throughput, or accept the default variance.

### Provisioned Throughput economics

Azure PTUs: a reserved block of inference compute. Up to ~70% savings vs on-demand for predictable workloads. Costs fixed per hour regardless of traffic — you pay for the reservation even when idle. The break-even is usually around 40-60% sustained utilization.

Bedrock Provisioned Throughput: $21-$50 per hour depending on model and region. Similar math — break-even is around half peak utilization. Monthly commitment required.

Vertex provisioned capacity is sold per Gemini SKU; pricing varies by model and region and is less publicly advertised.

### FinOps surface — the real differentiator

**Bedrock Application Inference Profiles** are the cleanest attribution in the marketplace. Tag a profile with `team`, `product`, `feature`; route all model invocations through it; CloudWatch breaks out cost per profile without post-processing. Added 2025, still the most granular hyperscaler native.

**Vertex** attribution is project-per-team plus labels-everywhere. You model each team as a GCP project, put labels on every resource, and use BigQuery Billing Export + DataStudio for rollups. More work, but BigQuery gives you arbitrary SQL on the cost data.

**Azure** relies on subscription/resource-group scopes plus tags, with PTU reservations as a first-class cost object. Tags are inherited from resource groups, not requests, so per-request attribution requires Application Insights custom metrics or a gateway that stamps headers.

The pattern: Bedrock is cleanest native, Vertex is most flexible via BigQuery, Azure is most opaque unless you instrument.

### Lock-in is the 2026 risk

Single-hyperscaler commitment was fine when one model dominated. In 2026 the frontier moves monthly — Claude 3.7 one quarter, Gemini 2.5 the next, GPT-5 the quarter after. Locking to one platform locks you out of two-thirds of the frontier.

The pattern working teams adopt: two-provider minimum for any product-critical LLM call. Bedrock plus Azure OpenAI is the common pair — Claude from one, GPT from the other, failover between them, same gateway. Cost uplift is negligible because gateway routes optimal; availability uplift during outages (like the Azure OpenAI January 2025 incident, the AWS us-east-1 outage) is decisive.

### Data residency, BAAs, and regulated industries

Bedrock: BAAs in most regions; VPC endpoints; guardrails. Common fintech default.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; EU data residency; the enterprise-regulated default.
Vertex: HIPAA, GDPR, data residency per region; Google Cloud's compliance stack.

All three meet the basic checkbox. The differences are in data retention policies, how logs are handled, and whether abuse-monitoring reads your traffic (default opt-in on most; opt-out available for enterprise).

### Numbers you should remember

- Azure OpenAI median TTFT on Llama 3.1 405B equivalents: ~50 ms (with PTUs).
- Bedrock median TTFT on-demand: ~75 ms.
- Bedrock Provisioned Throughput: $21-$50/hr per unit.
- Azure PTU break-even: ~40-60% sustained utilization.
- PTU savings vs on-demand at high utilization: up to 70%.

## Use It

`code/main.py` compares the three platforms on a synthetic workload — it models on-demand vs PTU economics, TTFT variance, and cost attribution fidelity.

```python
"""Managed LLM platform comparator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass
import random
import statistics

@dataclass
class Platform:
    name: str
    per_mtok_input: float
    per_mtok_output: float
    ptu_hourly: float | None
    ptu_tokens_per_hour: int
    ttft_median_ms: float
    ttft_p99_ms: float
    ttft_median_ptu_ms: float
    attribution: str

PLATFORMS = [
    Platform("Bedrock on-demand",    3.00, 15.00, 21.0, 1_200_000, 75, 180, 55, "A (Application Inference Profiles)"),
    Platform("Azure OpenAI (PTU)",    2.50, 10.00, 10.0, 2_000_000, 50, 140, 38, "B (scopes + tags + PTU obj)"),
    Platform("Vertex AI Gemini",     1.25,  5.00, None,          0, 60, 160,  0, "B+ (BQ billing export)"),
]

def simulate(tokens_in_per_day: int, tokens_out_per_day: int, sla_ttft_ms: float, use_ptu: bool) -> None:
    print(f"\nWorkload: {tokens_in_per_day/1e6:.1f}M input, {tokens_out_per_day/1e6:.1f}M output per day")
    print(f"SLA: TTFT P99 < {sla_ttft_ms:.0f} ms   |   PTU path: {'enabled' if use_ptu else 'off'}\n")
    header = f"{'Platform':25}  {'$/day':>9}  {'TTFT P50':>10}  {'TTFT P99':>10}  {'SLA':>6}  Attribution"
    print(header)
    print("-" * len(header))
    for p in PLATFORMS:
        cost_ondemand = (tokens_in_per_day / 1e6) * p.per_mtok_input + (tokens_out_per_day / 1e6) * p.per_mtok_output
        if use_ptu and p.ptu_hourly is not None:
            total_tokens = tokens_in_per_day + tokens_out_per_day
            daily_capacity_per_ptu = p.ptu_tokens_per_hour * 24
            ptu_count = max(1, (total_tokens + daily_capacity_per_ptu - 1) // daily_capacity_per_ptu)
            cost_ptu = ptu_count * p.ptu_hourly * 24
            cost = min(cost_ondemand, cost_ptu)
            ttft_p50 = p.ttft_median_ptu_ms if cost == cost_ptu else p.ttft_median_ms
            ttft_p99 = ttft_p50 * 1.5 if cost == cost_ptu else p.ttft_p99_ms
        else:
            cost = cost_ondemand
            ttft_p50 = p.ttft_median_ms
            ttft_p99 = p.ttft_p99_ms
        sla_ok = "PASS" if ttft_p99 < sla_ttft_ms else "FAIL"
        print(f"{p.name:25}  ${cost:8.2f}  {ttft_p50:7.0f} ms  {ttft_p99:7.0f} ms  {sla_ok:>6}  {p.attribution}")

def break_even_demo() -> None:
    print("\n" + "=" * 80)
    print("PTU BREAK-EVEN SWEEP — Azure OpenAI, GPT-4o class")
    print("=" * 80)
    p = PLATFORMS[1]
    print(f"On-demand rate: ${p.per_mtok_output:.2f}/M output  |  PTU: ${p.ptu_hourly:.0f}/hr, {p.ptu_tokens_per_hour/1e6:.1f}M tok/hr\n")
    print(f"{'Util %':>8}  {'On-demand $/day':>18}  {'PTU $/day':>12}  Winner")
    for util_pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        tokens_per_day = int(p.ptu_tokens_per_hour * 24 * (util_pct / 100.0))
        ondemand = (tokens_per_day / 1e6) * p.per_mtok_output
        ptu = 24 * p.ptu_hourly
        winner = "PTU" if ptu < ondemand else "on-demand"
        print(f"{util_pct:>7}%  ${ondemand:>16.2f}  ${ptu:>10.2f}  {winner}")

def lock_in_cost() -> None:
    print("\n" + "=" * 80)
    print("TWO-PROVIDER MINIMUM — cost uplift for redundancy")
    print("=" * 80)
    tokens_per_day = 5_000_000
    primary_cost = (tokens_per_day / 1e6) * 10.00
    gateway_overhead_pct = 3.0
    failover_headroom_pct = 10.0
    uplift = primary_cost * (gateway_overhead_pct + failover_headroom_pct) / 100
    print(f"Primary daily spend: ${primary_cost:.2f}")
    print(f"Gateway overhead ({gateway_overhead_pct:.0f}%): ${primary_cost * gateway_overhead_pct / 100:.2f}/day")
    print(f"Idle secondary headroom ({failover_headroom_pct:.0f}%): ${primary_cost * failover_headroom_pct / 100:.2f}/day")
    print(f"Total uplift: ${uplift:.2f}/day")
    print(f"Monthly uplift: ${uplift * 30:.2f}")
    print("Cost of one multi-hour regional outage without redundancy: customer churn, SLA credits, war-room time")

def main() -> None:
    print("=" * 80)
    print("MANAGED LLM PLATFORM COMPARATOR — 2026 approximations")
    print("=" * 80)
    simulate(tokens_in_per_day=3_000_000, tokens_out_per_day=1_000_000, sla_ttft_ms=200, use_ptu=False)
    simulate(tokens_in_per_day=30_000_000, tokens_out_per_day=15_000_000, sla_ttft_ms=100, use_ptu=True)
    break_even_demo()
    lock_in_cost()

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-managed-platform-picker.md`. Given a workload profile (models needed, TTFT SLA, daily volume, compliance requirements), it recommends a primary platform, a fallback, and a FinOps instrumentation plan.

## Exercises

1. Run `code/main.py`. At what sustained utilization does Azure PTU beat on-demand for a 70B class model? Compute the break-even and compare to the advertised 40-60% band.
2. Your product needs Claude 3.7 Sonnet and GPT-4o. Design a two-provider deployment — which goes to which hyperscaler, what gateway sits in front, what is the failover policy?
3. A regulated healthcare customer requires BAAs, US-East data residency, and sub-100ms P99 TTFT. Pick a platform and justify with three specific features.
4. You discover your Bedrock bill is up 4x this month with no traffic change. Without Application Inference Profiles, how would you find the culprit? With profiles, how long does it take?
5. Read the Azure OpenAI and Bedrock pricing pages. For a 100M-token/month Claude workload, which is cheaper — direct Anthropic API, Bedrock on-demand, or Bedrock Provisioned Throughput?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Bedrock | "AWS LLM service" | Model marketplace across Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "Azure's ChatGPT" | Exclusive OpenAI models in Azure datacenters with enterprise controls |
| Vertex AI | "Google's LLM" | Gemini-first platform with Model Garden for third-party models |
| PTU | "dedicated capacity" | Provisioned Throughput Unit — reserved inference GPUs, priced per hour |
| Application Inference Profile | "Bedrock tagging" | Per-product cost/usage profile with tags, CloudWatch-native |
| Model Garden | "Vertex catalog" | Vertex AI's third-party model section, separate from Gemini |
| Two-provider minimum | "LLM redundancy" | Policy of running every critical LLM path across ≥2 hyperscalers |
| BAA | "HIPAA paperwork" | Business Associate Agreement; required for PHI; provided by all three |
| Abuse monitoring | "the log watcher" | Provider-side safety scan on prompts/outputs; opt-out in enterprise |

## Further Reading

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/)
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/01-managed-llm-platforms)

---

## Part 2 (ch360): Inference Platform Economics — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> The 2026 inference market is no longer GPU time rental. It bifurcates into custom silicon (Groq, Cerebras, SambaNova), GPU platforms (Baseten, Together, Fireworks, Modal), and API-first marketplaces (Replicate, DeepInfra). Fireworks raised price $1/hr per GPU on May 1, 2026, and $4B valuation on 10T+ tokens/day tells you the volume-driven model works. Baseten closed $300M Series E at $5B in January 2026. The competitive positioning rule is simple: Fireworks optimizes latency, Together optimizes catalog breadth, Baseten optimizes enterprise polish, Modal optimizes Python-native DX, Replicate optimizes multimodal reach, Anyscale optimizes distributed Python.

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- Name the three market segments (custom silicon, GPU platforms, API-first) and map each vendor to a segment.
- Explain why the "per-token" API pricing model compresses toward the serving engine's cost curve, not the hardware's.
- Compute effective cost per request across at least three vendors and explain when per-minute (Baseten, Modal) beats per-token.
- Identify which platform is the right default for a given workload (serverless bursty, steady high-throughput, fine-tuned variants, multimodal).

## The Problem

You evaluated managed hyperscaler platforms. You decided you need a narrower, faster provider — Fireworks for latency, Together for breadth, Baseten for a fine-tuned custom model. Now you have six real choices and the pricing pages do not line up. Fireworks shows $/M tokens; Baseten shows $/minute; Modal shows $/second; Replicate shows $/prediction. You cannot compare them head-to-head without modeling the workload.

Worse, the business model behind each pricing page is different. Fireworks runs its own custom engine (FireAttention) on shared GPUs; the per-token rate reflects their utilization curve. Baseten gives you Truss + dedicated GPUs; per-minute reflects exclusivity. Modal is true Python serverless — per-second billing with sub-second cold starts. Same output (an LLM response), three different cost functions.

## The Concept

### The three segments

**Custom silicon** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Typically 5-10x faster decode than a GPU-based cluster on the same model. Higher per-token price (Groq was ~$0.99/M on Llama-70B late 2025) but unbeatable for latency-sensitive use cases. Groq is the production pick for voice agents and real-time translation.

**GPU platforms** — Baseten, Together, Fireworks, Modal, Anyscale. Run on NVIDIA (H100, H200, B200 in 2026) or sometimes AMD. The economic layer between "raw GPU rental" (RunPod, Lambda) and "hyperscaler managed service" (Bedrock).

**API-first marketplaces** — Replicate, DeepInfra, OpenRouter, Fal. Broad catalog, pay-per-prediction or pay-per-second, emphasize time-to-first-call.

### Fireworks — latency-optimized GPU platform

- FireAttention engine (custom); marketed as 4x lower latency than vLLM on equivalent configs.
- Batch tier at ~50% of serverless rate for non-interactive workloads.
- Fine-tuned model served at the same rate as the base model — a real differentiator versus providers that charge a premium for your LoRA.
- Mid-2026: raised on-demand GPU rental $1/hour effective May 1, 2026. Volume pricing negotiable at scale.
- Financial signal: $4B valuation, 10T+ tokens/day handled.

### Together — breadth-optimized

- 200+ models including open-source releases within days of upstream publication.
- 50-70% cheaper than Replicate on equivalent LLM models — the "AI Native Cloud" positioning is volume and catalog.
- Inference + fine-tuning + training in one API.

### Baseten — enterprise-polish-optimized

- Truss framework: model packaging with dependencies, secrets, serving config in one manifest.
- GPU range from T4 through B200. Per-minute billing with reasonable cold-start mitigation.
- SOC 2 Type II, HIPAA-ready. Common fintech and healthcare pick.
- $5B valuation, January 2026 Series E ($300M from CapitalG, IVP, NVIDIA).

### Modal — Python-native-optimized

- Infrastructure-as-code in pure Python. Decorate a function with `@modal.function(gpu="A100")` and deploy with one command.
- Per-second billing. Cold starts 2-4s with pre-warming; <1s for small models.
- $87M Series B at $1.1B valuation (2025). Strongest developer experience score in independent surveys.

### Replicate — multimodal breadth

- Pay-per-prediction. The default platform for image, video, and audio models.
- Integration ecosystem (Zapier, Vercel, CMS plugins).
- Less competitive on LLM per-token rates but wins on multimodal variety.

### Anyscale — Ray-native

- Built on Ray; RayTurbo is Anyscale's proprietary inference engine (competes with vLLM).
- Best for distributed Python workloads where the inference step is one node in a larger graph.
- Managed Ray clusters; tight integration with Ray AIR and Ray Serve.

### Per-token versus per-minute — when each wins

Per-token makes sense when the workload is latency-insensitive and bursty — you only pay for what you use. Per-minute makes sense when utilization is high and predictable — you beat per-token once you're saturating the GPU.

Rough rule: for workloads above ~30% sustained utilization of a dedicated GPU, per-minute (Baseten, Modal) starts to beat per-token (Fireworks, Together). Below that, per-token wins because you avoid paying for idle.

### Custom engine is the real moat

Every platform above vLLM and SGLang claims a custom engine. FireAttention, RayTurbo, Baseten's inference stack. Custom-engine claims shade marketing — the honest framing is that vLLM + SGLang represent about 80% of production open-source inference, and the differentiators at the platform layer are DX, attribution, and SLAs.

### Numbers you should remember

- Fireworks GPU rental: $1/hr raise effective May 1, 2026.
- Fireworks claim: 4x lower latency than vLLM on equivalent configs.
- Together: 50-70% cheaper than Replicate on LLMs.
- Baseten valuation: $5B (Series E, Jan 2026, $300M round).
- Modal valuation: $1.1B (Series B, 2025).
- Per-minute beats per-token above ~30% sustained utilization.

## Use It

`code/main.py` compares the six vendors on a synthetic workload across pricing models. Reports $/day and effective $/M tokens.

```python
"""Inference platform economics comparator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Vendor:
    name: str
    model: str
    per_mtok_output: float | None
    per_minute: float | None
    per_prediction: float | None
    tokens_per_minute: int
    cold_start_sec: float
    notes: str
    min_reserved_minutes_per_day: int = 0

VENDORS = [
    Vendor("Fireworks",    "Llama 70B",          0.90,  None,    None,  900_000, 1.5, "FireAttention, batch tier 50% off"),
    Vendor("Together",     "Llama 70B",          0.88,  None,    None,  850_000, 2.0, "200+ models, 50-70% below Replicate"),
    Vendor("Baseten",      "Custom Llama 70B",   None,  0.55,    None,  900_000, 5.0, "Truss, SOC2 HIPAA, per-min billing", 1440),
    Vendor("Modal",        "Custom Llama 70B",   None,  0.48,    None,  800_000, 2.5, "Python-native, per-sec billing, 60min warm-pool floor", 60),
    Vendor("Replicate",    "Llama 70B",          None,  None,    0.006, 750_000, 4.0, "Pay-per-prediction, multimodal"),
    Vendor("Anyscale",     "Llama 70B RayTurbo", None,  0.60,    None,  850_000, 3.0, "Ray-native, distributed Python", 1440),
]

def cost_per_day(v: Vendor, tokens_per_day: int, predictions_per_day: int) -> float:
    if v.per_mtok_output is not None:
        return (tokens_per_day / 1e6) * v.per_mtok_output
    if v.per_minute is not None:
        saturated_minutes = tokens_per_day / v.tokens_per_minute
        minutes = max(saturated_minutes, v.min_reserved_minutes_per_day)
        return minutes * v.per_minute
    if v.per_prediction is not None:
        return predictions_per_day * v.per_prediction
    return 0.0

def effective_rate(v: Vendor, tokens_per_day: int, predictions_per_day: int) -> float:
    c = cost_per_day(v, tokens_per_day, predictions_per_day)
    return (c / (tokens_per_day / 1e6)) if tokens_per_day else 0

def run_scenario(label: str, tokens_per_day: int, predictions_per_day: int) -> None:
    print(f"\n{label}")
    print(f"Workload: {tokens_per_day/1e6:.1f}M output tokens/day  |  {predictions_per_day} predictions/day")
    header = f"{'Vendor':12}  {'Model':22}  {'$/day':>8}  {'$/M tok':>10}  Notes"
    print(header)
    print("-" * len(header))
    for v in VENDORS:
        cost = cost_per_day(v, tokens_per_day, predictions_per_day)
        rate = effective_rate(v, tokens_per_day, predictions_per_day)
        print(f"{v.name:12}  {v.model:22}  ${cost:7.2f}  ${rate:9.2f}  {v.notes}")

def utilization_breakeven() -> None:
    print("\n" + "=" * 80)
    print("PER-TOKEN vs PER-MINUTE BREAK-EVEN — Fireworks (per-token) vs Baseten (per-min)")
    print("=" * 80)
    fw = VENDORS[0]
    bt = VENDORS[2]
    print(f"Fireworks: ${fw.per_mtok_output:.2f}/M output  |  Baseten: ${bt.per_minute:.2f}/min, {bt.tokens_per_minute/1e3:.0f}k tok/min\n")
    print(f"{'Util %':>8}  {'Fireworks $/day':>16}  {'Baseten $/day':>14}  Winner")
    for util_pct in (5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100):
        tokens_per_day = int(bt.tokens_per_minute * 60 * 24 * util_pct / 100)
        fw_cost = cost_per_day(fw, tokens_per_day, 0)
        bt_cost = cost_per_day(bt, tokens_per_day, 0)
        winner = "Baseten" if bt_cost < fw_cost else "Fireworks"
        print(f"{util_pct:>7}%  ${fw_cost:>15.2f}  ${bt_cost:>13.2f}  {winner}")

def cold_start_penalty() -> None:
    print("\n" + "=" * 80)
    print("COLD START PENALTY — bursty workload")
    print("=" * 80)
    print(f"{'Vendor':12}  {'Cold start':>11}  Impact at 100 cold invocations/day")
    for v in VENDORS:
        impact_sec = v.cold_start_sec * 100
        print(f"{v.name:12}  {v.cold_start_sec:>8.1f} s   +{impact_sec:.0f} seconds/day of extra latency")

def main() -> None:
    print("=" * 80)
    print("INFERENCE PLATFORM ECONOMICS — 2026 approximations")
    print("=" * 80)
    run_scenario("Scenario A — startup-scale LLM product", tokens_per_day=2_000_000, predictions_per_day=10_000)
    run_scenario("Scenario B — high-volume production", tokens_per_day=100_000_000, predictions_per_day=500_000)
    utilization_breakeven()
    cold_start_penalty()
    print("\nRule of thumb: under reserved-minute billing, per-minute beats per-token once GPU saturation stays above ~60-70% utilization; below that, per-token wins.")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-inference-platform-picker.md`. Given workload profile, SLA, and budget, picks the primary inference platform and names the runner-up.

## Exercises

1. Run `code/main.py`. At what sustained utilization does Baseten (per-minute) beat Fireworks (per-token) for a 70B model on one H100?
2. Your product serves image generation plus chat plus speech-to-text. Pick platforms for each modality and name the gateway pattern that unifies them.
3. Fireworks raises prices by $1/hr on your primary model. Model the blended cost impact if 40% of your traffic moves to batch tier (50% off).
4. A regulated customer requires SOC 2 Type II + HIPAA + dedicated GPUs. Which three platforms are viable and which one wins on FinOps?
5. Compare cost per 1,000 predictions for Llama 3.1 70B on Fireworks serverless, Together on-demand, Baseten dedicated, and Replicate API.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Custom silicon | "non-GPU chips" | Groq LPU, Cerebras WSE, SambaNova RDU — optimized for decode |
| FireAttention | "Fireworks engine" | Custom attention kernel; marketed at 4x lower latency than vLLM |
| Truss | "Baseten's format" | Model packaging manifest; dependencies + secrets + serving config |
| Per-token | "API pricing" | Charge by tokens consumed; pay for no idle |
| Per-minute | "dedicated pricing" | Charge by wall-clock GPU time; wins at high utilization |
| Per-prediction | "Replicate pricing" | Charge per model invocation; common for image/video |
| RayTurbo | "Anyscale engine" | Proprietary inference on Ray; competes with vLLM on Ray clusters |
| Batch tier | "50% off" | Non-interactive queue at reduced rate; common on Fireworks, OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | Charge LoRA-served requests at base model's rate (differentiator) |

## Further Reading

- [Fireworks Pricing](https://fireworks.ai/pricing)
- [Baseten Pricing](https://www.baseten.co/pricing/)
- [Modal Pricing](https://modal.com/pricing)
- [Together AI Pricing](https://www.together.ai/pricing)
- [Anyscale Pricing](https://www.anyscale.com/pricing)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/02-inference-platform-economics)

---

## Part 3 (ch377): AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> A gateway sits between your apps and model providers. Core features are provider routing, fallback, retries, rate limiting, secret references, observability, guardrails. Market split in 2026: **LiteLLM** is MIT OSS with 100+ providers, OpenAI-compatible, but breaks down around ~2000 RPS (8 GB memory, cascading failures in published benchmarks); best for Python, <500 RPS, dev/prototyping. **Portkey** is control-plane-positioned (guardrails, PII redaction, jailbreak detection, audit trails), went Apache 2.0 open-source March 2026, 20-40 ms latency overhead, $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $100/model/month pricing (max 5 on Plus tier); enterprise-fit if you're already on Kong. **Bifrost** (Maxim AI) — automatic retries with configurable backoff, fallback to Anthropic on OpenAI 429. **Cloudflare / Vercel AI Gateways** — managed, zero-ops, basic retry. Data residency drives the self-host decision; Portkey and Kong sit in the middle with OSS + optional managed.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Learning Objectives

- Enumerate the six core gateway features (routing, fallback, retries, rate limits, secrets, observability, guardrails).
- Map four 2026 gateways (LiteLLM, Portkey, Kong AI, Bifrost) to scale ceilings and use cases.
- Cite the Kong benchmark (228% vs Portkey, 859% vs LiteLLM) and explain why it matters for >500 RPS.
- Choose self-hosted vs managed given data residency and ops budget.

## The Problem

Your product calls OpenAI, Anthropic, and a self-hosted Llama. Each provider has a different SDK, error model, rate limit, and auth scheme. You want failover (if OpenAI 429s, try Anthropic), a single credential store, unified observability, and rate limits per tenant.

Reinventing this at the app layer couples every service to every provider. A gateway layer consolidates it into one process with one API (typically OpenAI-compatible) that fans out to providers.

## The Concept

### Six core features

1. **Provider routing** — OpenAI, Anthropic, Gemini, self-hosted, etc. behind one API.
2. **Fallback** — on 429, 5xx, or quality failure, retry elsewhere.
3. **Retries** — exponential backoff, bounded attempts.
4. **Rate limits** — per-tenant, per-key, per-model.
5. **Secret references** — pull credentials from vault at runtime (never in app).
6. **Observability** — OTel + GenAI attributes (Phase 17 · 13) + cost attribution.
7. **Guardrails** — PII redaction, jailbreak detection, allowed-topics filters.

### LiteLLM — MIT OSS, Python

- 100+ providers, OpenAI-compatible, router config, fallback, basic observability.
- Breaks down around 2000 RPS in Kong's benchmark; 8 GB memory footprint, cascading failures under sustained load.
- Best fit: Python app, <500 RPS, dev/staging gateways, experimental routing.
- Cost: $0 for OSS; cloud free tier exists.

### Portkey — control plane positioning

- Apache 2.0 OSS as of March 2026. Guardrails, PII redaction, jailbreak detection, audit trails.
- 20-40 ms per-request latency overhead.
- $49/mo for production tier with retention + SLA.
- Best fit: regulated industries needing guardrails + observability bundled.

### Kong AI Gateway — the scale play

- Built on Kong Gateway (mature API gateway product, lua+OpenResty).
- Kong's own benchmark on 12-CPU equivalent: 228% faster than Portkey, 859% faster than LiteLLM.
- Pricing: $100/model/month, max 5 on Plus tier.
- Best fit: already on Kong; >1000 RPS; willing to license.

### Bifrost (Maxim AI)

- Automatic retries with configurable backoff.
- Fallback to Anthropic on OpenAI 429 is a canonical recipe.
- Newer entrant; commercial.

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed, zero-ops. Basic retry and observability.
- Best fit: Edge-serving JavaScript apps on Cloudflare/Vercel.
- Limited compared to Kong/Portkey on guardrails and rate limits.

### Self-hosted vs managed

Data residency is the forcing function. Healthcare and finance default self-host (LiteLLM or Portkey OSS or Kong). Consumer products default managed (Cloudflare AI Gateway) or middle-tier (Portkey managed). Hybrid: self-hosted for regulated tenant, managed for others.

### Latency budget

- LiteLLM: 5-15 ms overhead typical.
- Portkey: 20-40 ms overhead.
- Kong: 3-8 ms overhead.
- Cloudflare/Vercel: 1-3 ms overhead (edge advantage).

Gateway latency directly adds to TTFT. For TTFT P99 < 100 ms SLA, Kong or Cloudflare. For P99 < 500 ms, any.

### Rate-limit semantics matter

Simple token-bucket works up to moderate scale. Multi-tenant requires sliding-window + burst allowance + per-tenant tiering. LiteLLM ships token-bucket; Kong ships sliding-window; Portkey ships tiered.

### Gateway + observability + routing compose

Phase 17 · 13 (observability) + 16 (model routing) + 19 (gateways) are the same layer in production. Pick one tool that covers all three or wire them carefully: most 2026 deployments combine Helicone (observability) or Portkey (guardrails) with Kong (scale) for split roles.

### Numbers you should remember

- LiteLLM: breaks at ~2000 RPS, 8 GB memory.
- Portkey: 20-40 ms overhead; Apache 2.0 since March 2026.
- Kong: 228% faster than Portkey, 859% faster than LiteLLM.
- Kong pricing: $100/model/month, 5 max on Plus tier.
- Cloudflare/Vercel: 1-3 ms overhead at the edge.

## Use It

`code/main.py` simulates gateway routing with fallback across 3 providers under 429/5xx injection. Reports latency, retry rate, and fallback hit rate.

## Ship It

This lesson produces `outputs/skill-gateway-picker.md`. Given scale, ops posture, compliance, latency budget, picks a gateway.

## Exercises

1. Run `code/main.py`. Configure fallback from OpenAI→Anthropic→self-hosted. What's the expected hit rate at 5% provider error rate?
2. Your SLA is TTFT P99 < 200 ms on a 300 ms baseline. Which gateways stay within budget?
3. A healthcare customer requires self-hosted + PII redaction + audit. Pick Portkey OSS or Kong.
4. Compare LiteLLM vs Kong: at what RPS ceiling should a team migrate?
5. Design a rate-limit policy for a multi-tenant SaaS: free tier, trial tier, paid tier. Token-bucket or sliding-window?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Further Reading

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/19-ai-gateways)

---

## Part 4 (ch385): FinOps for LLMs — Unit Economics and Multi-Tenant Attribution

> Traditional FinOps breaks on LLM spend. Costs are token-transactions, not resource-uptime. Tags don't map — an API call is a transaction, not an asset. Engineering decisions (prompt design, context window, output length) are financial decisions. The 2026 playbook has three attribution dimensions to instrument on day one: per-user (`user_id`) for seat pricing and expansion, per-task (`task_id` + `route`) for product surface cost and prioritization, per-tenant (`tenant_id`) for unit economics and renewal. Four token layers — prompt, tool, memory, response — one bucket hides spend. Enforcement ladder for multi-tenant products: rate limits per tenant (2-3x expected peak, clear 429 + retry-after); daily spend cap (1.5-3x contracted ceiling; triggers rate tightening + alert); kill switches on spend z-score > 4 (auto-pause + page on-call). Attribution patterns: tag-and-aggregate, telemetry-joiner (trace-ID → billing; highest accuracy), sampling-and-extrapolation, model-based allocation, event-sourced, real-time streaming. Unit metric: cost per resolved query, cost per generated artifact — not $/M tokens. Retroactive tagging always misses; instrument at request creation.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## Learning Objectives

- Explain why traditional FinOps (tags + tiers) breaks on LLM spend and name the three new attribution dimensions.
- Enumerate the four token layers (prompt, tool, memory, response) and why single-bucket billing hides cost.
- Design an enforcement ladder (rate → spend cap → kill switch) for a multi-tenant product.
- Pick a unit metric (cost per resolved query / artifact) instead of $/M tokens.

## The Problem

Your bill says $40,000. You don't know:
- Which tenant spent it.
- Which product feature drove it.
- Whether any individual user was abusive.
- Whether prompt bloat, tool calls, or memory amplification was the culprit.

Tag-and-aggregate on provider-side works for cloud resources (EC2, S3) where tags propagate to line items. LLM API calls do not auto-tag — you have to stamp user/task/tenant at the call site and carry through. Retroactive attribution always misses edge cases.

## The Concept

### Three attribution dimensions

**Per-user** (`user_id`): who is costing what. Drives seat pricing, expansion conversations, identifies power users.

**Per-task** (`task_id` + `route`): which product surface costs what. Drives feature prioritization, kill-expensive-features decisions.

**Per-tenant** (`tenant_id`): which customer is profitable. Drives unit economics, renewal pricing, tier thresholds.

Instrument all three at call site on day one. Retroactive is always worse.

### Four token layers

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

Bucketing all four together makes optimization blind. Break them out in your attribution schema.

### Enforcement ladder

1. **Rate limit** per tenant. 2-3x expected peak. Return 429 with `Retry-After`. Tenant sees friction; no surprise bill.

2. **Daily spend cap** per tenant. 1.5-3x contracted ceiling. Trigger: tighten rate limit + alert customer-success.

3. **Kill switch** on spend z-score > 4 relative to tenant baseline. Auto-pause tenant; page on-call; escalate to ops + CS.

### Attribution patterns

- **Tag-and-aggregate**: stamp metadata headers; aggregate later. Simple; rough.
- **Telemetry joiner**: join traces to billing via trace IDs. Highest accuracy. What mature teams do.
- **Sampling + extrapolation**: sample 5-10%, multiply. Cost-effective for rough spend; misses tails.
- **Model-based allocation**: regression to infer cost driver. For legacy data without tags.
- **Event-sourced**: cost as events in a stream (Kafka / Kinesis). Real-time.
- **Real-time streaming**: dashboard updates sub-second.

### Cost per X is the unit metric

$/M tokens is vendor speak. Product metrics:

- Cost per resolved support ticket.
- Cost per generated article.
- Cost per successful agent task.
- Cost per user-session-minute.

Tie cost to a product outcome. Otherwise optimization is unanchored.

### Cost attribution trace shape

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Emit on every call. Store in data lake. Aggregate per dimension. Phase 17 · 13 observability stack is where this lives.

### The compounded-savings stack

Stack: cache + batch + route + gateway. With all four:
- Cache L2 (Phase 17 · 14): ~10x cheaper input.
- Batch (Phase 17 · 15): 50% off.
- Route to cheap model (Phase 17 · 16): 60% cost reduction.
- Gateway efficiency (Phase 17 · 19): redundancy + retries.

Best-case stacked: ~5-10% of naive baseline. Most teams have 2-3 levers engaged; few stack all four.

### Numbers you should remember

- Attribution dimensions: per-user, per-task, per-tenant.
- Four token layers: prompt, tool, memory, response.
- Kill switch: spend z-score > 4.
- Unit metric: cost per resolved query, not $/M tokens.
- Stacked optimizations: ~5-10% of baseline possible.

## Use It

`code/main.py` simulates a multi-tenant LLM service with the three-tier enforcement ladder. Injects an abusive tenant and demonstrates the kill switch firing.

## Ship It

This lesson produces `outputs/skill-finops-plan.md`. Given product and scale, designs the attribution schema and enforcement ladder.

## Exercises

1. Run `code/main.py`. At what z-score does the kill switch fire? How do you pick the threshold?
2. Design a per-tenant, per-task cost dashboard. What are the 5 views you build first?
3. Your largest tenant is unit-economics-negative. Propose three interventions ordered by customer impact.
4. Compute cost per resolved ticket for a support product: 3M tokens/ticket, ~800 tickets/day, GPT-5 cached rate.
5. Argue whether retroactive tagging can ever work. When is it acceptable?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Per-user attribution | "user-level cost" | `user_id` stamped on every call |
| Per-task attribution | "feature cost" | `task_id` + `route` identify product surface |
| Per-tenant attribution | "customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "cost layers" | prompt + tool + memory + response |
| Rate limit | "429 guard" | Per-tenant ceiling enforced at gateway |
| Daily spend cap | "daily ceiling" | Tenant-scoped budget with alert |
| Kill switch | "auto-pause" | Spend z-score > 4 triggers auto-suspension |
| Cost per resolved | "product unit metric" | Cost tied to product outcome, not tokens |
| Telemetry joiner | "trace-to-billing" | Highest-accuracy attribution pattern |
| Stacked optimization | "cache+batch+route+gateway" | Compounding savings to ~5-10% baseline |

## Further Reading

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/27-finops-llms)
