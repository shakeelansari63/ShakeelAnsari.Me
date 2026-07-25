# Inference Platform Economics — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

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
