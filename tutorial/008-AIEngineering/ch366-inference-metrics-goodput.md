# Inference Metrics — TTFT, TPOT, ITL, Goodput, P99

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
