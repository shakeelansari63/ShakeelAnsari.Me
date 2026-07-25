# Evaluation and Coordination Benchmarks

> Five 2025-2026 benchmarks cover the multi-agent evaluation space. **MultiAgentBench / MARBLE** evaluates star/chain/tree/graph topologies. **COMMA** evaluates multimodal asymmetric-information coordination. **MedAgentBoard** covers medical tasks. **AgentArch** benchmarks enterprise architectures. **SWE-bench Pro** is the contamination-resistant reality check (frontier models ~23% on Pro vs 70%+ on Verified).

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Time:** ~75 minutes

## Problem

When a paper claims "our multi-agent system is better," the question is: better than what, on what, measured how? Without shared benchmarks, you cannot compare two multi-agent systems meaningfully.

## Concept

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935. Evaluates four coordination topologies (star, chain, tree, graph) on research, coding, and planning tasks. Milestone-based KPIs.

- **Graph** best for research scenarios.
- **Chain** best for stepwise-refinement coding.
- **Star** best for fast-factual consolidation.
- **Coordination tax** appears past ~4 agents on graph.
- **Cognitive planning** adds ~3% milestone achievement.

### COMMA — multimodal asymmetric information

Frontier models including GPT-4o struggle to beat a **random baseline** on agent-agent collaboration in COMMA. Multi-modality coordination collapses.

### MedAgentBoard — domain stress test

arXiv:2505.12371. Four medical categories. Multi-agent does NOT dominate single-LLM on most categories. The advantage is narrow.

### AgentArch — enterprise architectures

arXiv:2509.10769. Enterprise settings with tool use, memory, and orchestration layered. Isolates the contribution of each layer.

### SWE-bench Pro — the reality check

arXiv:2509.16941. 1865 problems across 41 repositories. Uncontaminated. Frontier models score ~23% on Pro vs 70%+ on Verified. The gap is the contamination signal.

April 2026 scores:
- Claude Opus 4.7 on Pro: **64.3%** (reported with agent-teams coordination; no primary source published yet — preliminary).
- Verdent on Verified: **76.1% pass@1**.
- Frontier raw on Pro without scaffolding: ~23-35%.

The takeaway: "we beat SWE-bench Verified" is no longer evidence of capability.

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination (https://multiagents.org/2026/). The 2026 community focal point.

### Read benchmark claims skeptically

1. Which benchmark, which split? SWE-bench Verified vs Pro matters.
2. Contamination check. Was the benchmark released after the model's training cutoff?
3. Baseline comparison. Vs single-LLM, vs random, vs prior work.
4. Statistical significance. N trials, p-value, confidence interval.
5. Task diversity. One task or many?
6. Cost disclosure. Tokens per task, wall-clock.

### What none of the benchmarks measure well

- Long-horizon coordination.
- Adversarial resilience.
- Drift under deployment.
- Cost-normalized performance.

## Build It

`code/main.py` is a non-interactive walk-through:
- Simulates 3 multi-agent systems on a toy task.
- Computes MARBLE-style milestone metrics.
- Runs a contamination check.
- Compares to a random baseline.
- Prints a benchmark-claims scorecard.

```bash
python3 code/main.py
```

## Ship It

- **Build an internal benchmark** that reflects your actual production distribution.
- **Include a random baseline** in every comparison.
- **Report cost alongside accuracy.**
- **Rebuild the benchmark quarterly.**
- **Avoid published-benchmark overfitting.**

## Exercises

1. Run `code/main.py`. Identify which system has the best cost-per-milestone.
2. Read MultiAgentBench. For your task domain, decide which topology MARBLE would recommend.
3. Read the SWE-bench Pro paper. What makes it contamination-resistant?
4. Read COMMA's finding. Design a simple multimodal coordination task for your internal benchmark.
5. Apply the benchmark-claims checklist to one recent multi-agent paper.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025; star/chain/tree/graph topologies. |
| COMMA | "Multimodal benchmark" | Multimodal asymmetric-info coordination. |
| MedAgentBoard | "Domain stress test" | Medical categories; multi-agent often does not dominate. |
| AgentArch | "Enterprise benchmark" | Tools + memory + orchestration layered. |
| SWE-bench Pro | "Contamination-resistant" | 1865 problems, 41 repos; ~23% on Pro. |
| Milestone achievement | "Partial credit" | Rewards progress, not only final success. |
| Contamination | "Benchmark leaked into training" | Post-release, scores inflate. |
| WMAC | "AAAI 2026 Bridge Program" | Workshop on Multi-Agent Coordination. |

## Further Reading

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935)
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE)
- [MedAgentBoard](https://arxiv.org/abs/2505.12371)
- [AgentArch](https://arxiv.org/abs/2509.10769)
- [SWE-bench leaderboards](https://www.swebench.com/)
- [AAAI 2026 WMAC](https://multiagents.org/2026/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks)
