# Voting, Self-Consistency, and Debate Topology

> The cheapest aggregation: sample N independent agents, majority-vote. Multi-agent extends it with **heterogeneous** agents to escape monoculture. Beyond majority vote, debate topology matters: MultiAgentBench (ACL 2025) evaluated star / chain / tree / graph coordination and found **graph best for research**, with a "coordination tax" past ~4 agents.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## Problem

Debate can improve accuracy (Du et al.). It can also degrade it. Whether debate helps depends on four structural choices: who talks to whom (topology), how many rounds, whether agents are heterogeneous, and whether an adversarial voice is present.

## Concept

### Self-consistency, the single-model baseline

Wang et al. 2022 sampled the same model N times at temperature > 0 and majority-voted on reasoning-path answers. Substantial gains on GSM8K with N=40. Limit: one base model, correlated errors.

### Multi-agent vote, the heterogeneous extension

Replace N samples with N *different* agents. Different base models, prompts, tool access. The benefit: uncorrelated errors.

### The four topologies

```
star                chain               tree                graph
    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

**Star:** one hub, all others talk only to hub.
**Chain:** linear, each agent sees the prior one's output.
**Tree:** hierarchical.
**Graph:** any-to-any. Includes fully-connected clique.

### The coordination tax (MultiAgentBench)

- **Graph** topology wins on research tasks.
- **Star** wins on fast-answer factual tasks.
- **Chain** wins on stepwise pipelines.
- **Coordination tax** appears past ~4 agents in graph topology.

### Multi-Agent Debate Strategies

MAD variants that are structurally similar to self-consistency often underperform self-consistency at equal budget. MAD helps most when agents are genuinely heterogeneous and debate has adversarial structure.

### AgentVerse emergent patterns

Two behaviors emerge: **Volunteer** (an agent offers help unprompted) and **Conformity** (an agent adjusts to match a critic, even when wrong).

### Heterogeneity: the actual knob that moves accuracy

Swapping one of your N agents for a different base model gives a bigger accuracy bump than increasing N by 1. Three different models beat five copies of one model on most tasks with clean ground truth.

## Build It

`code/main.py` implements:
- `run_star(agents, hub, question)` — hub polls each worker, aggregates.
- `run_chain(agents, question)` — sequential refinement.
- `run_tree(root, children, question)` — hierarchical with depth-2 aggregation.
- `run_graph(agents, question, rounds)` — all-to-all debate, bounded rounds.
- A measurement harness that runs each topology at N=3, 5, 7 and reports (accuracy, total_tokens, wallclock_simulated).

```
python3 code/main.py
```

Expected output: a table of topology × N → (accuracy, tokens, latency). Graph wins at N=3-5 on research-style tasks; star wins on fast-factual tasks.

## Ship It

- Start with **self-consistency at N=5** using one strong base model.
- Upgrade to **heterogeneous voting at N=3** if accuracy matters.
- Only upgrade to **debate topology** if the task has structure and bounded rounds are feasible.
- Always log the minority cluster.
- Benchmark wall-clock and tokens alongside accuracy.

## Exercises

1. Run `code/main.py`. Plot the coordination-tax curve for graph topology.
2. Implement A-HMAD: three agents with deliberately different biases.
3. Add a "judge" role to the graph topology that does not vote, only scores the final consensus.
4. Read the AgentVerse paper (ICLR 2024). Identify which emergent behavior your implementation exhibits most strongly.
5. Read MultiAgentBench (arXiv:2503.01935) Section 4. Reproduce the "graph-wins-research" result.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-consistency | "Sample N times, vote" | Wang 2022. Single model, N temperature>0 samples. |
| Heterogeneity | "Different models" | Ensemble of different base models. Breaks monoculture. |
| MAD | "Multi-agent debate" | Agents exchanging critiques over rounds. |
| A-HMAD | "Adversarial Heterogeneous MAD" | MAD with different models + adversarial structure. |
| Topology | "Who talks to whom" | Star, chain, tree, graph. |
| Coordination tax | "Diminishing returns" | Above ~4 agents on graph, cost grows faster than quality. |
| Volunteer behavior | "Unprompted help" | AgentVerse emergent pattern. |
| Conformity behavior | "Agreement under pressure" | AgentVerse emergent pattern. |
| Jury | "Small specialized panel" | Ensemble with roles (examiner, context, scorer). |

## Further Reading

- [Wang et al. — Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Du et al. — Multiagent Debate](https://arxiv.org/abs/2305.14325)
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935)
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371)
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf)
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/15-voting-debate-topology)
