# MARL — MADDPG, QMIX, MAPPO

> The reinforcement-learning heritage of multi-agent coordination, which still informs LLM-agent systems in 2026. **MADDPG** introduced Centralized Training, Decentralized Execution (CTDE). **QMIX** is value-decomposition with a monotonic mixing network. **MAPPO** is PPO with a centralized value function — the default 2026 cooperative-MARL baseline.

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## Problem

LLM-agent systems increasingly train policies for inter-agent coordination: when to defer, when to act, which peer to call. The literature that tells you how to train such policies is Multi-Agent Reinforcement Learning (MARL).

Reading MARL papers without the pattern vocabulary is painful. Centralized training with decentralized execution (CTDE), value decomposition, and centralized critics are specific answers to specific problems.

## Concept

### Three environments the papers use

- **Particle World.** Simple 2D physics. MADDPG's testbed.
- **StarCraft Multi-Agent Challenge (SMAC).** Cooperative micro-management. QMIX's testbed.
- **Google Research Football, Hanabi, MPE.** MAPPO baselines.

### MADDPG (2017) — the CTDE pattern

Each agent `i` has an actor `mu_i(o_i)` that maps its own observation to action. Each agent also has a critic `Q_i(x, a_1, ..., a_n)` that sees all observations and all actions during training.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Why CTDE: at training time, we know everyone's actions. At deploy time, each agent only sees `o_i` and calls `mu_i(o_i)`. Failure mode: critics grow with N agents.

### QMIX (2018) — value decomposition

Cooperative only. Global reward is the sum of a monotone function of per-agent Q-values:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

The monotonicity guarantees `argmax_a Q_tot` can be computed by each agent choosing independently. Failure mode: monotonicity constraint is restrictive.

### MAPPO (2022) — the overlooked default

Multi-Agent PPO: PPO with a centralized value function. Each agent has its own policy; all agents share value functions that see the full state. MAPPO matches or beats off-policy MARL methods with minimal tuning. In 2026, MAPPO is the default baseline for cooperative MARL.

### Why LLM-agent engineers should care

1. **Router training.** A meta-agent chooses which sub-agent handles a task. MAPPO fits.
2. **Role emergence.** QMIX-style value decomposition forces complementarity by construction.
3. **Multi-agent tool use.** CTDE produces deployable local policies that respect resource constraints.

### CTDE as a design pattern beyond RL

Even without training, CTDE is useful: during design, assume full team visibility. At runtime, enforce decentralized execution. This forces you to keep per-agent state explicit.

### The non-stationarity problem

When multiple agents learn simultaneously, each agent's environment is non-stationary. MARL algorithms address this: MADDPG with global critic, QMIX with value decomposition, MAPPO with centralized value function.

## Build It

`code/main.py` implements three pattern demonstrations on a tiny 2-agent cooperative grid-world (4x4 grid, one reward pellet):
- `IndependentAgents` — each agent treats others as environment. Baseline.
- `MADDPGStyle` — centralized critic computes a joint value.
- `QMIXStyle` — value decomposition with a monotone mixer.
- `MAPPOStyle` — centralized value function.

```
python3 code/main.py
```

Expected output: independent agents take ~6 steps on average; CTDE variants converge toward ~3.5 steps.

## Ship It

- **Start with MAPPO.** Reproducing it first saves weeks of chasing fancier methods.
- **Log every agent's observation and action stream.**
- **Separate training code from execution code.** CTDE is a discipline.
- **Reward shaping warning.** MARL is exquisitely sensitive to reward design.
- **For LLM agents**, consider prompt-level policies first.

## Exercises

1. Run `code/main.py`. Measure the steps-to-goal gap between independent and MAPPO-style agents.
2. Implement a competitive variant: two agents, one pellet, only the first to reach gets reward.
3. Read MADDPG Section 3. Implement the exact critic update rule symbolically.
4. Read MAPPO. Why do the authors argue centralized value + PPO beats off-policy MARL?
5. Apply CTDE as a design pattern to a hypothetical LLM-agent system.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARL | "Multi-Agent RL" | Reinforcement learning for multi-agent systems. |
| CTDE | "Centralized Training, Decentralized Execution" | Train with global info; deploy with local policies. |
| MADDPG | "Multi-Agent DDPG" | CTDE with per-agent critic seeing all observations + actions. |
| QMIX | "Value decomposition" | Monotonic mixing of per-agent Qs. Cooperative. |
| MAPPO | "Multi-Agent PPO" | PPO with centralized value function. 2026 default baseline. |
| Value decomposition | "Sum of individual Qs" | Joint Q represented as a monotone function of per-agent Qs. |
| Non-stationarity | "Moving targets" | Each agent's env changes as others learn. |
| SMAC | "StarCraft Multi-Agent Challenge" | Cooperative micromanagement benchmark. |

## Further Reading

- [Lowe et al. — MADDPG](https://arxiv.org/abs/1706.02275)
- [Rashid et al. — QMIX](https://arxiv.org/abs/1803.11485)
- [Yu et al. — MAPPO](https://arxiv.org/abs/2103.01955)
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/)
- [SMAC repository](https://github.com/oxwhirl/smac)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo)
