# Swarm Optimization for LLMs (PSO, ACO)

> Bio-inspired optimization is making an LLM comeback. **LMPSO** uses PSO where each particle's velocity is a prompt; **Model Swarms** treats each LLM expert as a PSO particle on a model-weight manifold (13.3% average gain); **AMRO-S** is ACO-inspired pheromone specialists for multi-agent LLM routing (4.7x speedup).

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## Problem

You have a prompt that scores 62% on your task eval. You want to improve it. The naive move is gradient-free manual tweaking. Reinforcement learning needs reward signals. Backprop through prompts is not really possible.

Classical bio-inspired optimization — PSO for continuous search spaces, ACO for path selection — was designed exactly for this regime: gradient-free, population-based, cheap per evaluation.

## Concept

### PSO refresher (Kennedy & Eberhart 1995)

Particle Swarm Optimization: population of particles in a continuous search space. Each particle has position `x_i` and velocity `v_i`. Each iteration:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

### PSO on LLM outputs — LMPSO

arXiv:2504.09247 adapts PSO for LLM-generated structured outputs. Each particle is a candidate output. Velocity is a *prompt* that describes how to modify the current output toward the personal/global best. Works well when the output is structured and fitness is automatic.

### Model Swarms

arXiv:2410.11163 takes PSO into the *model* layer. Each "particle" is an expert LLM. The swarm moves parameters toward the collective best via gradient-free update. 13.3% average gain over 12 baselines on 9 datasets.

### ACO refresher (Dorigo 1992)

Ant Colony Optimization: ants traverse a graph; each path has a pheromone trail. Ant move probabilities weight by pheromone strength. Ants that complete the task deposit pheromone proportional to solution quality.

### AMRO-S — ACO for agent routing

arXiv:2603.12933 uses ACO for multi-agent routing. Pheromones strengthen routes that produce good outputs. Quality-gated asynchronous update decouples inference from learning. 4.7x speedup.

### When to use PSO / ACO for LLMs

**Use PSO when:** search space is continuous or maps to continuous parameters, fitness is cheap and automatic, population can be small (10-30).

**Use ACO when:** you have a routing or path-selection problem, decisions reinforce over time, you need interpretable evidence for routing decisions.

### Why bio-inspired still wins

PSO and ACO need only an *evaluator* function. If you can score a candidate output or a routing decision, you can optimize over the space.

### Practical limits

- Population budget: N particles × T iterations × per-eval cost.
- Exploration vs exploitation: pheromone decay rate and PSO inertia trade off.
- Catastrophic drift: both algorithms can converge and then diverge if the fitness landscape shifts.

## Build It

`code/main.py` implements:
- `LMPSO` — PSO over numeric prompt parameters (temperature, top_k weights). Runs for 30 iterations and shows g_best convergence.
- `AMRO_S` — ACO-style routing. 3 agents, 4 task types, pheromone matrix, 100 routed tasks.
- Comparison: random routing vs ACO routing on the same task stream.

```
python3 code/main.py
```

Expected output: LMPSO g_best improves from random to near-optimal over 30 iterations. ACO routing beats random by ~30-40% on quality.

## Ship It

- **Start small.** 10-20 particles, 20-50 iterations.
- **Log pheromones or g_best per iteration.**
- **Quality-gate updates.** Especially for ACO routing.
- **Reset decay on distribution shift.**
- **Cap the per-iteration cost.**

## Exercises

1. Run `code/main.py`. Observe LMPSO convergence. Vary population size.
2. Implement a "catastrophic drift" experiment: change the fitness function after iteration 30.
3. Add a quality gate to AMRO-S: pheromone deposit only on runs with eval score > 0.7.
4. Read LMPSO (arXiv:2504.09247). Map the paper's "velocity as a prompt" back to your numeric velocity.
5. Read AMRO-S (arXiv:2603.12933). Implement the decoupled "inference fast-path" with asynchronous pheromone update.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PSO | "Particle Swarm Optimization" | Kennedy-Eberhart 1995. Population-based gradient-free optimizer. |
| ACO | "Ant Colony Optimization" | Dorigo 1992. Path/route optimization via pheromone trails. |
| LMPSO | "PSO with LLM generation" | arXiv:2504.09247. Velocity is a prompt. |
| Model Swarms | "PSO on expert weights" | arXiv:2410.11163. Gradient-free update on model parameter subspace. |
| AMRO-S | "ACO for agent routing" | arXiv:2603.12933. Pheromone matrix over task-type × agent. |
| p_best / g_best | "Personal / global best" | Per-particle and swarm-wide best solutions found so far. |
| Pheromone | "Routing memory" | Strength on an edge; decays over time. |
| Quality-gated update | "Only learn from good runs" | Pheromone deposit conditioned on quality check. |
| Catastrophic drift | "Distribution shift" | Fitness landscape changes; old p_best and pheromones become stale. |

## Further Reading

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968)
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html)
- [LMPSO](https://arxiv.org/abs/2504.09247)
- [Model Swarms](https://arxiv.org/abs/2410.11163)
- [AMRO-S](https://arxiv.org/abs/2603.12933)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco)
