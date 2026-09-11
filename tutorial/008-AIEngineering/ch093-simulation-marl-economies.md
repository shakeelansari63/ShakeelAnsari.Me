# Generative Agents, MARL & Agent Economies

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch350): Generative Agents and Emergent Simulation

> Park et al. 2023 populated **Smallville**, a sandbox of 25 agents, with a three-part architecture: **memory stream** (natural-language log), **reflection** (higher-level syntheses), and **plan** (day-level behavior). The landmark result was the Valentine's Day party emergence: one agent seeded with "wants to throw a Valentine's Day party" produced invitations spread through the population, coordinated dates, and the party happened.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

Most multi-agent systems are tightly-scripted teams: planner plans, coder codes, reviewer reviews. That does not capture the emergent, unscripted behavior that arises when agents have memory, priorities, and an open world.

The Smallville architecture is the benchmark for it. If you build an agent simulation in 2026, you are either using Smallville's three components or explicitly justifying why you are not.

## Concept

### The three components

**Memory stream.** An append-only log of observations, actions, reflections, and plans. Each entry has a timestamp, type, description, and derived metadata: **recency**, **importance** (self-rated 1-10), and **relevance** (cosine similarity to current query).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Memory retrieval combines the three scores: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`.

**Reflection.** Periodically, the agent generates higher-order syntheses from recent memories. Reflection entries go back into the stream and are retrievable like any other memory.

**Plan.** Top-down decomposition: day-level, hour-level, action-level. Plans are revisable when observations contradict them.

### Why all three matter (ablation)

Without observation: stale beliefs. Without reflection: shallow interactions. Without plan: reactive noise. All three are required for believability.

### The Valentine's Day emergence

One agent, Isabella Rodriguez, is seeded with "wants to throw a Valentine's Day party at Hobbs Cafe on Feb 14 at 5pm." The 24 other agents receive no such seed. Over simulated days, invitations spread through bilateral conversations until multiple agents converge at Hobbs Cafe at 5pm. This is emergence: system-level behavior from local interactions without a central orchestrator.

### The documented failure modes

- **Spatial norm errors.** Agents walk into closed stores, use the same single-person bathroom.
- **Memory overflow.** Deep simulation runs cause memory-retrieval cost to grow.
- **Reflection hallucination.** Reflections can invent relationships that do not exist in the memory stream.

### Implementation rules

1. Memory is append-only.
2. Importance scores are cheap — call the LLM at write time.
3. Retrieval is ranked, not filtered.
4. Reflection runs periodically (when sum of importance of unprocessed memories exceeds a threshold).
5. Plans are revisable — regenerate the affected segment only.

## Build It

`code/main.py` implements the three components in stdlib Python with scripted agent policies. The demo reproduces the Valentine's-party emergence in miniature: 5 agents, Agent 1 starts with "throw party at 5pm," over simulated ticks the invitation spreads and agents converge.

```
python3 code/main.py
```

Expected output: tick-by-tick trace. By the final tick, at least 3 of the 5 agents show the party in their plan.

## Ship It

- **Memory is the database.** Pick a real store at scale.
- **Log the retrieval trace.** For every action, log the top-k memories that drove it.
- **Budget per-agent tokens.** N agents × T ticks × calls-per-tick can dwarf your budget.
- **Compact memory periodically.** Summarize-and-prune low-importance entries.
- **Detect spatial/social norm violations** explicitly.

## Exercises

1. Run `code/main.py`. Confirm 3+ agents converge at the party. Increase agents to 10.
2. Remove the reflection step. What does behavior look like?
3. Introduce a competing seeded goal ("Klaus wants to give a research talk at 5pm").
4. Add spatial constraints: Hobbs Cafe holds at most 4 agents.
5. Read Park et al. Section 6. Identify one behavior not reproducible in your miniature.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory stream | "The agent's diary" | Append-only log of observations, actions, reflections, plans. |
| Recency | "How new is the memory" | Exponential-decay score by age. |
| Importance | "How much does the agent care" | Self-rated 1-10 at write time. |
| Relevance | "How related to the current query" | Cosine similarity (embedding-based). |
| Reflection | "Higher-order belief" | Synthesis generated from recent memories. |
| Plan | "Day/hour/action decomposition" | Top-down plan tree. Revisable. |
| Smallville | "Park 2023's sandbox" | 25-agent simulation. |
| Believability | "The quality metric" | Human-rater score for plausible behavior. |

## Further Reading

- [Park et al. — Generative Agents](https://arxiv.org/abs/2304.03442)
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763)
- [Smallville code release](https://github.com/joonspk-research/generative_agents)
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/17-generative-agents-simulation)

---

## Part 2 (ch351): Theory of Mind and Emergent Coordination

> Li et al. showed that LLM agents in a cooperative text game exhibit **emergent high-order Theory of Mind** (ToM) — reasoning about what another agent believes about a third agent's beliefs — but fail on long-horizon planning. Riedl measured higher-order synergy across a population and found that **only** the ToM-prompt condition produces identity-linked differentiation and goal-directed complementarity.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## Problem

Multi-agent coordination often looks magical: agents divide labor, anticipate each other, avoid redundancy. Usually this "emergence" is an artifact of prompt engineering. Remove the prompt, remove the coordination.

Riedl's 2025 finding is stricter: under controlled conditions, coordination only emerges when agents are prompted to reason about **other agents' minds** (ToM). Without the ToM prompt, even strong models show coordination patterns that do not survive statistical controls.

## Concept

### What ToM means

- **Zeroth-order:** no model of others. The agent acts on its own observations only.
- **First-order:** the agent has a model of each other agent's beliefs. "Alice believes X."
- **Second-order:** the agent models recursive beliefs. "Alice believes that Bob believes X."

Li et al. 2023 found that first- and second-order ToM emerge in LLM agents in cooperative games but degrade with long horizon and unreliable communication.

### The Sally-Anne test

A 1985 false-belief test: Sally puts a marble in basket A, leaves. Anne moves it to basket B. Where will Sally look when she returns? GPT-4-era LLMs pass this when posed plainly. They fail when the narrative is long or the question is phrased indirectly.

### Riedl's coordination measurement

Riedl built a population-scale test: N agents, a cooperative objective, variable prompt conditions. Measure:
1. **Identity-linked differentiation.** Do agents develop stable role distinctions over time?
2. **Goal-directed complementarity.** Do agents' actions complement each other?
3. **Higher-order synergy.** Does the group achieve what no subset could?

Result: only under the ToM prompt condition do all three metrics produce signal above baseline.

### The coordination illusion

Without statistical controls, "emergent coordination" often reflects prompt engineering, observer bias, or post-hoc selection of successful runs.

### A minimal ToM-aware agent

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]
```

The `other_models` attribute is the ToM state.

### Why long-horizon hurts

Context limits cause agents to forget which belief belongs to whom. Hallucination adds false beliefs. Mitigations: explicit ToM state in the prompt, shorter reasoning chains, external ToM store.

### Where ToM fails in production

- Adversarial settings (agents with good ToM are easier to manipulate).
- Heterogeneous teams (different models, ToM does not generalize).
- Ground-truth-dependent tasks (ToM can be a distraction).

## Build It

`code/main.py` implements:
- `ToMAgent` — tracks own beliefs and per-other-agent belief models.
- A cooperative task: three agents must collect three tokens from three boxes. Agents cannot communicate; they infer intent from each other's actions.
- Two configurations: `zeroth_order` (no ToM) and `first_order` (ToM with one-level belief model).
- Measurement over 200 randomized trials: completion rate, duplication rate, average turns.

```
python3 code/main.py
```

Expected output: zeroth-order agents duplicate effort at ~35% rate and complete ~60% of trials. First-order ToM agents duplicate at ~5% and complete ~95%.

## Ship It

Coordination claims checklist:
- **Control condition.** A version without the coordination prompt.
- **Statistical test.** Is the difference significant at p < 0.05?
- **Complementarity measure.** Action-disjointness over time.
- **Failure-case log.** What does the ToM state look like when agents miscoordinate?
- **Model-capacity disclosure.** Does the effect vanish on smaller models?

## Exercises

1. Run `code/main.py`. Confirm first-order ToM reduces duplication rate by ~7x.
2. Implement second-order ToM (agent A models what B thinks about C).
3. Inject a hallucination into the ToM state: randomly flip one belief per turn.
4. Read Li et al. Reproduce the "long-horizon degradation" finding.
5. Read Riedl 2025. Implement the higher-order synergy statistic on your simulation logs.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | "Understanding others' minds" | The capacity to model another agent's beliefs. |
| Sally-Anne test | "The false-belief test" | 1985 developmental psychology test. |
| First-order ToM | "A believes X" | Modeling one other's beliefs about facts. |
| Second-order ToM | "A believes B believes X" | Recursive modeling one level deeper. |
| Identity-linked differentiation | "Stable roles over time" | Riedl's metric: roles persist, not random. |
| Goal-directed complementarity | "Disjoint actions" | Agents target different subtasks. |
| Higher-order synergy | "Group exceeds any subset" | Riedl's statistical measure for real coordination. |
| Coordination illusion | "It looks coordinated" | Prompt-dressed appearance without measurable signal. |

## Further Reading

- [Li et al. — Theory of Mind for Multi-Agent Collaboration](https://arxiv.org/abs/2310.10701)
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174)
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595)
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination)

---

## Part 3 (ch352): Swarm Optimization for LLMs (PSO, ACO)

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

---

## Part 4 (ch353): MARL — MADDPG, QMIX, MAPPO

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

---

## Part 5 (ch354): Agent Economies, Token Incentives, Reputation

> Long-horizon autonomous agents need economic agency. The emerging **5-layer stack** is: **DePIN** (physical compute) → **Identity** (W3C DIDs + reputation capital) → **Cognition** (RAG + MCP) → **Settlement** (account abstraction) → **Governance** (Agentic DAOs). Production agent-incentive networks include Bittensor, Fetch.ai / ASI Alliance, and Gonka.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Multi-agent systems get complicated when agents produce value jointly but need to be rewarded individually. Classical mechanisms — equal split, last-contributor-takes-all — are unfair or gameable.

Beyond credit attribution, the field has turned to actual economic agents: Bittensor TAO rewards mining compute, Fetch.ai/ASI rewards ASI-1 Mini LLM usage with FET tokens, Gonka reallocates transformer proof-of-work toward productive AI tasks.

## Concept

### The 5-layer agent-economy stack

1. **DePIN (physical compute).** Decentralized infrastructure that rents GPU, storage, bandwidth.
2. **Identity.** W3C DIDs give each agent a durable ID. Reputation accrues to the DID.
3. **Cognition.** The agent's reasoning loop: LLM + RAG + MCP.
4. **Settlement.** Account abstraction (ERC-4337) lets agents pay gas from their own balances.
5. **Governance.** Agentic DAOs where humans and agents vote on protocol changes.

### Bittensor, Fetch.ai, Gonka — what runs

**Bittensor (TAO).** Subnets are specialized tasks. Miners submit model outputs. Validators rank them; stake-weighted scoring distributes TAO rewards. Pay for task-specific output quality, not compute used.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM runs on Fetch.ai's network; users pay FET tokens for inference.

**Gonka.** Transformer proof-of-work: the "work" is forward passes of a transformer.

### Shapley-value credit attribution

Three agents collaborate on a task. The output scores 0.8. Who contributed what?

Shapley value: the unique credit allocation satisfying four axioms (efficiency, symmetry, linearity, null). For agent `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

For N=3, 6 permutations. For N=10, 3.6M — so in practice you sample.

### Second-price auction for aggregation

N agents each propose a completion; each has a private value. The auctioneer picks the highest-value proposal and pays the *second-highest* value. Under monotone aggregation, this is truthful.

### Reputation capital

A DID-bound reputation score from confirmed contributions:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

### Where the economics falls apart

- Price oracle manipulation, Sybil attacks, verification cost, regulatory overhang.

## Build It

`code/main.py` implements:
- `shapley(value_fn, agents)` — exact Shapley computation by enumeration for small N.
- `second_price_auction(bids)` — truthful mechanism.
- `Reputation` — DID-bound reputation with exponential decay and slashing.
- Demo 1: three agents collaborate, exact Shapley attributes credit.
- Demo 2: five agents bid for a task slot; second-price auction.
- Demo 3: 100 rounds of task assignment with rep-weighted routing.

```
python3 code/main.py
```

## Ship It

- **Start with reputation, not tokens.**
- **Verify before you reward.** Self-reported quality accrues sybil games.
- **Shapley-sample, not Shapley-exact.** Sample 100-1000 orderings.
- **Cap decay factor and floor reputation.**
- **Audit mechanisms adversarially.**

## Exercises

1. Run `code/main.py`. Confirm Shapley values sum to total value.
2. Implement Shapley sampling (Monte Carlo over K orderings).
3. Implement a coalition-forming step before the auction.
4. Read the Google Research mechanism-design post. Identify one assumption that breaks truthfulness.
5. Read AAMAS 2025 decentralized LaMAS. Implement their Shapley step over 10 agents.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | Token-incentivized compute/storage/bandwidth. |
| DID | "Decentralized identifier" | W3C spec for portable IDs. |
| ERC-4337 | "Account abstraction" | Contract accounts that can sponsor gas. |
| Shapley value | "Fair credit attribution" | Unique allocation satisfying efficiency, symmetry, linearity, null. |
| Second-price auction | "Vickrey auction" | Truthful mechanism: winner pays second-highest bid. |
| Reputation capital | "Accumulated quality score" | DID-bound score from confirmed contributions. |
| Agentic DAO | "Agents + humans govern" | DAO with agent voters as first-class. |

## Further Reading

- [The Agent Economy](https://arxiv.org/abs/2602.14219)
- [Google Research — Mechanism design for LLMs](https://research.google/blog/mechanism-design-for-large-language-models/)
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf)
- [Bittensor TAO documentation](https://docs.bittensor.com/)
- [Fetch.ai / ASI Alliance](https://fetch.ai/)
- [W3C DIDs spec](https://www.w3.org/TR/did-core/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/21-agent-economies)
