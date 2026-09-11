# Roles, Swarms & Voting Topologies

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch340): Society of Mind and Multi-Agent Debate

> Minsky's 1986 premise — intelligence is a society of specialists — gets rediscovered every decade. In 2023 Du et al. turned it into a concrete algorithm: multiple LLM instances propose answers, read each other's answers, critique, and update. Over N rounds they converge on a consensus that beats zero-shot CoT and reflection.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Self-consistency — sample one model many times and take the majority answer — is the cheapest reasoning improvement you can bolt on. It works, but it saturates fast. You can double your samples and not see another meaningful jump.

Debate breaks the saturation. Instead of N independent samples from one model, N agents read each other's reasoning and revise. The correlation between samples drops, and the convergence point is often correct where i.i.d. voting was confidently wrong.

## Concept

### The Du et al. 2023 algorithm

From arXiv:2305.14325 (ICML 2024):

1. Each of N agents produces an initial answer to the question.
2. For round r = 2..R: each agent is shown the other agents' round r-1 answers and asked "considering these, give your updated answer."
3. After R rounds, majority-vote the final answers.

The paper tests on MMLU, GSM8K, biographies, MATH, and factuality benchmarks. Debate consistently beats CoT and Self-Reflection.

### Two independent knobs

- **Agent count alone** (1 round, majority vote of N) beats single-agent on most tasks, but plateaus.
- **Round count alone** (1 agent seeing its own prior reasoning) barely helps.
- **Both together** produces the big jumps.

### Why it works

1. **Exposure to disagreement.** When an agent sees another agent's reasoning chain with a different conclusion, it has to either justify or update.
2. **Correlated error reduction.** In self-consistency, all samples come from the same model, so the errors correlate. Different debated views decorrelate further.

### Heterogeneous debate

A-HMAD and related follow-ups use *different base models* for different agents. Llama + Claude + GPT debating reduces monoculture collapse because the correlated errors of one model family are not shared by the others.

Downside: a weak model participating in a debate can drag the consensus toward its wrong answer.

### NLSOM — the 129-agent extension

Zhuge et al. ("Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066) scaled this idea to 129-member societies. Specialization and self-organization emerge with scale.

### Failure modes

- **Sycophancy cascade.** All agents defer to whichever agent sounds most confident. Prompting for adversarial roles helps.
- **Topic drift.** Debates over many rounds drift from the original question. Re-inject the question every round.
- **Compute blowup.** N agents × R rounds = N·R LLM calls. A 5-agent, 5-round debate is 25 calls at growing context.

## Build It

`code/main.py` runs a 3-agent × 3-round debate on a math question where each agent starts with a different (possibly wrong) answer. Agents are scripted — each "updates" by averaging the neighbors' answers weighted by a scripted confidence.

Key effects:
- A single round of exchange moves agents closer to the correct answer.
- Extra rounds past round 2 show diminishing returns.

```
python3 code/main.py
```

## Ship It

- **Cap rounds at 3.** Du et al. show 3 rounds capture most of the gain.
- **Cap agents at 5.** Beyond 5, context bloat and cost dominate.
- **Heterogeneous by default.** At least two different base models in the pool.
- **Adversarial slot.** One agent prompted to disagree regardless.
- **Log every round.** Debate systems that hide intermediate rounds cannot be debugged.

## Exercises

1. Run `code/main.py`, then set the round count to 5 and watch diminishing returns.
2. Add a fourth agent with an adversarial role: always disagree with the current majority.
3. Plot the agreement score per round. When does it hit 1.0 and is that equivalent to "correct"?
4. Read Du et al. Section 4 ablations. Replicate the "agents-only" vs "rounds-only" vs "both" result.
5. Read "Should we be going MAD?" (arXiv:2311.17371) and list two debate variants beyond round-robin.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Society of Mind | "Minsky's idea" | Intelligence as interacting specialists. |
| Multi-agent debate | "Agents argue" | N agents propose, critique each other, revise over R rounds, majority-vote. |
| Consensus | "They agree" | Not epistemic truth — just fraction-on-majority-answer. |
| Rounds | "Exchange steps" | One round = each agent reads the others and updates once. |
| Heterogeneous debate | "Mix model families" | Using different base models to decorrelate errors. |
| Sycophancy cascade | "Everyone agrees with the loud one" | Agents defer to the most confident agent regardless of correctness. |
| NLSOM | "129-agent society" | Natural-language society of mind; Zhuge et al.'s scaled version. |
| Correlated error | "Same model, same bug" | Why self-consistency saturates; debate across different views decorrelates. |

## Further Reading

- [Du et al. — Improving Factuality and Reasoning through Multiagent Debate](https://arxiv.org/abs/2305.14325)
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066)
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371)
- [Debate project page](https://composable-models.github.io/llm_debate/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/07-society-of-mind-debate)

---

## Part 2 (ch341): Role Specialization — Planner, Critic, Executor, Verifier

> The most common multi-agent decomposition in 2026: one agent plans, one executes, one critiques or verifies. MetaGPT formalizes this as SOPs encoded into role prompts. ChatDev chains designer, programmer, reviewer, tester through a "chat chain." The verifier is load-bearing: Cemri et al. (MAST) show every multi-agent failure can be traced to missing or broken verification.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 minutes

## Problem

Generic multi-agent systems produce generic output. Three coders in a group chat write three flavors of the same mediocre code. The fix is not more agents — it is *different* agents. Assign distinct roles. Give the critic tools the planner does not have. Give the verifier an objective test suite.

## Concept

### The four canonical roles

**Planner.** Reads the goal, produces a step list or a spec. Tools: knowledge retrieval, docs. Output: structured plan.

**Executor.** Reads one plan step at a time, produces the artifact. Tools: the actual work tools. Output: the artifact.

**Critic.** Reads the executor's output against the planner's intent. Tools: read-only access. Output: accept/reject with reasons.

**Verifier.** Reads the artifact and runs a deterministic check. Tools: test runner, type checker, schema validator. Output: pass/fail with evidence.

Critic is subjective, opinionated, often LLM-based. Verifier is objective, deterministic, often code-based. They are not the same role.

### MetaGPT's SOP pattern

MetaGPT (arXiv:2308.00352) encodes software engineering SOPs as role prompts:
- **Product Manager** writes the PRD.
- **Architect** produces the system design.
- **Project Manager** splits tasks.
- **Engineer** implements.
- **QA Engineer** runs tests.

Each role has a strict input/output schema. The `Code = SOP(Team)` formulation — deterministic SOPs turn a team of LLMs into a predictable pipeline.

### ChatDev's communicative dehallucination

ChatDev adds: when an executor needs a specific detail that was not in the plan, it explicitly asks the designer before continuing. This prevents the classic LLM failure of plausibly inventing the detail.

### Why verifier matters most

Cemri et al. (MAST) traced 1642 multi-agent execution failures. 21.3% were verification gaps — the system shipped an answer no one had checked. PwC reported (CrewAI deployments, 2025) that adding a structured validation loop moved accuracy from 10% to 70%.

### Critic vs verifier

- A critic is an LLM reviewing an artifact for quality. Subjective. Can be fooled by plausible prose.
- A verifier is a deterministic program running on the artifact. Objective. Gives pass/fail with evidence.

Use both.

### The anti-pattern

Every role in your system is an LLM and every role's output is "looks good to me." Classic MAST failure mode. Add at least one verifier whose pass/fail is decided by code, not by an LLM.

## Build It

`code/main.py` implements a 4-role pipeline building a simple Python function:
- **Planner** produces a spec.
- **Executor** generates a code string.
- **Critic** (LLM-simulated) flags obvious issues.
- **Verifier** runs the generated code in a sandbox (`exec`) against a test case.

Demo runs twice: once where the executor produces correct code (critic + verifier both pass), once where the executor produces off-spec code (critic misses the bug because it looks plausible, verifier catches it because the test fails).

```
python3 code/main.py
```

## Ship It

Checklist:
- **At least one deterministic verifier.** Never all-LLM.
- **Explicit I/O schema per role.** The planner returns a spec, not prose.
- **Communicative dehallucination.** Executor must ask the planner when info is missing.
- **Critic/verifier ordering.** Run critic first (cheap), verifier second (slow).
- **Loop budget.** Max 2 critic-executor revision rounds before escalating to human.

## Exercises

1. Run `code/main.py` and observe how the verifier catches the bug the critic missed. Add a static-analysis check as an additional verifier.
2. Add a 5th role: "requirements analyst" that translates user wish into planner-ready spec.
3. Read MetaGPT Section 3 ("Agents"). List the input/output schema of each of MetaGPT's 5 roles.
4. Read ChatDev's chat-chain diagram (arXiv:2307.07924 Figure 3). Identify where communicative dehallucination breaks a loop that would otherwise be infinite.
5. PwC's 7x accuracy gain came from verification loops. Hypothesize three tasks where adding a verifier would not help.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Role specialization | "Different agents, different jobs" | Distinct system prompts tuned for planner/executor/critic/verifier roles. |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT's framing: strict I/O schemas per role. |
| Communicative dehallucination | "Ask before inventing" | ChatDev pattern: executor asks planner when a detail is missing. |
| Critic | "LLM reviewer" | Subjective, opinionated reviewer. |
| Verifier | "Deterministic check" | Code-based pass/fail. Test runner, type checker. |
| Verification gap | "No one checked" | 21.3% of MAST failures. |
| Revision loop | "Critic sends it back" | Critic rejection triggers executor re-run with feedback. |
| All-LLM anti-pattern | "Looks good to me" | Every role is an LLM, no deterministic check. |

## Further Reading

- [Hong et al. — MetaGPT](https://arxiv.org/abs/2308.00352)
- [Qian et al. — ChatDev](https://arxiv.org/abs/2307.07924)
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/08-role-specialization)

---

## Part 3 (ch342): Parallel / Swarm / Networked Architectures

> Contrast with supervisor: no central decider. Agents read a shared event bus, pick up work asynchronously, write results back. The tradeoff is explicit: determinism and traceability for scalability. Swarm fits tasks with many independent sub-problems; it does not fit tasks that need a single coherent plan.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Supervisor scales to a few workers. What about hundreds? The supervisor itself becomes the bottleneck: every decision about who does what funnels through one agent. One slow plan step stalls the whole system.

Swarm architectures flip the design. Instead of a central planner dispatching work, workers pick work off a shared queue. No orchestrator; the system scales until the queue does.

## Concept

### The shape

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

No orchestrator. Each worker repeats: pull a task, process, write result.

### When swarm fits

- **Many independent tasks.** Scraping, transforming, classifying.
- **Variable-duration work.** If some tasks take 100ms and others take 10s, a swarm balances load automatically.
- **Throughput over determinism.** You care about total completion time, not strict ordering.

### When swarm fails

- **Ordered workflows.** If step 3 needs step 2's output, a swarm risks step 3 firing before step 2 is done.
- **Global-plan tasks.** Complex research questions benefit from a planner.
- **Debugging.** With no central log and asynchronous work, reproducing a bug is expensive.

### Matrix (arXiv:2511.21686)

Matrix takes swarm to its natural conclusion: both control flow and data flow are serialized messages on distributed queues. No central coordinator. Fault tolerance comes from message durability. Coordination becomes "what message topic does this agent subscribe to?" rather than "which agent does the supervisor pick next?"

### LangGraph's Swarm Architecture

LangGraph explicitly describes "Swarm Architecture" as one of the multi-agent patterns: agents are nodes, but edges form a directed graph with cycles and any node can be activated from the pool.

### Failure mode: starvation and hot-spotting

If all workers pull the fastest-available task, long-running tasks never get picked until they are the only ones left. Mitigations: priority queues with explicit aging, worker specialization, back-pressure.

## Build It

`code/main.py` implements a swarm of 4 worker threads pulling from a shared `queue.Queue`. Tasks have variable durations. The demo contrasts:
- **Sequential baseline:** one worker processes all tasks serially.
- **Fixed assignment:** each task pre-assigned to a specific worker.
- **Swarm:** workers pull from a shared queue.

Swarm balances load automatically; fixed assignment leaves fast workers idle when their assigned task is slow.

```
python3 code/main.py
```

## Ship It

Checklist:
- **Priority queue with aging.** Prevent long-task starvation.
- **Worker idempotency.** A task may be pulled more than once if a worker crashes mid-run.
- **Durable queue.** Use Kafka, Redis Streams, or a database-backed queue for production.
- **Observability per task.** Every task has a trace ID.
- **Back-pressure.** If the queue grows faster than workers drain it, slow the producer.

## Exercises

1. Run `code/main.py`. How much faster is swarm than sequential on the variable-duration workload?
2. Add a priority queue variant (use `queue.PriorityQueue`). Observe whether low-priority tasks ever starve.
3. Implement a hot-spot detector: log when any worker processes 3x more tasks than the slowest worker.
4. Read the Matrix paper (arXiv:2511.21686) abstract and Section 3. Identify one specific tradeoff Matrix accepts.
5. Convert the swarm demo to use a `queue.Queue` of (task_type, payload) tuples, with workers subscribing only to specific types.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Swarm architecture | "Decentralized agents" | Workers pull from shared queue; no central orchestrator. |
| Event bus | "Agents subscribe to topics" | Message broker that routes tasks to workers by type or content. |
| Starvation | "Task never runs" | Low-priority task never gets picked. |
| Hot-spotting | "One worker drowns" | Load imbalance where one worker gets most tasks. |
| Back-pressure | "Slow down the producer" | Mechanism that signals upstream to stop producing when the queue fills up. |
| Idempotent worker | "Safe to re-run" | A task processed twice produces the same result. |
| Durable queue | "Survives crashes" | Queue backed by disk or replicated storage. |
| Matrix framework | "Full message-passing swarm" | Both data and control flow are serialized messages on distributed queues. |

## Further Reading

- [LangGraph — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686)
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system)
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/09-parallel-swarm-networks)

---

## Part 4 (ch348): Voting, Self-Consistency, and Debate Topology

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
