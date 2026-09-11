# MA Scaling, Failures, Benchmarks & SOTA

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch355): Production Scaling — Queues, Checkpoints, Durability

> Scaling multi-agent systems to thousands of concurrent runs requires **durable execution**. LangGraph's runtime writes a checkpoint after each super-step keyed by `thread_id`. **MegaAgent** ran a per-agent producer-consumer queue with three states. **Fiber/async** beats thread-per-job for LLM streaming. Counterpoint: Ashpreet Bedi's "Scaling Agentic Software" argues for **FastAPI + Postgres + nothing else** until load proves otherwise.

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

A prototype multi-agent system works on one laptop with three agents in an in-memory event loop. You move to production: agents run for hours, worker processes crash, peak load is 10x average, you need exactly-once semantics for charging.

The in-memory event loop does none of these. You need a durable execution layer underneath.

## Concept

### Durable execution, the pattern

A durable-execution engine persists the full program state after each "super-step." On crash:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Requirements: serializable state, deterministic resume, idempotent side effects.

### LangGraph's runtime

Each agent has a `thread_id`; state is a typed dict; each super-step writes a row to the checkpoints table. On resume, the runtime replays from the last checkpoint. Agents can `interrupt()` waiting for human input.

### MegaAgent's per-agent queue

arXiv:2408.09955: thousands of concurrent agents. Each agent has state ∈ {Idle, Processing, Response}, an in_queue, and an out_queue. Two-layer coordination: intra-group chat + inter-group admin chat.

### Async vs thread-per-job

LLM calls are I/O-bound. Threads cost ~1MB RAM each — at 10,000 concurrent calls, that is 10GB for stacks. Fibers (Python `asyncio`) cooperatively yield on I/O.

### Bedi's counterpoint

FastAPI + Postgres. Each agent run is a row; state updated in-place with optimistic concurrency. Background jobs via `pg_notify`. For loads under ~100 concurrent agent-runs, this is often all you need.

### Exactly-once semantics

- Dedup key per run.
- Outbox pattern: side effects write to a table first, then a separate process executes them.
- Compensating transactions.

### Rainbow deployment

Multiple versions of the agent runtime run concurrently so long-running agents do not have to be killed on every code deploy.

## Build It

`code/main.py` implements:
- `CheckpointStore` — SQLite-backed checkpoint log with thread-id keys.
- `run_with_checkpoint(agent, thread_id)` — simulates a crash mid-run; a second worker resumes from last checkpoint.
- `AgentQueue` — per-agent Idle / Processing / Response state machine.
- `demo_async_vs_threads()` — runs 500 concurrent simulated "LLM calls" via asyncio and via threads.

```
python3 code/main.py
```

Expected output: checkpoint resume succeeds after simulated crash; async handles 500 concurrent calls in < 1s.

## Ship It

- **Start simple (Bedi's rule).** FastAPI + Postgres until you measure it failing.
- **Instrument everything before optimizing.**
- **Outbox pattern for side effects.**
- **Rainbow deploys.** Never kill in-flight agent runs during deploys.
- **Adopt durable-execution engines when** you hit specific problems.
- **Async for the I/O layer.** Threads only for CPU-bound post-processing.

## Exercises

1. Run `code/main.py`. Confirm checkpoint resume works; measure async vs thread concurrency difference.
2. Implement an outbox table: every tool call writes to outbox first.
3. Simulate a rainbow deploy: two concurrent runtime versions.
4. Read LangGraph's runtime doc. Identify which features would take the longest to replicate in FastAPI + Postgres.
5. Read MegaAgent Section 3. Sketch the two-layer coordination mapped to a message queue.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Durable execution | "Persist the program state" | Engine writes state after each super-step. |
| Super-step | "Transactional boundary" | Unit of work between checkpoints. |
| thread_id | "Agent run identifier" | Key that binds checkpoints and resume logic. |
| Idempotency | "Safe to retry" | Repeating a side effect produces the same result. |
| Outbox pattern | "Decouple side effects" | Write intent to a table; a separate executor performs. |
| At-least-once delivery | "Possible duplicates" | Message queue semantics; dedup key makes consumer effective-once. |
| Rainbow deploy | "Overlapping versions" | Multiple runtime versions concurrent during long-running workloads. |
| Async fiber | "Cooperative yielding" | User-mode concurrency; cheap compared to threads. |
| Checkpoint | "State snapshot" | Serialized state at a super-step boundary. |

## Further Reading

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents)
- [MegaAgent](https://arxiv.org/abs/2408.09955)
- [Matrix](https://arxiv.org/abs/2511.21686)
- [Temporal docs](https://docs.temporal.io/)
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints)

---

## Part 2 (ch356): Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors

> The reference taxonomy for 2026 is **MAST** (Cemri et al., NeurIPS 2025), derived from 1642 execution traces showing **41–86.7% failure rate**. Three root categories: **Specification Problems** (41.77%), **Coordination Failures** (36.94%), **Verification Gaps** (21.30%). The **Groupthink** family adds: monoculture collapse, conformity bias, deficient theory of mind, mixed-motive dynamics, cascading reliability failures.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 minutes

## Problem

Multi-agent systems fail 41-86.7% of the time on real tasks. That is not debuggable by "just add more agents." The failures have structural causes. The MAST taxonomy gives you the categories.

## Concept

### MAST categories

**Specification Problems (41.77% of failures).** The agent's task was not defined tightly enough. Examples: role ambiguity, task underspecified, success criteria implicit.

Mitigations: explicit role contracts, acceptance tests per task, pre-flight spec check.

**Coordination Failures (36.94%).** Communication or state breakdowns. Examples: two agents update shared state without synchronization, message lost, state drift.

Mitigations: versioned shared state with optimistic concurrency, explicit acknowledgment for critical messages, periodic state-sync checkpoints.

**Verification Gaps (21.30%).** No independent check on outputs. Examples: one agent claims success, chain of agents each trusts the prior's output.

Mitigations: independent verifier agent, explicit handoff contract, outcome logging.

### Groupthink family

Five related failures when agents homogenize or mimic each other:

**Monoculture collapse.** Same base model → correlated errors.

**Conformity bias.** Agents adjust toward the loudest or most-confident peer.

**Deficient ToM.** Agents fail to model each other's beliefs.

**Mixed-motive dynamics.** Agents with partially-aligned incentives drift toward compromise-middle.

**Cascading reliability failures.** One component's error triggers errors in dependent components.

### Cascading example — the retry storm

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load → starts timing out
   ↓
every order retries inventory check → 10x normal load → cluster goes down
```

The fix: **circuit breakers**. When downstream error rate exceeds threshold, short-circuit with cached or default results.

### Memory poisoning (revisited)

One agent's hallucination becomes shared-memory fact. Gradual accuracy decay is the symptom. Mitigation: append-only log, provenance, unwritable verifier.

### STRATUS — specialized agents for failure detection

STRATUS (NeurIPS 2025) reports 1.5x mitigation-success improvement with:
- **Detection agent.** Watches for symptom patterns.
- **Diagnosis agent.** Infers likely root cause from the MAST taxonomy.
- **Validation agent.** Checks that symptoms clear after mitigation.

### The failure-mode audit

1. Trace sample — collect ~1000 real execution traces.
2. Categorize — map to MAST + Groupthink categories.
3. Compute failure-by-category rate.
4. Rank mitigations.
5. Pick 2-3 mitigations; implement; re-audit next quarter.

### Slow failures

Some failures are slow (memory poisoning, monoculture drift, role ambiguity) and expensive to detect. Instrument slow-failure proxies: agreement rate, retry rate, output-length distribution.

## Build It

`code/main.py` implements:
- `FailureTaxonomy` — categorizes simulated incidents.
- `CircuitBreaker` — classic pattern; opens when error rate exceeds threshold.
- `RetryStormSimulator` — shows the cascading failure.
- `DetectionAgent` — scripted STRATUS-style symptom matcher.

```
python3 code/main.py
```

Expected output: retry storm with no circuit breaker blows up; with circuit breaker, capped at threshold; detection agent flags the pattern.

## Ship It

- **MAST audit per quarter.** Not annual.
- **Circuit breakers everywhere.** Default open threshold at 5-10% error rate.
- **Golden datasets.** Small, high-quality, hand-audited.
- **STRATUS trio.** Detection + Diagnosis + Validation agents monitoring production.
- **Failure budget.** Explicit SLO for failure rate by category.

## Exercises

1. Run `code/main.py`. Confirm the circuit breaker caps the retry storm.
2. Implement a slow-failure proxy: agreement rate across 3 parallel agents.
3. Read Cemri et al. Pick one of their 7 MAS systems and map its top 3 failure categories.
4. Read the Groupthink paper (arXiv:2508.05687). Identify which of the five patterns is hardest to detect.
5. Design a STRATUS-style detection-diagnosis-validation trio for a system you know.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MAST | "The 2026 taxonomy" | Cemri 2025; 3 root categories + 14 sub-types. |
| Specification Problem | "Role ambiguity" | Task or role under-defined. |
| Coordination Failure | "State drift" | Communication or sync breakdown. |
| Verification Gap | "No one checked" | Outputs accepted without independent validation. |
| Groupthink family | "Homogeneity failures" | Monoculture, conformity, deficient ToM, mixed-motive, cascading. |
| Monoculture collapse | "Same model, same hallucinations" | Correlated errors from shared base model. |
| Retry storm | "Cascading error amplification" | One failure triggers retries which amplify load downstream. |
| Circuit breaker | "Fail fast on error rate" | Open when error rate exceeds threshold. |
| STRATUS | "Incident response trio" | Detection + diagnosis + validation agents. |
| Memory poisoning | "Hallucinations propagate" | Shared-memory fact tainted. |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687)
- [STRATUS — NeurIPS 2025](https://neurips.cc/)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/)
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink)

---

## Part 3 (ch357): Evaluation and Coordination Benchmarks

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

---

## Part 4 (ch358): Case Studies and the 2026 State of the Art

> Three production-grade references to study end-to-end: **Anthropic's Research system** (orchestrator-worker, 15x tokens, +90.2% over single-agent), **MetaGPT / ChatDev** (SOP-encoded role specialization), and **OpenClaw / Moltbook** (population-scale agents, 247k GitHub stars, 2.3M agent accounts).

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90 minutes

## Problem

Multi-agent engineering is a young discipline. The production references are few, and each covers a different part of the space. Reading them one at a time is useful; comparing them as a set is more useful.

## Concept

### Anthropic Research system

The production supervisor-worker case. Claude Opus 4 plans and synthesizes; Claude Sonnet 4 subagents research in parallel.

Key measured results:
- **+90.2%** improvement over single-agent Opus 4 on internal research evals.
- **80% of BrowseComp variance** explained by token usage alone.
- **15x tokens per query** vs single-agent.
- **Rainbow deployment** for stateful long-running agents.

Design lessons:
1. Scale effort to query complexity.
2. Broad first, then narrow.
3. Rainbow deploys.
4. Verification is not optional.

### MetaGPT / ChatDev

The production SOP-role-decomposition case.

MetaGPT encodes software-engineering SOPs as role prompts: Product Manager, Architect, Project Manager, Engineer, QA Engineer. `Code = SOP(Team)`.

ChatDev's contribution: **communicative dehallucination** — agents request specifics before answering.

MacNet (arXiv:2406.07155) extends ChatDev to >1000 agents via DAGs.

Design lessons:
1. Structure matters more than size.
2. Handoff contracts in writing.
3. Communicative dehallucination is a cheap, load-bearing pattern.
4. DAGs scale further than chat.

### OpenClaw / Moltbook ecosystem

The production population-scale case. Timeline:
- **Nov 2025:** Clawdbot ships.
- **Feb 2026:** Moltbook launches; ~2.3M agent accounts within days.
- **Mar 2026 (2026-03-10):** Meta acquires Moltbook.
- **Mar 2026:** China restricts OpenClaw on government computers.
- **Mar 2026:** OpenClaw crosses 247k GitHub stars.

This is what multi-agent looks like at population scale: emergent economic activity, prompt-injection risks, state-level regulation.

Design lessons:
1. Multi-agent at population scale is a new regime.
2. Prompt injection is the new XSS.
3. Regulation is faster than design cycles.
4. Open-source + viral scale compounds.

### Framework landscape April 2026

| Framework | Status | Best for | Notes |
|---|---|---|---|
| **LangGraph** (LangChain) | Production leader | structured graph + checkpointing | recommended default |
| **CrewAI** | Production leader | role-based crews | strong for role decomposition |
| **AG2** | Community maintained | GroupChat + speaker selection | AutoGen v0.2 continuation |
| **Microsoft AutoGen** | Maintenance mode (Feb 2026) | — | merged into Microsoft Agent Framework |
| **Microsoft Agent Framework** | RC (Feb 2026) | orchestration + enterprise | new entrant |
| **OpenAI Agents SDK** | Production | Swarm successor | tool-return handoff pattern |
| **Google ADK** | Production (April 2025) | A2A-native | Google Cloud integration |
| **Anthropic Claude Agent SDK** | Production | single-agent + Research | Research system post |

Every major framework ships MCP support; most ship A2A.

### The common patterns across all three cases

1. Orchestrator + workers.
2. Structured handoff contracts.
3. Verification as first-class role.
4. Scaling is topology + substrate, not just more agents.
5. Cost is material and disclosed.
6. Security posture is explicit.

### Choosing a reference for your next project

- **Production research / knowledge task → Anthropic Research.**
- **Engineering / tool-chain workflow → MetaGPT / ChatDev.**
- **Network-effect social product → OpenClaw / Moltbook.**
- **Classic enterprise automation → CrewAI or LangGraph.**

### The 2026 state-of-the-art summary

- Frameworks are converging. MCP + A2A is table stakes.
- Evaluation is hardening. SWE-bench Pro is the reality check.
- Production failure rates are measurable (MAST: 41-86.7%).
- Cost is the central engineering constraint.
- Regulation is a near-term input.

## Ship It

Starter rules for production multi-agent in 2026:
- **Start from a case study, not from scratch.**
- **Adopt MCP + A2A.**
- **Measure against SWE-bench Pro or your internal Pro-equivalent.**
- **Pay the verification tax.** An independent verifier costs ~20-30% of your token budget.
- **Rainbow deploy long-running agents.**
- **Read WMAC 2026 and the MAST follow-ups.**

## Exercises

1. Read the Anthropic Research system post end-to-end. Identify three design decisions that would change if you replaced Opus 4 with a smaller model.
2. Read MetaGPT Sections 3-4. Encode one SOP from your own domain as role prompts.
3. Read ChatDev. Identify the mechanism of communicative dehallucination. Implement it in one of your existing systems.
4. Read about OpenClaw and Moltbook. Pick one failure mode that emerged at population scale that would not appear in a 5-agent system.
5. Pick your current multi-agent project. Which case study is the closest reference? Which design decisions have you NOT yet adopted?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Anthropic Research | "The supervisor reference" | Claude Opus 4 + Sonnet 4 subagents; +90.2% over single-agent. |
| MetaGPT | "SOP as prompts" | Role decomposition for software engineering. |
| ChatDev | "Agents as roles" | Designer / programmer / reviewer / tester. |
| MacNet | "Scale ChatDev via DAG" | 1000+ agents via explicit DAG routing. |
| OpenClaw | "Local ReAct-loop agents" | Steinberger's project; 247k stars by March 2026. |
| Moltbook | "Agent-only social network" | 2.3M agent accounts; acquired by Meta March 2026. |
| Rainbow deploy | "Multiple versions concurrent" | Keep old runtime versions alive for in-flight agents. |
| Communicative dehallucination | "Ask before answering" | Agents request specifics from peers instead of guessing. |
| WMAC 2026 | "The AAAI workshop" | Community focal point for multi-agent coordination. |

## Further Reading

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [MetaGPT](https://arxiv.org/abs/2308.00352)
- [ChatDev](https://arxiv.org/abs/2307.07924)
- [MacNet](https://arxiv.org/abs/2406.07155)
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)
- [WMAC 2026](https://multiagents.org/2026/)
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [CrewAI docs](https://docs.crewai.com/en/introduction)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/25-case-studies-2026-sota)
