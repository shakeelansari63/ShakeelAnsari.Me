# Generative Agents and Emergent Simulation

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
