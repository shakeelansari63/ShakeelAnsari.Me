# Shared Memory and Blackboard Patterns

> Two approaches coexist in 2026 multi-agent systems: the **message pool** (everyone sees everyone's messages) and the **blackboard with subscription** (agents subscribe to relevant events). Both are the only stateful part of a multi-agent system — which means both are where the interesting bugs live. The reference failure mode is **memory poisoning**.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Multi-agent systems need a place for agents to share facts. When one of the agents hallucinates and writes the hallucination to shared state, every downstream agent that reads that state adopts the hallucination as fact. By the time the human notices, the reasoning chain is five steps deep.

This is memory poisoning. It is the second-most-documented failure family in the MAST taxonomy and it is structural: any shared-memory design without provenance and an unwritable verifier will exhibit it eventually.

## Concept

### The two main topologies

**Full message pool.** Every agent reads every message. AutoGen GroupChat and MetaGPT use this. Simple, transparent, inspectable, but does not scale past ~10 agents.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription.** Agents declare interest in topics; the substrate routes only relevant messages. CA-MCP and Matrix use this. Scales further, but requires upfront schema design.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### Memory poisoning, in one scenario

Three agents work on a research task. Agent A hallucinates a decimal (42% instead of 4.2%). Agent B reads and amplifies it. Agent C recommends adoption based on the inflated number. The final report cites a 42% number that never existed.

No agent crashed. No test failed. The hallucination crossed from one agent's context into every downstream agent's reasoning via shared state.

### Three mitigations

1. **Attribute provenance on every write.** Every entry records who wrote it, when, what source they cited.
2. **Version writes; treat them as append-only.** A correction is a new entry that supersedes the old.
3. **Keep at least one agent that cannot write to shared state.** A read-only verifier agent samples entries, re-fetches sources, and flags inconsistencies.

### Blackboard precedent (Hayes-Roth, 1985)

The blackboard pattern predates LLM agents by four decades. Hayes-Roth described specialist Knowledge Sources that observe a global blackboard, contribute partial solutions, and trigger other sources.

### Projection vs full view

A pure blackboard gives every subscriber the same projection (topic-scoped). A more aggressive design is **per-agent projection**: each agent gets a view customized to its role. LangGraph's state reducers are the canonical implementation.

### The unwritable verifier

The most load-bearing mitigation is the read-only verifier. It shares state with the team (reads the blackboard), has no write handle to shared state, independently fetches sources, and routes outputs to a human or separate decision agent.

## Build It

`code/main.py` implements both topologies in stdlib Python plus a toy poisoning attack and the three mitigations.

- `MessagePool` — thread-safe append-only log with full read-out.
- `Blackboard` — topic-keyed pub/sub with per-agent subscriptions.
- `ProvenanceEntry` — every write records (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — runs a three-agent research task where agent A hallucinates a decimal.
- `Verifier` — a read-only agent that re-fetches sources and flags inconsistencies.

Expected output:
- Run 1 (no verifier): the hallucinated 42% propagates to the final report.
- Run 2 (with verifier): the verifier flags the inconsistency.

```
python3 code/main.py
```

## Ship It

For any shared-memory design:
- Record provenance on every write.
- Make the log append-only.
- Deploy at least one read-only verifier agent with independent source access.
- Route verifier output to a separate channel, not back into the shared pool.
- Log the ratio of writes that are supersessions.

## Exercises

1. Run `code/main.py`. Confirm run 1 propagates the hallucination and run 2 catches it.
2. Add a second hallucination: agent B invents a dataset size. The verifier should catch both.
3. Switch the full pool to a blackboard with topic partitions. Which poisoning scenarios does topic partitioning make harder?
4. Read Hayes-Roth (1985, "A Blackboard Architecture for Control"). Identify two control patterns not discussed here.
5. Read CA-MCP (arXiv:2601.11595). Map its Shared Context Store to either the MessagePool or Blackboard class.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Message pool | "Shared chat history" | Append-only log that every agent reads. |
| Blackboard | "Shared workspace" | Topic-keyed pub/sub. Agents subscribe to relevant topics. |
| Provenance | "Who wrote what" | Metadata on each write: writer, timestamp, prompt, sources. |
| Memory poisoning | "Hallucinations spreading" | One agent's error enters shared state, downstream agents adopt it as fact. |
| Append-only | "No in-place updates" | Corrections are new entries that supersede. |
| Unwritable verifier | "Independent auditor" | Read-only agent that re-fetches sources and flags inconsistencies. |
| Projection | "Scoped view" | Per-agent view computed from global state. |
| Knowledge Source | "Specialist agent" | Hayes-Roth's 1985 term for a blackboard participant. |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595)
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686)
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/13-shared-memory-blackboard)
