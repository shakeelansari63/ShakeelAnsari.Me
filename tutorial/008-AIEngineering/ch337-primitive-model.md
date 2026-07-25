# The Multi-Agent Primitive Model

> Every multi-agent framework shipping in 2026 — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework — is a point in a four-dimensional design space. Four primitives, nothing more: the agent, the handoff, the shared state, the orchestrator.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## Problem

Every six months a new multi-agent framework ships. Each press release claims to be "the right abstraction." Underneath the marketing, the four primitives are stable. Learn them once, read every new framework in one paragraph.

## Concept

### The four primitives

1. **Agent** — a system prompt plus a tool list. Stateless; every run starts from its system prompt and the current message history.
2. **Handoff** — a structured transfer of control from one agent to another. Mechanically, a tool call that returns a new agent or a graph edge that follows a condition.
3. **Shared state** — any data structure that more than one agent can read (sometimes write). Message pool, blackboard, key-value store, vector memory.
4. **Orchestrator** — whoever decides who speaks next. Options: an explicit graph (deterministic), an LLM speaker-selector (soft), the last speaker's handoff call (OpenAI Swarm), or a scheduler over a queue (swarm architecture).

### How every 2026 framework maps to it

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | the LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | the graph, deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

### Why this matters

Once you see the primitives, framework comparison becomes a short checklist:
- Does the orchestrator trust the LLM to route (Swarm) or does it pin routing in code (LangGraph)?
- Is shared state full-history (GroupChat) or projected (StateGraph reducer)?
- Can agents modify each other's prompts (CrewAI manager) or only hand off (Swarm)?

### The stateless insight

Every primitive except shared state is stateless. Agent is a function of (prompt, tools). Handoff is a function call. Orchestrator is a scheduler. **The only stateful thing in the system is shared state.** That is where all the interesting bugs live: memory poisoning, message ordering, versioning, write contention.

### Anatomy of a single primitive

**Agent** = `(system_prompt, tools, model, optional_name)`. No memory. No state.

**Handoff** = `(from_agent, to_agent, reason, payload)`. Three implementations: function return, graph edge, speaker selection.

**Shared state** = `{ messages: [], artifacts: {}, context: {} }`. Two topologies: **full pool** (every agent sees every message) and **projected** (agents see a role-scoped view).

**Orchestrator** = `({state, last_speaker}) -> next_agent`. Four flavors: static, LLM-selected, handoff-driven, queue-driven.

## Build It

`code/main.py` implements the four primitives in ~150 lines of stdlib Python. No real LLM — each agent is a scripted policy so the focus stays on the coordination structure.

The file exports:
- `Agent` — a dataclass of name, system prompt, tools, policy function.
- `Handoff` — a function that returns a new agent.
- `SharedState` — a thread-safe message pool.
- `Orchestrator` — three variants: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (simulated).

```
python3 code/main.py
```

Expected output: three orchestrator runs, one per pattern. Each prints the final message pool. The handoff-driven run reaches fewer agents if the researcher decides it is done early.

## Exercises

1. Run `code/main.py` three times with different agent policies. Observe how the orchestrator choice changes which agents run.
2. Implement a fourth orchestrator type: a queue-driven one where agents poll shared state for work. What deadlock can happen, and how do you detect it?
3. Take the LangGraph quickstart and rewrite it as the four primitives.
4. Read the OpenAI Swarm cookbook. Identify which of the four primitives Swarm makes most ergonomic, and which one it pushes to the caller.
5. Find one framework in this table that hides shared state entirely. Explain what breaks when agents need to coordinate across handoffs without re-reading history.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "An LLM with tools" | A `(system_prompt, tools, model)` triple. Stateless. |
| Handoff | "Transfer of control" | A structured call that names the next agent and optional payload. |
| Shared state | "Memory" / "context" | The only stateful part of a multi-agent system. |
| Orchestrator | "Coordinator" | Whoever decides who runs next. |
| Primitive | "Abstraction" | One of the four axes every framework parameterizes. |
| Message pool | "Shared chat history" | Full-history shared state. |
| Projected state | "Scoped view" | Role-specific view into shared state. |
| Speaker selection | "Who talks next" | Orchestrator pattern where a function picks the next agent from a group. |

## Further Reading

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [CrewAI introduction](https://docs.crewai.com/en/introduction)
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/04-primitive-model)
