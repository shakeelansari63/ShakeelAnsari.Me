# Handoffs and Routines — Stateless Orchestration

> OpenAI's Swarm (October 2024) distilled multi-agent orchestration to two primitives: **routines** (instructions + tools as a system prompt) and **handoffs** (a tool that returns another Agent). No state machine, no branching DSL — the LLM routes by calling the right handoff tool.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Every multi-agent framework wants you to learn its DSL: LangGraph nodes and edges, CrewAI crews and tasks, AutoGen GroupChat and managers. Swarm pushes in the opposite direction: use the tool-calling capability the model already has. Handoffs become tool calls.

## Concept

### Two primitives

**Routine.** A system prompt that defines an agent's role and available tools.

**Handoff.** A tool the agent can call that returns a new Agent object. The Swarm runtime detects the Agent return value and switches the active agent for the next turn.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

### Why it is viral

- **Small API.** Two concepts to learn.
- **Uses what the model already does.** Tool calling is already production-grade across providers.
- **No state-machine burden.** The agents' prompts describe who they hand off to.

### The stateless trade

Swarm is explicitly stateless between runs. Memory, continuity, long-running tasks — all the caller's problem. In production (OpenAI Agents SDK, March 2025) this was one of the main things that changed: the SDK adds built-in session management, guardrails, and tracing while keeping the handoff primitive.

### When Swarm/handoffs fit

- **Triage patterns.** Front-line agent routes user to a specialist.
- **Skill-based handoffs.** "If the task needs code, call the coder."
- **Short, bounded conversations.** Customer support, FAQ-to-ticket.

### When Swarm struggles

- **Long sessions with shared memory.** Handoffs reset state.
- **Parallel execution.** Handoff is one-at-a-time.
- **Audit and replay.** LLM's handoff choice is not deterministic.

### OpenAI Agents SDK (March 2025)

Adds session state, guardrails, tracing, and handoff filters. The handoff primitive survives.

### Swarm vs GroupChat

GroupChat: a selector picks the next speaker from outside. Swarm: the current agent picks its successor by calling a handoff tool.

## Build It

`code/main.py` implements Swarm from scratch: an Agent dataclass, a handoff mechanism (tool returns Agent), and a run loop that detects agent switches.

Demo: a triage agent routes to refund, sales, or support specialists.

```
python3 code/main.py
```

## Ship It

Checklist:
- **Handoff logging.** Every handoff writes a trace event.
- **Context transfer rules.** Decide what moves on handoff.
- **Guardrail on handoff.** Handoff to a specialist with different tool permissions must be authenticated.
- **Loop detection.** Detect two agents handing back and forth.
- **Fallback agent.** If a handoff target does not exist, fall back to a safe default.

## Exercises

1. Run `code/main.py`, triage to the refund agent. Confirm the second turn's active agent is refund.
2. Add a loop-detection rule: if the same two agents have handed off 3 times in a row, force an exit.
3. Read the OpenAI Agents SDK docs on handoff filters. Implement a "summarize-on-handoff" version.
4. Compare the Swarm handoff to a GroupChatManager selector. Which pattern makes prompt injection worse, and why?
5. Read the Swarm cookbook. Identify one explicit design decision Swarm makes that OpenAI Agents SDK changed or kept.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routine | "The agent prompt" | System prompt + tool list. Defines role and available handoffs. |
| Handoff | "Transfer to another agent" | A tool that returns a new Agent. The runtime switches active agent. |
| Stateless | "No memory between runs" | Swarm does not persist anything. |
| Active agent | "Who's speaking now" | The agent currently holding the conversation. |
| Context transfer | "What moves on handoff" | Policy for what history the incoming agent sees. |
| Handoff loop | "Agents ping-pong" | Failure mode where two agents keep handing back to each other. |
| OpenAI Agents SDK | "Production Swarm" | March 2025 successor; adds sessions, guardrails, tracing. |
| Handoff filter | "Gate on transfer" | SDK feature to inspect and modify context at the handoff boundary. |

## Further Reading

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)
- [OpenAI Swarm repo](https://github.com/openai/swarm)
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/11-handoffs-and-routines)
