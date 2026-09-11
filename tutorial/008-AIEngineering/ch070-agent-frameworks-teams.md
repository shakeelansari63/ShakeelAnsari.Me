# AutoGen, CrewAI, Agents SDKs & Software Teams

> Combined lessons (6 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch283): AutoGen v0.4: Actor Model and Agent Framework

> AutoGen v0.4 (Microsoft Research, Jan 2025) redesigned agent orchestration around the actor model. Async message exchange, event-driven agents, fault isolation, natural concurrency. The framework is now in maintenance mode while Microsoft Agent Framework (public preview Oct 2025) becomes the successor.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- Describe the actor model: agents as actors, messages as the only IPC, failure isolation per actor.
- Name AutoGen v0.4's three API layers — Core, AgentChat, Extensions — and what each is for.
- Explain why decoupling message delivery from handling gives fault isolation and natural concurrency.
- Implement a stdlib actor runtime in Python and port a two-agent code-review flow onto it.

## The Problem

Most agent frameworks are synchronous: one agent produces, one agent consumes, in a call stack. Failures crash the stack. Concurrency is bolted on. Distribution requires rewriting.

AutoGen v0.4's answer: the actor model. Each agent is an actor with a private inbox. Messages are the only interaction. The runtime decouples delivery from handling. Failures isolate to one actor. Concurrency is native. Distribution is just different transport.

## The Concept

### Actors

An actor has:

- A private state (never directly touched from outside).
- An inbox (message queue).
- A handler: `receive(message) -> effects` where effects can be "reply," "send to other actor," "spawn new actor," "update state," "stop self."

Two actors cannot share memory. They can only send messages.

### Three API layers in AutoGen v0.4

1. **Core.** Low-level actor framework. `AgentRuntime`, `Agent`, `Message`, `Topic`. Async message exchange, event-driven.
2. **AgentChat.** Task-driven high-level API (replacement for v0.2's ConversableAgent). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Integrations — OpenAI, Anthropic, Azure, tools, memory.

### Why decoupling matters

In the v0.2 model, calling `agent_a.chat(agent_b)` synchronously blocks agent_a until agent_b returns. In v0.4, `send(agent_b, msg)` puts the message in agent_b's inbox and returns. The runtime delivers later. Three consequences:

- **Fault isolation.** Agent B crashing does not crash Agent A — the runtime catches the failure in B's handler and decides what to do (log, retry, dead-letter).
- **Natural concurrency.** Many messages in flight at once; actors process their inbox concurrently.
- **Distribution-ready.** Inbox + transport is the same abstraction whether the actor is in-process or on another host.

### Topologies

- **RoundRobinGroupChat.** Agents take turns in a fixed rotation.
- **SelectorGroupChat.** A selector agent picks who goes next based on conversation context.
- **Magentic-One.** Reference multi-agent team for web browsing, code execution, file handling. Built on AgentChat.

### Observability

OpenTelemetry support is built in. Every message emits a span; tool calls carry `gen_ai.*` attributes per the 2026 OTel GenAI semantic conventions (Lesson 23).

### Status: maintenance mode

Early 2026: AutoGen v0.7.x is stable for research and prototyping. Microsoft has shifted active development to the Microsoft Agent Framework (public preview Oct 1 2025; 1.0 GA targeted end of Q1 2026). AutoGen patterns port forward cleanly — the actor model is the durable idea.

## Build It

`code/main.py` implements a stdlib actor runtime:

- `Message` — typed payload with `sender`, `recipient`, `topic`, `body`.
- `Actor` — abstract with `receive(message, runtime)`.
- `Runtime` — event loop with a shared queue, delivery, failure isolation.
- A two-actor demo: `ReviewerAgent` reviews code, `ChecklistAgent` runs a checklist; they exchange messages until consensus.

Run it:

```
python3 code/main.py
```

The trace shows message delivery, a simulated failure in one actor that does not crash the other, and convergence on a shared verdict.

## Use It

- **AutoGen v0.4/v0.7** (maintenance) — stable for research, prototyping, multi-agent patterns.
- **Microsoft Agent Framework** (public preview) — the forward path; same actor-model ideas in a refreshed API.
- **LangGraph swarm topology** (Lesson 13) — similar pattern via shared-tool handoffs.
- **Custom actor runtime** — when you need specific transport (NATS, RabbitMQ, gRPC).

## Exercises

1. Add a dead-letter queue: when a handler raises, park the failing message for human inspection. How often does DLQ get hit in your toy?
2. Implement `SelectorGroupChat`: a selector actor picks who processes the next message based on conversation state.
3. Add distributed transport: swap the in-process queue for a JSON-over-HTTP server so actors can run in separate processes.
4. Wire an OTel span per message (or a no-op stand-in). Emit `gen_ai.agent.name`, `gen_ai.operation.name` per Lesson 23.
5. Read AutoGen v0.4's architecture post. Port your toy to the real `autogen_core` API. What did you skip that matters in production?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Actor | "Agent" | Private state + inbox + handler; no shared memory |
| Message | "Event" | Typed payload; the only way actors interact |
| Inbox | "Mailbox" | Per-actor queue of pending messages |
| Runtime | "Agent host" | Event loop that routes messages and isolates failures |
| Topic | "Channel" | Named publish-subscribe route between actors |
| Fault isolation | "Let it crash" | One actor failing does not crash others |
| RoundRobinGroupChat | "Fixed-rotation team" | Agents take turns in order |
| SelectorGroupChat | "Context-routed team" | Selector picks who goes next |
| Magentic-One | "Reference team" | Multi-agent squad for web + code + files |

## Further Reading

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — the redesign post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graph-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spans AutoGen emits by default

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/14-autogen-actor-model)

---

## Part 2 (ch284): CrewAI: Role-Based Crews and Flows

> CrewAI is the 2026 role-based multi-agent framework. Four primitives: Agent, Task, Crew, Process. Two top-level shapes: Crews (autonomous, role-based collaboration) and Flows (event-driven, deterministic). The docs are blunt: "for any production-ready application, start with a Flow."

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Learning Objectives

- Name CrewAI's four primitives (Agent, Task, Crew, Process) and what each owns.
- Distinguish Sequential, Hierarchical, and the planned Consensus process; pick one per workload.
- Distinguish Crews (autonomous role-based) from Flows (event-driven deterministic), and explain the docs' production recommendation.
- Wire tools with the `@tool` decorator and `BaseTool` subclass; reason about structured outputs vs free text.
- Name the four CrewAI memory types and when each pays off.
- Implement a stdlib three-agent crew (researcher, writer, editor) that produces a brief.
- Spot the three CrewAI failure modes: prompt-bloat, manager-LLM tax, brittle handoffs.

## The Problem

Teams adopting multi-agent frameworks hit the same wall. "Autonomous collaboration" sounds great in a demo. Then a customer files a bug and you need deterministic replay. Or finance asks how much an LLM-routed crew costs per run. Or on-call needs to know which agent stalled at 3 AM.

Free-form LLM-routed crews answer none of those cleanly. Pure DAGs answer them all but lose the exploratory shape a brainstorming agent needs.

CrewAI's split is honest about the trade. Crews for collaborative, role-based, exploratory work. Flows for event-driven, code-owned, auditable production. Same framework, two shapes, pick per surface.

## The Concept

### Four primitives

CrewAI's surface is small. Memorize this and the rest is config.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. The backstory is load-bearing. It shapes tone, judgment, when the agent stops. Tools are functions the agent can call (more below).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. A reusable unit of work. `expected_output` is the contract. `context` lists upstream tasks whose outputs are passed in. `output_pydantic` forces a structured shape.
- **Crew.** Container. Owns the list of `agents`, the list of `tasks`, the `process`, and optional `memory` + `verbose` + `manager_llm` settings.
- **Process.** Execution strategy. Sequential, Hierarchical, Consensus (planned). Picks the shape of the run.

Agents do not see each other directly. Tasks reference agents. The Crew sequences tasks. The Process decides who picks the next task. That is the whole mental model.

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Tasks run in declaration order. Output of task N is available as `context` to task N+1. Lowest cost. Most predictable. Use when the order is fixed.
- **Hierarchical.** A manager Agent (separate LLM call) routes between specialists. CrewAI spawns the manager either from your `manager_llm` config or a default. The manager picks the next task each round and can refuse or re-route. Use when you have four or more specialists and order genuinely depends on prior output.
- **Consensus.** Planned, not currently implemented in the public API. The docs reserve the name for a future voting-based process. Do not rely on it today.

Hierarchical adds a per-round LLM call (the manager) on top of every specialist call. Token cost can triple on a five-step run. Pay for it only when you need the routing.

### Crews vs Flows

This is the framing the docs lead with in 2026.

- **Crew.** LLM-driven autonomy. The framework picks the shape at runtime. Good for: research, brainstorming, first drafts, anywhere the path is part of the answer. Hard to replay. Hard to test. Cheap to prototype.
- **Flow.** Event-driven graph you own. `@start` marks the entry. `@listen(topic)` marks a step that fires when another step emits that topic. Each step is plain Python (can call a Crew internally). Good for: production. Observable. Testable. Deterministic.

The docs' 2026 production recommendation: start with a Flow. Fold Crews in as `Crew.kickoff()` calls from inside Flow steps when autonomy earns its cost. The Flow gives you the audit trail, the Crew gives you the exploration. Compose, do not pick.

### Tool integration

Three ways to give an Agent a tool. Pick the simplest one that fits.

1. **`@tool` decorator.** Pure functions become tools. Signature is the schema; docstring is the description the LLM sees. Best for one-off helpers.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.** Class-based tool with explicit args schema, async support, retries. Use when the tool has state (a client, a cache) or needs structured args.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.** CrewAI ships first-party adapters: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Wired with one import.

Structured outputs use Pydantic. Pass `output_pydantic=MyModel` on the Task. CrewAI validates the LLM response against the model and either coerces or retries. Pair this with a tight `expected_output` string. Free-text outputs are fine for drafts; structured outputs are what downstream Flows can consume.

### Memory hooks

CrewAI ships four memory types out of the box. They compose: a Crew can enable all four at once.

- **Short-term.** Conversation buffer within a single run. Wiped at the end.
- **Long-term.** Persisted across runs. Stored in a vector DB (Chroma by default, swappable). Retrieved by similarity to the current task.
- **Entity.** Per-entity facts. "Customer X is on the enterprise plan." Keyed by entity, not by similarity. Survives across runs.
- **Contextual.** Assembly-time retrieval. Pulls relevant memory at the moment the Agent needs it, not preloaded.

Enable on the Crew with `memory=True` or per-type config. Backed by an embeddings provider you configure (defaults to OpenAI, swappable to local). Memory is one of the places CrewAI earns its keep against thinner frameworks; pure LangGraph requires you to wire each of these yourself.

### When CrewAI fits

- Three to six agents with named roles and a collaborative workflow. Drafting, reviewing, planning, brainstorming.
- Routing where the LLM's judgment about the next step is part of the value (Hierarchical).
- Anywhere the team is happier reading `role + goal + backstory` than reading a graph definition.

### When CrewAI does not fit

- Deterministic DAGs with strict ordering. Use LangGraph (Lesson 13). The graph shape is the right abstraction; CrewAI's role framing is friction.
- Sub-second latency budgets. Hierarchical adds round trips. Even Sequential serializes prompts that include backstories and prior outputs.
- Single-agent loops. Skip the framework; an agent loop (Lesson 1) plus a tool registry is shorter.

### Where this pattern goes wrong

- **Prompt-bloat from backstories.** A 2000-word backstory per agent and a five-agent crew burns the context budget before the first tool call. Keep backstories under 200 words.
- **Manager-LLM token tax.** Hierarchical process adds a manager LLM call before every specialist call. On a five-task crew that is six LLM calls instead of five, and the manager call carries the full task list plus prior outputs. Switch to Sequential unless routing depends on output.
- **Brittle handoffs.** Task N's `expected_output` is "an outline". Task N+1 reads it as `context` and tries to parse three sections. The LLM produced four. Fix with `output_pydantic` on Task N so Task N+1 reads a typed object, not free text.
- **Crew-as-prod.** Free-form Crew shipped to production without a Flow wrapper. Output variability is high; wrap with a Flow.

## Build It

`code/main.py` implements stdlib versions of both shapes plus a three-agent crew. Shape includes `Agent`, `Task` dataclasses, `SequentialCrew`, `HierarchicalCrew`, `Flow` with `@start` and `@listen`, `tool(name)` decorator, and `Memory` with short-term, long-term, entity stores. Demo: researcher, writer, editor crew producing a brief on "agent engineering 2026."

Run it:

```bash
python3 code/main.py
```

Trace covers sequential crew threading outputs through `context`, hierarchical crew with manager picks, flow running the same three steps with explicit topics, tool calls routed through `@tool`, and long-term memory surviving across two kickoffs.

## Use It

- **CrewAI Flow** for production. Even when the Flow is one step that calls `Crew.kickoff()`.
- **CrewAI Crew (Sequential)** for clear-ordering collaborative work.
- **CrewAI Crew (Hierarchical)** when routing depends on output and you have four or more specialists.
- **LangGraph** for explicit state machines, durable resume, strict ordering.

## Exercises

1. Convert the Sequential crew to a Flow. Count the touchpoints where variability drops.
2. Add entity memory to the crew: facts about a customer persist across kickoffs.
3. Implement a Hierarchical process where the manager refuses to route to the editor until the writer's output has at least three paragraphs.
4. Wire a `BaseTool` subclass for a (mocked) web search. Compare the trace shape vs the `@tool` decorator version.
5. Add `output_pydantic=Brief` to the editor task and verify CrewAI's retry behavior on malformed JSON.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process |

## Further Reading

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/15-crewai-role-based-crews)

---

## Part 3 (ch285): OpenAI Agents SDK: Handoffs, Guardrails, Tracing

> OpenAI Agents SDK is the lightweight multi-agent framework built on the Responses API. Five primitives: Agent, Handoff, Guardrail, Session, Tracing. Handoffs are tools named `transfer_to_<agent>`. Guardrails trip on input or output. Tracing is on by default.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Learning Objectives

- Name the five primitives of the OpenAI Agents SDK.
- Explain handoffs: why they are modeled as tools, what name shape the model sees, and how context transfers.
- Distinguish input guardrails, output guardrails, and tool guardrails; explain `run_in_parallel` vs blocking mode.
- Implement a stdlib runtime with handoffs + guardrails + span-style tracing.

## The Problem

Agents that cannot delegate cleanly end up stuffing everything into one prompt. Agents without guardrails ship PII, policy-violating output, or loop forever. OpenAI's SDK codifies the three primitives that make multi-agent work tractable.

## The Concept

### Five primitives

1. **Agent.** LLM + instructions + tools + handoffs.
2. **Handoff.** Delegation to another agent. Represented to the model as a tool named `transfer_to_<agent_name>`.
3. **Guardrail.** Validation on input (first agent only), output (last agent only), or tool invocation (per function tool).
4. **Session.** Automatic conversation history across turns.
5. **Tracing.** Built-in spans for LLM generations, tool calls, handoffs, guardrails.

### Handoffs as tools

The model sees `transfer_to_billing_agent` in its tool list. Calling it signals the runtime to:

1. Copy the conversation context (or collapse it via `nest_handoff_history` beta).
2. Initialize the target agent with its instructions.
3. Continue the run with the target agent.

This is the supervisor pattern (Lesson 13 / Lesson 28) productized.

### Guardrails

Three flavors:

- **Input guardrails.** Run on the first agent's input. Reject unsafe or out-of-scope requests before any LLM call.
- **Output guardrails.** Run on the last agent's output. Catch PII leaks, policy violations, malformed responses.
- **Tool guardrails.** Run per-function-tool. Validate arguments, check permissions, audit execution.

Mode:

- **Parallel** (default). Guardrail LLM runs alongside the main LLM. Lower tail latency. If tripped, the main LLM's work is discarded (token waste).
- **Blocking** (`run_in_parallel=False`). Guardrail LLM runs first. If tripped, no tokens wasted on the main call.

Tripwires raise `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Tracing

On by default. Every LLM generation, tool call, handoff, and guardrail emits a span. `OPENAI_AGENTS_DISABLE_TRACING=1` opts out. `add_trace_processor(processor)` fans spans to your own backend alongside OpenAI's.

### Sessions

`Session` stores conversation history in a backend (SQLite, Redis, custom). `Runner.run(agent, input, session=session)` auto-loads and appends.

### Where this pattern goes wrong

- **Handoff drift.** Agent A hands off to Agent B which hands back to Agent A. Add a hop counter.
- **Guardrail bypass.** Tool guardrails only fire on function tools; built-in tools (file reader, web fetch) need separate policy.
- **Over-tracing.** Sensitive content in spans. Pair with OTel GenAI content-capture rules (Lesson 23) — store externally, reference by ID.

## Build It

`code/main.py` implements the SDK shape in stdlib:

- `Agent`, `FunctionTool`, `Handoff` (as a function tool with transfer semantics).
- `Runner` with input/output/tool guardrails, handoff dispatch, and hop counter.
- A simple span emitter to show the trace shape.
- A triage agent that hands off to billing or support based on the user's query; guardrail trips on one input.

Run it:

```
python3 code/main.py
```

The trace shows two successful handoffs, one input guardrail trip, and a span tree mirroring what the real SDK emits.

## Use It

- **OpenAI Agents SDK** for OpenAI-first products.
- **Claude Agent SDK** (Lesson 17) for Claude-first products.
- **LangGraph** (Lesson 13) when you want explicit state and durable resume.
- **Custom** when you need exact control (voice, multi-provider, federated deployments).

## Exercises

1. Add a handoff hop counter: refuse after N transfers. Trace the behavior.
2. Implement `nest_handoff_history` as an option — collapse prior messages into one summary before transferring.
3. Write a blocking output guardrail. Compare latency on prompts that would trip it vs ones that pass.
4. Wire `add_trace_processor` to a JSON logger. What shape does it emit per span?
5. Read the SDK docs. Port your stdlib toy to `openai-agents-python`. What did you model wrong?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | Agent type in the SDK; owns tools and handoffs |
| Handoff | "Transfer" | Tool the model calls to delegate to another agent |
| Guardrail | "Policy check" | Validation on input / output / tool invocation |
| Tripwire | "Guardrail trip" | Exception raised when guardrail rejects |
| Session | "History store" | Conversation memory persisted between runs |
| Tracing | "Spans" | Built-in observability over LLM + tool + handoff + guardrail |
| Blocking guardrail | "Sequential check" | Guardrail runs first; no token waste on trip |
| Parallel guardrail | "Concurrent check" | Guardrail runs alongside; lower latency, wastes tokens on trip |

## Further Reading

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — primitives, handoffs, guardrails, tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude-flavored counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when to reach for handoffs at all
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the standard Agents SDK spans map to

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/16-openai-agents-sdk)

---

## Part 4 (ch286): Claude Agent SDK: Subagents and Session Store

> The Claude Agent SDK is the library form of the Claude Code harness. Built-in tools, subagents for context isolation, hooks, W3C trace propagation, session store parity. Claude Managed Agents is the hosted alternative for long-running async work.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## Learning Objectives

- Explain the difference between the Anthropic Client SDK (raw API) and the Claude Agent SDK (harness shape).
- Describe subagents — parallelization and context isolation — and when to reach for them.
- Name the Python SDK's session store surface (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) and the role of `--session-mirror`.
- Implement a stdlib harness with built-in tools, subagent spawning with isolated context, lifecycle hooks, and a session store.

## The Problem

A raw LLM API gets you one round-trip. A production agent needs tool execution, MCP servers, lifecycle hooks, subagent spawning, session persistence, trace propagation. Claude Agent SDK ships this shape as a library — the same harness Claude Code uses, exposed for custom agents.

## The Concept

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** Raw Messages API. You own the loop, the tools, the state.
- **Agent SDK (`claude-agent-sdk`).** Built-in tool execution, MCP connections, hooks, subagent spawning, session store. The Claude Code loop as a library.

### Built-in tools

The SDK ships 10+ tools out of the box: file read/write, shell, grep, glob, web fetch, more. Custom tools register via the standard tool-schema interface.

### Subagents

Two purposes documented by Anthropic:

1. **Parallelization.** Run independent work concurrently. "Find the test file for each of these 20 modules" is 20 parallel subagent tasks.
2. **Context isolation.** Subagents use their own context window; only results return to the orchestrator. The orchestrator's budget is preserved.

Python SDK recent additions: `list_subagents()`, `get_subagent_messages()` for reading subagent transcripts.

### Session store

Protocol parity with TypeScript:

- `append(session_id, message)` — add a turn.
- `load(session_id)` — restore conversation.
- `list_sessions()` — enumerate.
- `delete(session_id)` — with cascade to subagent sessions.
- `list_subkeys(session_id)` — list subagent keys.

`--session-mirror` (CLI flag) mirrors the transcript to an external file as it streams, for debugging.

### Hooks

Lifecycle hooks you can register:

- `PreToolUse`, `PostToolUse` — gate or audit tool calls.
- `SessionStart`, `SessionEnd` — set up and tear down.
- `UserPromptSubmit` — act on user input before the model sees it.
- `PreCompact` — run before context compaction.
- `Stop` — cleanup on agent exit.
- `Notification` — side-channel alerts.

Hooks are how pro-workflow and similar systems add cross-cutting behavior.

### W3C trace context

OTel spans active on the caller propagate into the CLI subprocess via W3C trace context headers. The whole multi-process trace shows up as one trace in your backend.

### Claude Managed Agents

The hosted alternative (beta header `managed-agents-2026-04-01`). Long-running async work, built-in prompt caching, built-in compaction. Trade control for managed infrastructure.

### Where this pattern goes wrong

- **Subagent over-spawn.** Spawning 100 subagents for 100 tiny tasks. Overhead dominates. Batch instead.
- **Hook creep.** Every team adds hooks; startup time balloons. Review hooks quarterly.
- **Session bloat.** Sessions accumulate; size grows. Use `list_sessions` + expiry policy.

## Build It

`code/main.py` implements the SDK shape in stdlib:

- `Tool`, `ToolRegistry` with built-in `read_file`, `write_file`, `list_dir`.
- `Subagent` — private context, isolated run, results returned.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- A demo: main agent spawns 3 subagents in parallel (each isolated), aggregates results, persists session.

Run it:

```
python3 code/main.py
```

The trace shows subagent context isolation (orchestrator context size stays bounded), hook execution, and session persistence.

## Use It

- **Claude Agent SDK** for Claude-first products that want the Claude Code harness shape.
- **Claude Managed Agents** for hosted long-running async work.
- **OpenAI Agents SDK** (Lesson 16) for OpenAI-first counterparts.
- **LangGraph + custom tools** if you want the graph-shaped state machine instead.

## Exercises

1. Add a subagent spawner that batches 20 tasks into groups of 5 parallel subagents. Measure orchestrator context size vs one-per-task.
2. Implement a `PreToolUse` hook that rate-limits `write_file` calls (5 per minute per session). Trace the behavior.
3. Wire `list_subkeys` to render a subagent tree. What does deep nesting look like?
4. Port the toy to the real `claude-agent-sdk` Python package. What changes about tool registration?
5. Read the Claude Managed Agents docs. When would you switch from self-hosted to managed?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | Harness shape: tools, MCP, hooks, subagents, session store |
| Subagent | "Child agent" | Separate context, own budget; results bubble up |
| Session store | "Conversation DB" | Persist, load, list, delete turns with subagent cascade |
| Hook | "Lifecycle callback" | Pre/post tool, session, prompt submit, compact, stop |
| W3C trace context | "Cross-process trace" | Parent span propagates into CLI subprocess |
| Managed Agents | "Hosted harness" | Anthropic-hosted long-running async work |
| `--session-mirror` | "Transcript mirror" | Writes session turns to an external file as they stream |
| MCP server | "Tool surface" | External tool/resource source attached to the agent |

## Further Reading

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — the library form of Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — production patterns
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted alternative
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — counterpart

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/17-claude-agent-sdk)

---

## Part 5 (ch287): Agno and Mastra: Production Runtimes

> Agno (Python) and Mastra (TypeScript) are the 2026 production-runtime pairing. Agno aims at microsecond agent instantiation and stateless FastAPI backends. Mastra ships agents, tools, workflows, unified model routing, and composite storage on the Vercel AI SDK substrate.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Learning Objectives

- Identify Agno's performance targets and when they matter.
- Name Mastra's three primitives — Agents, Tools, Workflows — and the supported server adapters.
- Explain why a stateless session-scoped FastAPI backend is the recommended Agno production path.
- Pick Agno vs Mastra for a given stack (Python-first vs TypeScript-first).

## The Problem

LangGraph, AutoGen, CrewAI are framework-heavy. Teams that want "just the agent loop, fast, in my runtime" reach for Agno (Python) or Mastra (TypeScript). Both trade some of the framework-owned primitives for raw speed and a tighter fit to the surrounding stack.

## The Concept

### Agno

- Python runtime, formerly Phi-data.
- "No graphs, chains, or convoluted patterns — just pure python."
- Performance targets from their docs: ~2μs agent instantiation, ~3.75 KiB memory per agent, ~23 model providers.
- Production path: stateless session-scoped FastAPI backend. Each request starts a fresh agent; session state lives in a DB.
- Native multimodal (text, image, audio, video, file) and agentic RAG.

The speed targets matter when you have thousands of short-lived agents per second (chat fan-in, evaluation pipelines). They matter less when one agent runs for 10 minutes.

### Mastra

- TypeScript, built on Vercel AI SDK.
- Three primitives: **Agents**, **Tools** (Zod-typed), **Workflows**.
- Unified Model Router — 3,300+ models across 94 providers (March 2026).
- Composite storage: memory, workflows, observability to different backends; ClickHouse recommended for observability at scale.
- Apache 2.0 with `ee/` directories under source-available enterprise license.
- Server adapters for Express, Hono, Fastify, Koa; first-class Next.js and Astro integration.
- Ships Mastra Studio (localhost:4111) for debugging.
- 22k+ GitHub stars, 300k+ weekly npm downloads at 1.0 (Jan 2026).

### Positioning

Neither is trying to be LangGraph. They compete on:

- **Language fit.** Agno for Python-first teams; Mastra for TypeScript-first.
- **Runtime ergonomics.** Agno = near-zero overhead; Mastra = integrated with the Vercel ecosystem.
- **Observability.** Both integrate with Langfuse/Phoenix/Opik (Lesson 24) but Mastra Studio is first-party.

### When to pick each

- **Agno** — Python backend, many short-lived agents, strong perf requirements, FastAPI shop.
- **Mastra** — TypeScript backend, Next.js / Vercel deploy, unified multi-provider model routing, Zod-typed tools.
- **LangGraph** (Lesson 13) — when durable state and explicit graph reasoning matter more than raw speed.
- **OpenAI / Claude Agent SDK** — when you want the provider's productized shape (Lessons 16–17).

### Where this pattern goes wrong

- **Perf-for-perf's-sake.** Picking Agno because "2μs" sounds good when the workload is one slow agent call per request. Overhead is not the bottleneck.
- **Ecosystem lock-in.** Mastra's Vercel-flavored integration is a plus on Vercel, a minus elsewhere.
- **Enterprise license confusion.** Mastra's `ee/` directories are source-available, not Apache 2.0. Read the licenses if you're planning to fork.

## Build It

This lesson is primarily comparative — no single code artifact would do both frameworks justice. See `code/main.py` for a side-by-side toy: a minimal "run an agent, stream the output, persist session" flow implemented twice (once Agno-shaped, once Mastra-shaped).

Run it:

```
python3 code/main.py
```

Two structurally different but functionally equivalent traces.

## Use It

- **Agno** — Python backend that needs speed and FastAPI shape.
- **Mastra** — TypeScript backend with many providers and workflow primitives.
- Both ship first-party observability hooks. Both integrate with Langfuse.

## Exercises

1. Read Agno's docs. Port the stdlib ReAct loop (Lesson 01) to Agno. What disappeared? What stayed?
2. Read Mastra's docs. Port the same loop to Mastra. What changed in tool typing (Zod vs nothing)?
3. Benchmark: measure agent instantiation latency on your stack. Does Agno's 2μs matter to your workload?
4. Design a migration: if you've been running CrewAI in Python, what breaks if you move to Agno?
5. Read Mastra's `ee/` license terms. What restrictions would affect an open-source fork?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Further Reading

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — performance targets, FastAPI integration
- [Mastra docs](https://mastra.ai/docs) — primitives, server adapters, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — the stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) — observability comparisons cited by Mastra integrations

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/18-agno-and-mastra-runtimes)

---

## Part 6 (ch426): Multi-Agent Software Engineering Team

> SWE-AF's factory architecture, MetaGPT's role-based prompting, AutoGen 0.4's typed actor graph, Cognition's Devin, and Factory's Droids all converged on the same 2026 shape: an architect plans, N coders work in parallel worktrees, a reviewer gates, a tester verifies. Parallel worktrees convert wall-clock into throughput. Shared state and handoff protocols become the failure surface. The capstone is to build the team, evaluate on SWE-bench Pro, and report which handoffs break and how often.

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Time:** 40 hours

## Problem

Single-agent coding harnesses hit a ceiling on large tasks. Not because any individual agent is weak, but because a 200k-token context cannot hold an architecture plan plus four parallel codebase slices plus reviewer commentary plus test output. Multi-agent factories split the problem: an architect owns the plan, coders own implementation in parallel worktrees, a reviewer gates, a tester verifies. SWE-AF's "factory" architecture, MetaGPT's roles, AutoGen's typed actor graph — all three framings describe the same shape.

The failure surface is the handoff. Architect plans something the coders cannot implement. Coders produce conflicting diffs. Reviewer approves a hallucinated fix. Tester races a still-writing coder. You will build one of these teams, run it on 50 SWE-bench Pro issues, track every handoff, and publish the post-mortem.

## Concept

Roles are typed agents. **Architect** (Claude Opus 4.7) reads the issue, writes a plan, and breaks it into subtasks with explicit interfaces. **Coders** (Claude Sonnet 4.7, N parallel instances, each in a `git worktree` + Daytona sandbox) implement subtasks independently. **Reviewer** (GPT-5.4) reads the merged diff and either approves or requests specific changes. **Tester** (Gemini 2.5 Pro) runs the test suite in isolation and reports pass/fail with artifacts.

Communication is through a shared task board (file-backed or Redis). Each role consumes tasks it is permitted to handle. Handoffs are A2A-protocol-typed messages. Coordination concerns: merge-conflict resolution (coordinator role or automatic three-way merge), shared-state synchronization (the plan is frozen once coders start; replans are separate events), and reviewer gatekeeping (the reviewer cannot approve its own changes or changes it proposed).

Token amplification is the hidden cost. Every role boundary adds summary prompts and handoff context. A 40-turn single-agent run becomes 160 total turns across four roles. The rubric specifically weighs token efficiency vs single-agent baseline because the question is not "does multi-agent work" but "does it win per dollar."

## Architecture

```mermaid
graph TD
    issue[GitHub issue URL] --> architect[Architect (Opus 4.7)]
    architect --> board[Task board]
    board --> coderA[Coder A / Sonnet / worktree A]
    board --> coderB[Coder B / Sonnet / worktree B]
    board --> coderC[Coder C / Sonnet / worktree C]
    board --> coderD[Coder D / Sonnet / worktree D]
    coderA --> merge[Merge coordinator]
    coderB --> merge
    coderC --> merge
    coderD --> merge
    merge --> reviewer[Reviewer (GPT-5.4)]
    reviewer --> tester[Tester (Gemini 2.5 Pro)]
    tester --> pr[open PR]
    tester --> loop[route back to coder]
```

## Stack

- Orchestration: LangGraph with shared state + per-agent sub-graphs
- Messaging: A2A protocol (Google 2025) for typed inter-agent messages
- Models: Opus 4.7 (architect), Sonnet 4.7 (coders), GPT-5.4 (reviewer), Gemini 2.5 Pro (tester)
- Worktree isolation: `git worktree add` per coder + Daytona sandbox
- Merge coordinator: custom three-way merge + LLM-mediated conflict resolution
- Eval: SWE-bench Pro (50 issues), SWE-AF scenarios, HumanEval++ for unit tests
- Observability: Langfuse with role-tagged spans, per-agent token accounting
- Deployment: K8s with each role as a separate Deployment + HPA on backlog

## Build It

1. **Task board.** File-backed JSONL with typed messages: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agents subscribe to tags.

2. **Architect.** Reads the GitHub issue, runs Opus 4.7 with a plan template requiring explicit subtask interfaces (files touched, public functions, test impact). Emits one `plan_request` with a DAG of subtasks.

3. **Coders.** N parallel workers, each claims one subtask from the board. Each spawns a fresh `git worktree add` branch plus a Daytona sandbox. Implements the subtask. Emits `diff_ready` with the patch + test deltas.

4. **Merge coordinator.** On all-coders-done, three-way merges the N branches into a staging branch. LLM-mediated conflict resolution only when file-level overlap exists.

5. **Reviewer.** GPT-5.4 reads the merged diff. Cannot approve diffs it authored. Emits `approved` (no-op) or `review_feedback` with specific change requests routed back to the relevant coder.

6. **Tester.** Gemini 2.5 Pro runs the test suite in a clean sandbox. Captures artifacts. Emits `test_passed` or `test_failed` with stacktraces. Failed tests loop back to the coder owning the failing subtask.

7. **Handoff accounting.** Every message crossing a role boundary gets a span in Langfuse with payload size and model used. Compute per-subtask token amplification (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens).

8. **Eval.** Run on 50 SWE-bench Pro issues. Compare pass@1 and $-per-solved-issue against a single-agent baseline (one Sonnet 4.7 in a single worktree).

9. **Post-mortem.** For each failed issue, identify the handoff that broke (plan too vague, merge conflict, reviewer false-approve, tester flake). Produce a handoff-failure histogram.

## Use It

```console
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Ship It

`outputs/skill-multi-agent-team.md` is the deliverable. Given an issue URL and parallelism level, the team produces a merge-ready PR with per-role token accounting.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Matched 50-issue subset, pass@1 |
| 20 | Parallel speedup | Wall-clock vs single-agent baseline |
| 20 | Review quality | False-approval rate on injected-bug probe |
| 20 | Token efficiency | Total tokens per solved issue vs single-agent |
| 15 | Coordination engineering | Merge-conflict resolution, handoff-failure histogram |
| **100** | | |

## Exercises

1. Inject an obvious bug into a diff mid-run (extra `return None` before the main body). Measure the reviewer's false-approve rate. Tune the reviewer prompt until false-approval is under 5%.

2. Reduce to two coders (architect + coder + reviewer + tester, coder runs two subtasks sequentially). Compare wall-clock and pass rate.

3. Replace the merge coordinator with a single-writer constraint (subtasks touch disjoint file sets). Measure the planning burden on the architect.

4. Swap reviewer from GPT-5.4 to Claude Opus 4.7. Measure false-approval rate and token cost delta.

5. Add a fifth role: documenter (Haiku 4.5). After review, it produces a changelog entry. Measure whether documentation quality justifies the extra token spend.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Parallel worktree | "Isolated branch" | `git worktree add` producing a fresh working tree per coder |
| Task board | "Shared message bus" | File or Redis store of typed messages agents subscribe to |
| Handoff | "Role boundary" | Any message crossing from one role's context to another's |
| Token amplification | "Multi-agent overhead" | Total tokens across roles / single-agent tokens for the same task |
| A2A protocol | "Agent-to-agent" | Google's 2025 spec for typed inter-agent messages |
| Merge coordinator | "Integrator" | Component that runs three-way merge and mediates conflicts |
| False approval | "Reviewer hallucination" | Reviewer approves a diff with known bugs |

## Further Reading

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF)
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT)
- [AutoGen v0.4](https://github.com/microsoft/autogen)
- [Cognition AI (Devin)](https://cognition.ai)
- [Factory Droids](https://www.factory.ai)
- [Google A2A protocol](https://developers.google.com/agent-to-agent)
- [git worktree documentation](https://git-scm.com/docs/git-worktree)
- [SWE-bench Pro](https://www.swebench.com)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/10-multi-agent-software-team)
