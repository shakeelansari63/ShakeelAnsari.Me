# CrewAI: Role-Based Crews and Flows

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
