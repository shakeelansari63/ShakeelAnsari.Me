# HTN, Workflow & Orchestration Patterns

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch280): Planning with HTN and Evolutionary Search

> Symbolic planning handles the cases where the plan is provably correct. Evolutionary code search handles the cases where the fitness function is machine-checkable. ChatHTN (2025) and AlphaEvolve (2025) show what each unlocks when paired with an LLM.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## Learning Objectives

- Explain Hierarchical Task Networks: tasks, methods, operators, preconditions, effects.
- Describe ChatHTN's hybrid loop — symbolic search with LLM fallback decomposition.
- Explain AlphaEvolve's evolutionary loop and why it only works with a programmatic evaluator.
- Implement a toy HTN planner plus a toy evolutionary search in stdlib.

## The Problem

ReWOO (Lesson 02), Plan-and-Execute, and ReAct cover most agent planning. Two cases they don't cover well:

1. **Plans with provable correctness.** Scheduling, flight pathing, compliance workflows — the plan must be sound by construction. A fluent LLM plan that sometimes hallucinates a step is unacceptable.
2. **Optimizations with a machine-checkable fitness function.** Matrix multiplication, scheduling heuristics, compiler passes — the goal is not "a correct plan" but "the best plan."

HTN planning and AlphaEvolve solve the two different problems. Both use LLMs as amplifiers, not replacements.

## The Concept

### Hierarchical Task Networks

An HTN is:

- **Tasks** — compound (to be decomposed) and primitive (directly executable).
- **Methods** — ways to decompose a compound task into subtasks, with preconditions.
- **Operators** — primitive actions with preconditions and effects.
- **State** — a set of facts.

Planning: given a goal task and an initial state, find a decomposition into primitive operators whose preconditions are satisfied in sequence.

HTN is older than LLMs and still the reference for provably-correct plans.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) interleaves symbolic HTN with LLM queries:

1. Try to decompose the current compound task with existing methods.
2. If no method applies, ask the LLM: "how would you decompose `task` in state `s`?"
3. Translate the LLM response into candidate subtasks.
4. Validate against the operator schema; reject invalid decompositions.
5. Recurse.

The paper's central claim: every plan produced is provably sound because LLM suggestions only enter as candidate decompositions, never as direct plan edits. The symbolic layer owns correctness; the LLM expands the method library.

Online method learning (OpenReview `gwYEDY9j2x`, 2025 follow-up) adds a learner that generalizes LLM-produced decompositions by regression — cutting LLM query frequency up to 75%.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, June 2025) is a different beast: evolutionary code search orchestrated by a Gemini 2.0 Flash/Pro ensemble.

Loop:

1. Start with a seed program + a programmatic evaluator (returns a fitness score).
2. Ensemble of LLMs proposes mutations.
3. Run mutations through the evaluator.
4. Keep the best; mutate again.

Published wins:

- First improvement over Strassen for 4x4 complex matrix multiplication in 56 years (48 scalar multiplications).
- 0.7% recovered Google compute via a Borg scheduling heuristic.
- 32% FlashAttention speedup on a frontier workload.

The hard constraint: the fitness function must be machine-checkable. Evolutionary search over prose answers does not converge.

### When to use which

| Problem class | Use | Why |
|---------------|-----|-----|
| Scheduling with hard constraints | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, no formal guarantees |
| Code improvement with tests | AlphaEvolve | Tests are the evaluator |
| Policy-bound automation | HTN | Preconditions encode policy |

### Where this pattern goes wrong

- **HTN without operators.** Without precondition/effect schemas the soundness claim collapses. ChatHTN's "LLM suggests decomposition" requires the schema to reject invalid moves.
- **AlphaEvolve without a real evaluator.** "Ask the LLM if the code is better" is not a fitness function. The evaluator must be deterministic and fast.
- **Over-engineering.** Most agent tasks don't need either. Reach for ReAct or ReWOO first.

## Build It

`code/main.py` implements two toys:

- A stdlib HTN planner with operators, methods, preconditions, effects, and a `LLMFallback` that kicks in when no method matches a compound task. The "LLM" is a scripted decomposer so the planner runs offline.
- A stdlib evolutionary search over arithmetic programs: grow expressions whose output minimizes `|f(x) - target|` over a test set. Evaluator is deterministic.

Run it:

```
python3 code/main.py
```

The trace shows the HTN planner decomposing a compound task (with a mid-plan LLM fallback) and the evolutionary loop converging on a target expression.

## Use It

- **HTN planners** — `pyhop`, `SHOP3`, or build your own for domain-specific policy enforcement.
- **ChatHTN** — research code; the pattern (symbolic + LLM fallback) ports cleanly to any HTN planner.
- **AlphaEvolve** — DeepMind paper; the pattern (ensemble + evaluator) is reproducible. OpenEvolve and similar open-source forks are emerging.
- **Agent frameworks** — none ship first-class HTN or AlphaEvolve yet. Build it as a subagent or a background worker.

## Exercises

1. Extend the HTN planner with backtracking: when an operator's postcondition fails at runtime, roll back and try the next method.
2. Add a LLM-method cache to ChatHTN: when the LLM decomposes task `T` in state pattern `P`, store the result. Re-check the method library first on the next call.
3. Swap the evolutionary search evaluator to a real test suite. Evolve a sort function that passes 20 test cases; report generations to convergence.
4. Read AlphaEvolve's evaluator design notes. Design an evaluator for a domain you care about (SQL query optimization, test-suite minimization, deployment YAML).
5. Combine: use HTN to decompose a compound task into subtasks, then use evolutionary search on each subtask's primitive operator. Where does it shine, where does it over-engineer?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| HTN | "Hierarchical planner" | Task decomposition with operators, preconditions, effects |
| Method | "Decomposition rule" | Way to break a compound task into subtasks |
| Operator | "Primitive action" | Concrete step with precondition and effect |
| ChatHTN | "LLM + HTN" | Symbolic planner asks LLM when no method matches |
| AlphaEvolve | "Evolutionary code search" | Ensemble LLMs mutate code; deterministic evaluator selects |
| Fitness function | "Evaluator" | Deterministic, machine-checkable score over outputs |
| Online method learning | "Cached LLM decomposition" | Store + generalize LLM plans to cut query cost |

## Further Reading

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary code search with LLM mutations
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when to reach for a planner vs a simple loop

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/11-planning-htn-and-evolutionary)

---

## Part 2 (ch281): Anthropic's Workflow Patterns: Simple Over Complex

> Schluntz and Zhang (Anthropic, Dec 2024) distinguish workflows (predefined paths) from agents (dynamic tool-use). Five workflow patterns cover most cases. Start with direct API calls. Add agents only when steps cannot be predicted.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- Name Anthropic's five workflow patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- Explain the agent-vs-workflow distinction and the engineering cost of each.
- Identify when to pick a workflow over an agent (and vice versa).
- Implement all five patterns in stdlib against a scripted LLM.

## The Problem

Teams reach for multi-agent frameworks for problems that want a single function call. The cost is real: frameworks add layers that obscure prompts, hide control flow, and invite premature complexity. Schluntz and Zhang's Dec 2024 post is the most-cited industry pushback: start simple, add complexity only when it earns its cost.

## The Concept

### Workflows vs agents

- **Workflow.** LLMs and tools orchestrated through predefined code paths. Engineers own the graph.
- **Agent.** LLMs dynamically direct their own tools and take their own steps. The model owns the graph.

Both have their place. Workflows are cheaper, faster, and easier to debug. Agents unlock open-ended problems but make failure modes harder to reason about.

### The augmented LLM

Foundation for all five patterns: one LLM with three capabilities wired in — search (retrieval), tools (actions), memory (persistence). Any API call can use these.

### The five patterns

1. **Prompt chaining.** Output of call 1 is input to call 2. Use when a task has a clean linear decomposition. Optional programmatic gates between steps.

2. **Routing.** A classifier LLM picks which downstream LLM or tool to invoke. Use when categorically different inputs need different handling (tier-1 support vs refund vs bug vs sales).

3. **Parallelization.** Run N LLM calls concurrently, aggregate results. Two shapes: sectioning (different chunks) and voting (same prompt, N runs, majority/synthesis).

4. **Orchestrator-workers.** An orchestrator LLM dynamically decides which workers (also LLMs) to run and synthesizes their output. Similar to agent loops but the orchestrator does not loop indefinitely.

5. **Evaluator-optimizer.** One LLM proposes an answer, another LLM evaluates it. Iterate until the evaluator passes. This is Self-Refine (Lesson 05) generalized.

### Where workflows beat agents

- **Predictable tasks.** If you can enumerate the steps, you should.
- **Cost-bound tasks.** Workflows have bounded step counts; agents can spiral.
- **Compliance-bound tasks.** Auditors want to read the graph, not infer it from trajectories.

### Where agents beat workflows

- **Open-ended research.** When the next step depends on what the last step returned.
- **Variable-length tasks.** Minutes to hours of work where step count is unknown.
- **Novel domains.** When you don't yet know the right workflow — exploration first, codify later.

### The context-engineering companion

"Effective context engineering for AI agents" (Anthropic 2025) formalizes the adjacent discipline: the 200k window is a budget, not a container. What to include, when to compact, when to let context grow. Covered in detail in Phase 14 lesson on context compression.

## Build It

`code/main.py` implements all five workflow patterns against a `ScriptedLLM`:

- `prompt_chain(input, steps)` — sequential.
- `route(input, classifier, handlers)` — classification + dispatch.
- `parallel_vote(prompt, n, aggregator)` — N runs, aggregate.
- `orchestrator_workers(task, workers)` — orchestrator picks workers.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — loop until pass.

Run it:

```
python3 code/main.py
```

Each pattern prints its trace. Total lines of code per pattern is ~10-15; the cost of a framework is measured in thousands.

## Use It

- Direct API calls for most tasks.
- Framework only when the pattern genuinely needs durable state (LangGraph), actor-model concurrency (AutoGen v0.4), or role templating (CrewAI).
- Reach for the Claude Agent SDK when you want the Claude Code harness shape without rebuilding it.

## Exercises

1. Implement routing with a confidence threshold. Below threshold -> escalate to human. Where does the threshold land for a tier-1 support use case?
2. Add a timeout to `parallel_vote`. What happens when one call hangs? How do you aggregate with missing votes?
3. Turn `evaluator_optimizer` into a bandit: keep the top-2 outputs across iterations so a late good result doesn't get overwritten by a late bad one.
4. Combine prompt chaining with routing: a router picks one of three chains. Measure token cost vs a single big-prompt alternative.
5. Pick one of your production features. Draw the workflow graph. Count steps. Would an agent actually be better here?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Workflow | "Predefined flow" | Engineer-owned graph of LLM and tool calls |
| Agent | "Autonomous AI" | Model-owned graph; dynamic tool direction |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory; the atomic unit |
| Prompt chaining | "Sequential calls" | Output of call N is input to call N+1 |
| Routing | "Classifier dispatch" | Pick which chain/model handles the input |
| Parallelization | "Fan out" | N concurrent calls; aggregate by sectioning or voting |
| Orchestrator-workers | "Dispatcher agent" | Orchestrator LLM picks specialist LLMs dynamically |
| Evaluator-optimizer | "Proposer + judge" | Iterate until evaluator passes; Self-Refine generalized |

## Further Reading

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — the five workflow patterns
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — the companion discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — when stateful graphs earn their cost
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — the orchestrator-workers pattern, productized

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/12-anthropic-workflow-patterns)

---

## Part 3 (ch282): LangGraph: Stateful Graphs and Durable Execution

> LangGraph is the 2026 reference for low-level stateful orchestration. Agent is a state machine; nodes are functions; edges are transitions; state is immutable and checkpointed after every step. Resume from any failure exactly where it left off.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- Describe LangGraph's core model: state machine with immutable state, function nodes, conditional edges, and post-step checkpoints.
- Name the four capabilities the docs highlight: durable execution, streaming, human-in-the-loop, comprehensive memory.
- Explain the three orchestration topologies LangGraph supports: supervisor, peer-to-peer (swarm), hierarchical (nested subgraphs).
- Implement a stdlib state graph with immutable state, conditional edges, and a checkpoint/resume cycle.

## The Problem

Agents and workflows share a problem: when a 40-step run fails at step 38, you want to resume from step 38, not start over. Second-class state models leave operators hacking retries around a library that assumes fresh runs.

LangGraph's design answer: state is a first-class typed object, mutations are explicit, and checkpoints persist after every node. Resume is a `load_state(session_id)` call.

## The Concept

### The graph

A graph is defined by:

- **State type.** A typed dict (or Pydantic model) that every node reads and mutates.
- **Nodes.** Pure functions `(state) -> state_update`. Updates are merged into state after return.
- **Edges.** Conditional or direct transitions between nodes.
- **Entry and exit.** `START` and `END` sentinel nodes mark the boundary.

Example: an agent with `classify`, `refund`, `bug`, `sales`, `done` nodes — a routing workflow as a graph.

### Durable execution

After each node returns, the runtime serializes the state and writes it to a checkpointer (SQLite, Postgres, Redis, custom). On failure at step N, the runtime can `resume(session_id)` and pick up from step N+1 with exact state.

The LangGraph docs explicitly highlight production users where this matters: Klarna, Uber, J.P. Morgan. The claim isn't the graph shape; it's that the graph shape plus checkpointing makes recovery cheap.

### Streaming

Every node can yield partial output. The graph streams per-node-delta events to the caller so UIs update as the graph runs.

### Human-in-the-loop

Inspect and modify state between nodes. Implementations: pause before a critical node, surface state to a human, accept modifications, resume. The checkpointer makes this easy because state is already serialized.

### Memory

Short-term (within a run — conversation history in state) and long-term (across runs — persistent via the checkpointer plus a separate long-term store). LangGraph integrates with external memory systems (Mem0, custom) via tools.

### Three topologies

1. **Supervisor.** Central router LLM dispatches to specialist subagents. `create_supervisor()` in `langgraph-supervisor` (though the LangChain team in 2026 recommends doing this through tool calls directly for more context control).
2. **Swarm / peer-to-peer.** Agents hand off directly via a shared tool surface. No central router.
3. **Hierarchical.** Supervisors managing sub-supervisors, implemented as nested subgraphs.

### Where this pattern goes wrong

- **Checkpoints too small.** Only checkpointing conversation turns leaves tool state and memory writes unrecoverable. Full state must serialize.
- **Non-deterministic nodes.** Resume assumes node inputs produce the same state update. Random seeds, wall-clock, external APIs must be captured.
- **Over-use of conditional edges.** A graph with every edge conditional is a state machine that cannot be reasoned about. Prefer linear chains with occasional branches.

## Build It

`code/main.py` implements a stdlib stateful graph:

- `State` — a typed dict with `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — callable taking state and returning an update dict.
- `StateGraph` — nodes + edges + conditional edges + run + resume.
- `SQLiteCheckpointer` (in-memory fake) — serializes state after every node; `load(session_id)` restores.
- A demo graph: classify -> branch(refund / bug / sales) -> human gate -> send.

Run it:

```
python3 code/main.py
```

The trace shows the first run failing at the human gate, persistence, then resume producing the final output.

## Use It

- **LangGraph** — the reference, production-ready. Use `create_react_agent`, `create_supervisor`, or build your own graph.
- **AutoGen v0.4** (Lesson 14) — actor model alternative for high-concurrency scenarios.
- **Claude Agent SDK** (Lesson 17) — managed harness with built-in session store.
- **Custom** — when you need exact control over state shape or checkpointer backend.

## Exercises

1. Add a conditional edge from `classify` to `end` when classification confidence is below a threshold. Resume the run after a human sets `route` manually.
2. Swap the SQLite-like fake for a real SQLite checkpointer. Measure per-step serialization overhead.
3. Implement parallel edges: two nodes run concurrently, merge by a custom reducer. What does immutable state buy here?
4. Read `langgraph-supervisor` reference. Port the toy to `create_supervisor`. Compare the trace shapes.
5. Add streaming: each node yields partial state while it runs. Print the deltas as they arrive.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — the reference docs
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store and subagents

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/13-langgraph-stateful-graphs)

---

## Part 4 (ch297): Orchestration Patterns: Supervisor, Swarm, Hierarchical

> Four orchestration patterns recur across 2026 frameworks: supervisor-worker, swarm / peer-to-peer, hierarchical, debate. Anthropic's guidance: "It's about building the right system for your needs." Start simple; add topology only when a single agent plus five workflow patterns is insufficient.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## Learning Objectives

- Name the four recurring orchestration patterns and when each fits.
- Describe the 2026 LangChain recommendation: tool-call-based supervision vs supervisor libraries.
- Explain Anthropic's "build the right system" rule and how it gates topology choice.
- Implement all four in stdlib against a common scripted LLM.

## The Problem

Teams reach for "multi-agent" before they need it. Four patterns recur across frameworks; once you can name them, you can pick the right one — or skip topology entirely.

## The Concept

### Supervisor-worker

- A central routing LLM dispatches to specialist agents.
- Decides: loop back to self, hand off to specialist, terminate.
- Specialists do not talk to each other; all routing goes through the supervisor.

Frameworks: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**2026 LangChain recommendation:** do supervision through direct tool calls rather than `create_supervisor`. Gives finer context engineering control — you decide exactly what each specialist sees.

### Swarm / peer-to-peer

- Agents hand off directly via a shared tool surface.
- No central router.
- Lower latency than supervisor (fewer hops).
- Harder to reason about (no single point of control).

Frameworks: LangGraph swarm topology, OpenAI Agents SDK handoffs (when all agents can hand off to all others).

### Hierarchical

- Supervisors managing sub-supervisors managing workers.
- Implemented as nested subgraphs in LangGraph; nested crews in CrewAI.
- Scales to large agent populations at the cost of operational complexity.

When you need it: when a single supervisor's context budget cannot hold descriptions of all specialists.

### Debate

- Parallel proposers + iterative cross-critique (Lesson 25).
- Not really orchestration — more verification — but shows up as a topology choice in frameworks.

### CrewAI Crew vs Flow

CrewAI formalizes two deployment modes:

- **Flow** for deterministic event-driven automation (recommended starting point for production).
- **Crew** for autonomous role-based collaboration.

This is orthogonal to the four patterns above but maps to topology: Flow is typically supervisor or hierarchical; Crew is typically supervisor with an LLM router.

### Anthropic's guidance

"Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."

Decision order:

1. Single agent + workflow patterns (Lesson 12) — start here.
2. Supervisor-worker — when you have 2-4 specialists.
3. Swarm — when latency matters more than reasoning clarity.
4. Hierarchical — only when supervisor context budget fails.
5. Debate — when accuracy matters more than cost.

### Where this pattern goes wrong

- **Topology-first thinking.** "We need multi-agent" before identifying what problem multi-agent solves.
- **Bouncing handoffs in swarm.** A -> B -> A -> B. Use hop counters.
- **Fake hierarchy.** Three layers because "enterprise"; two actual teams. Collapse.

## Build It

`code/main.py` implements all four patterns in stdlib against a scripted LLM:

- `Supervisor` — central router.
- `Swarm` — peer-to-peer with direct handoffs.
- `Hierarchical` — supervisors of supervisors.
- `Debate` — parallel proposers + critique.

Each pattern handles the same three-intent task (refund / bug / sales). Trace shapes differ.

Run it:

```
python3 code/main.py
```

Output: per-pattern trace + op count. Supervisor is cleanest; swarm is shortest; hierarchical is deepest; debate is most expensive.

## Use It

- **LangGraph** for supervisor and hierarchical (nested subgraphs).
- **OpenAI Agents SDK** for handoffs-as-tools (supervisor-shaped).
- **CrewAI Flow** for production deterministic.
- **Custom** for debate or when you want exact control.

## Exercises

1. Convert a supervisor-worker to a swarm by removing the router. What breaks? What improves?
2. Add a hop counter to the swarm: refuse after 3 handoffs. Does it catch A->B->A bouncing?
3. Build a two-level hierarchical system for a 12-specialist domain. Where does the context budget fail without nesting?
4. Profile the four patterns on a production-shaped workload. Which wins on which metric (latency, cost, accuracy, debuggability)?
5. Read Anthropic's "Building Effective Agents" post. Map each of your production flows to one of the four. Any that don't map cleanly?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | Central LLM dispatches to specialists; they don't talk to each other |
| Swarm | "Peer-to-peer" | Direct handoffs via shared tools; no central router |
| Hierarchical | "Supervisors of supervisors" | Nested subgraphs for large populations |
| Debate | "Proposer + critique" | Parallel proposers, cross-critique (Lesson 25) |
| Tool-call-based supervision | "Supervisor without a library" | Implement supervisor as direct tool calls for context control |
| Crew | "Autonomous team" | CrewAI's role-based collaboration mode |
| Flow | "Deterministic workflow" | CrewAI's event-driven production mode |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/28-orchestration-patterns)
