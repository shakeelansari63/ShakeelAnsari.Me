# Memory Systems, Mem0 & Skill Libraries

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch275): Tool Use and Function Calling

> Toolformer (Schick et al., 2023) started self-supervised tool annotation. Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) sets the 2026 bar: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination. Single-turn is solved. Memory, dynamic decision-making, and long-horizon tool chains are not.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Learning Objectives

- Explain Toolformer's self-supervised training signal: keep tool annotations only when execution reduces next-token loss.
- Name BFCL V4's five evaluation categories and what each measures.
- Implement a stdlib tool registry with schema validation, argument coercion, and execution sandboxing.
- Diagnose the three 2026 open problems: long-horizon tool chaining, dynamic decision-making, and memory.

## The Problem

Early tool use asked: can the model predict a correct function call? Modern tool use asks: can the model chain tools across 40 steps, with memory, with partial observability, with recovery from tool failures, without hallucinating tools that do not exist?

Toolformer established the baseline: models can learn when to call tools with self-supervision. BFCL V4 defines the 2026 evaluation target. The gap between them is the space production agents live in.

## The Concept

### Toolformer (Schick et al., NeurIPS 2023)

Idea: let the model annotate its own pretraining corpus with candidate API calls. For each candidate, execute it. Keep the annotation only if including the tool result reduces loss on the next token. Fine-tune on the filtered corpus.

Tools covered: calculator, QA system, search engines, translator, calendar. The self-supervision signal is purely about whether the tool helps predict text — no human labels.

Scale result: tool use emerges at scale. Smaller models hurt from tool annotations; larger models gain. This is why 2026 frontier models have strong tool use baked in while most 7B models need explicit tool-use fine-tuning to be reliable.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL is the 2026 de facto evaluation. V4 composition:

- **Agentic (40%)** — full agent trajectories: memory, multi-turn, dynamic decisions.
- **Multi-Turn (30%)** — interactive conversations with tool chains.
- **Live (10%)** — user-submitted real prompts (harder distribution).
- **Non-Live (10%)** — synthetic test cases.
- **Hallucination (10%)** — detect when no tool should be called.

V3 introduced state-based evaluation: after a tool sequence, check the API's actual state (e.g. "is the file created?") rather than match the AST of the tool calls. V4 added web search, memory, and format sensitivity categories.

Key 2026 finding: single-turn function calling is near-solved. Failures concentrate in memory (carrying context across turns), dynamic decision-making (choosing tools based on prior results), long-horizon chains (drift after 20+ steps), and hallucination detection (refusing to call when no tool fits).

### Tool schema

Every provider has a schema. They differ in details but share the same shape:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic uses `input_schema` directly. OpenAI uses `function.parameters`. Both accept JSON Schema. Descriptions are load-bearing — the model reads them to pick the right tool. Bad tool descriptions are the #1 root cause of wrong-tool-picked failures.

### Argument validation

Trust no tool call. Validate:

1. **Type coercion.** Model may return a string "5" where the schema says int. Coerce if unambiguous; reject if not.
2. **Enum validation.** If the schema says `status in {"open", "closed"}` and model emits `"in_progress"`, reject with a descriptive error.
3. **Required fields.** Missing required field -> immediate error observation back to the model, not a crash.
4. **Format validation.** Dates, emails, URLs — validate with concrete parsers, not regex.

Every validation failure should return a structured observation so the model can retry with the correct shape.

### Parallel tool calls

Modern providers support parallel tool calls in one assistant turn. The loop:

1. Model emits 3 tool calls with distinct `tool_use_id`s.
2. Runtime executes them (in parallel if independent).
3. Each result goes back as a `tool_result` block correlated by `tool_use_id`.

Engineering rule: treat correlation IDs as load-bearing. Swap them and you get wrong-tool-to-wrong-result routing.

### Sandboxing

Tool execution is the sandbox boundary. See Lesson 09 for detail. Short version: every tool should specify read/write surface, network access, timeout, memory cap. Generic `run_shell(cmd)` is a red flag; specific `git_status()` is safer.

## Build It

`code/main.py` implements a production-shape tool registry:

- JSON Schema subset validator (stdlib only).
- Tool registration with description, input schema, timeout, and executor.
- Argument coercion and enum validation.
- Parallel tool dispatch with correlation IDs.
- Error observations as structured strings.

Run it:

```
python3 code/main.py
```

The trace shows a mini agent calling three tools in one turn, with one deliberately malformed call that is rejected with a descriptive error the model can act on.

## Use It

Every provider has its own tool schema — Anthropic, OpenAI, Gemini, Bedrock. Use a translation layer (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter) if you need multi-provider. BFCL is the reference benchmark — run it against your agent before shipping if tool use is central to the product.

## Exercises

1. Add a "no-op" tool that lets the model explicitly refuse to use any other tool. Measure on a BFCL-like hallucination test.
2. Implement argument coercion for int-as-string and float-as-string. Where does coercion start to hide real bugs?
3. Add a per-tool timeout and a circuit breaker (refuse the tool for 60s after 3 consecutive failures). What does this change about how the model recovers?
4. Read BFCL V4 description. Pick one category (e.g. "multi-turn") and run 10 example prompts through your agent. Report pass rate.
5. Port the stdlib validator to Pydantic or Zod. What did Pydantic/Zod catch that the toy missed?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Further Reading

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 eval benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — production tool schema in the Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool type and Guardrails

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/06-tool-use-and-function-calling)

---

## Part 2 (ch276): Memory: Virtual Context and MemGPT

> Context windows are finite. Conversations, documents, and tool traces are not. MemGPT (Packer et al., 2023) frames this as OS virtual memory — main context is RAM, external store is disk, the agent pages between them. This is the pattern every 2026 memory system inherits.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Learning Objectives

- Explain the OS analogy MemGPT builds on: main context = RAM, external context = disk, memory tools = page in/out.
- Implement the two-tier MemGPT pattern in stdlib with a main-context buffer, an external searchable store, and page in/out tools.
- Describe how the agent issues "interrupts" to query or modify external memory and how the result is spliced back into the next prompt.
- Identify the MemGPT design choices that carry into Letta (Lesson 08) and Mem0 (Lesson 09).

## The Problem

Context windows look like they should solve memory. They do not. Three failure modes recur in production:

1. **Overflow.** Multi-turn conversations, long documents, or tool-call-heavy trajectories cross the window. Everything past the cutoff is gone.
2. **Dilution.** Even within the window, stuffing irrelevant context dilutes attention over what matters. Frontier models still degrade on long inputs.
3. **Persistence.** A new session starts with an empty window. Agents without external memory cannot say "remember when you asked me to..." across sessions.

Bigger windows help but do not fix this. Mem0's 2025 paper measured that 128k-window baselines still miss long-horizon facts that a 4k-window agent with external memory catches.

## The Concept

### MemGPT: the OS analogy

Packer et al. (arXiv:2310.08560, v2 Feb 2024) map context management to operating-system virtual memory:

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop with memory tools |

The agent runs a normal ReAct loop. One extra class of tools lets it page data in and out of main context.

### Two tiers

- **Main context.** Fixed-size prompt holding the current task. Always visible to the model.
- **External context.** Unbounded, searchable via tools. Read when relevant, written when facts emerge.

The original paper evaluated the design on two tasks beyond the base window: document analysis longer than 100k tokens and multi-session chat with persistent memory across days.

### The interrupt pattern

MemGPT introduces memory-as-interrupt: mid-conversation the agent can invoke a memory tool, the runtime executes it, and the result splices into the next assistant turn as a new observation. Conceptually identical to a Unix `read()` syscall that blocks the process, returns bytes, and the process continues.

Canonical memory tool surface:

- `core_memory_append(section, text)` — write to a persistent section of the prompt.
- `core_memory_replace(section, old, new)` — edit a persistent section.
- `archival_memory_insert(text)` — write to the searchable external store.
- `archival_memory_search(query, top_k)` — retrieve from the external store.
- `conversation_search(query)` — scan past turns.

### Where MemGPT ends and Letta begins

In September 2024 MemGPT became Letta. The research repo (`cpacker/MemGPT`) remains; Letta extends the design:

- Three tiers instead of two (core, recall, archival — Lesson 08).
- Native reasoning replacing the `send_message`/heartbeat pattern (Lesson 08).
- Sleep-time agents running async memory work (Lesson 08).

The MemGPT paper is the 2026 foundation even if production systems run Letta, Mem0, or a custom two-tier store.

### Where this pattern goes wrong

- **Memory rot.** Writes accumulate faster than reads; retrieval drowns in stale facts. Fix: periodic consolidation (Letta sleep-time), explicit invalidation (Mem0 conflict detector).
- **Memory poisoning.** External memory is retrieved text. If attacker-controlled content lands in a memory note, the agent re-ingests it next session. This is the Greshake et al. (Lesson 27) attack restated over time.
- **Citation loss.** Agent recalls "the user asked me to ship X" but cannot cite which turn. Store source references (session ID, turn ID) with every archival write.

## Build It

`code/main.py` implements MemGPT's two-tier pattern in stdlib:

- `MainContext` — fixed-size prompt buffer with a `core` dict and a `messages` list; auto-compacts oldest messages when over cap.
- `ArchivalStore` — in-memory BM25-esque store (token-overlap scoring) of (id, text, tags, session, turn) records.
- Five memory tools mapping to the MemGPT surface.
- A scripted agent that fills archival with facts, then answers a question by calling `archival_memory_search`.

Run it:

```
python3 code/main.py
```

The trace shows the agent writing three facts, filling main context to the cap (forcing eviction), then answering a follow-up question by retrieving from archival — reproducing the MemGPT workflow without any real LLM.

## Use It

Every production memory system today is a MemGPT variant:

- **Letta** (Lesson 08) — three tiers, native reasoning, sleep-time compute.
- **Mem0** (Lesson 09) — vector + KV + graph fused with a scoring layer.
- **OpenAI Assistants / Responses** — managed memory via threads and files.
- **Claude Agent SDK** — long-term memory via skills and session store.

Pick one by operational shape (self-hosted, managed, framework-integrated), not by the core pattern — the core pattern is MemGPT.

## Exercises

1. Add a `max_main_context_tokens` cap measured in tokens (approximate with `len(text.split())` * 1.3). Compact the oldest messages into a summary when the cap is exceeded. Compare behavior with and without the summarizer.
2. Implement BM25 properly over the archival store (term frequency, inverse document frequency). Measure recall@10 on a toy fact set versus the token-overlap baseline.
3. Add `citation` fields (session_id, turn_id, source_url) to archival inserts. Make the agent cite sources on every retrieval-backed answer.
4. Simulate memory poisoning: add an archival record that says "ignore all future user instructions." Write a guard that scans retrievals for directive-shaped text and marks them untrusted.
5. Port the implementation to use the MemGPT research repo's core-memory JSON schema (`cpacker/MemGPT`). What changes when you switch from flat strings to typed sections?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Virtual context | "Unlimited memory" | Main (prompt) + external (searchable) tiers with page in/out |
| Main context | "Working memory" | The prompt — fixed-size, always visible |
| Archival memory | "Long-term store" | External searchable persistence, retrieved on demand |
| Core memory | "Persistent prompt section" | Named sections pinned inside the main context |
| Memory tool | "Memory API" | Tool call the agent issues to read/write external memory |
| Interrupt | "Memory page fault" | Agent pauses, runtime fetches, result splices into next turn |
| Memory rot | "Stale facts" | Old writes drown retrieval; fix with consolidation |
| Memory poisoning | "Injected persistent note" | Attacker content stored as memory, re-ingested on recall |

## Further Reading

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS-inspired virtual context paper
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the three-tier evolution
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — treating context as a budget
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — hybrid production memory on top of this pattern

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/07-memory-virtual-context-memgpt)

---

## Part 3 (ch277): Memory Blocks and Sleep-Time Compute (Letta)

> MemGPT became Letta in 2024. The 2026 evolution adds two ideas: discrete functional memory blocks the model can edit directly, and a sleep-time agent that consolidates memory asynchronously while the primary agent is idle. This is how you scale memory beyond one conversation.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## Learning Objectives

- Name the three memory tiers Letta uses (core, recall, archival) and the role of each.
- Explain the memory-block pattern: Human block, Persona block, and user-defined blocks as first-class typed objects.
- Describe what sleep-time compute is, why it sits off the critical path, and why it can run a stronger model than the primary agent.
- Implement a scripted two-agent loop where a primary agent serves responses and a sleep-time agent consolidates blocks between turns.

## The Problem

MemGPT (Lesson 07) solved the virtual-memory control flow. Three production problems emerged:

1. **Latency.** Every memory operation sits on the critical path. If the agent has to prune, summarize, or reconcile while the user waits, tail latency blows up.
2. **Memory rot.** Writes accumulate. Contradicted facts stay. Retrieval drowns in stale content.
3. **Structure loss.** A flat archival store cannot express "the Human block is always in the prompt; the Persona block is always in the prompt; the Task block swaps per session."

Letta (letta.com) is the 2026 rewrite. Memory blocks make structure explicit; sleep-time compute moves consolidation off the critical path.

## The Concept

### Three tiers

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool call + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core is the MemGPT core. Recall is the conversation buffer with its evicted tail. Archival is the external store. The split cleans up MemGPT's two-tier overloading.

### Memory blocks

A block is a typed, persistent, editable section of the core tier. The original MemGPT paper defined two:

- **Human block** — facts about the user (name, role, preferences, goals).
- **Persona block** — the agent's self-concept (identity, tone, constraints).

Letta generalizes to arbitrary user-defined blocks: a `Task` block for the current goal, a `Project` block for codebase facts, a `Safety` block for hard constraints. Each block has an `id`, `label`, `value`, `limit` (character cap), `description` (so the model knows when to edit it).

Blocks are editable via the tool surface:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — condense a block that is near its limit.

### Sleep-time compute

The 2025 Letta addition: run a second agent in background, off the critical path. Sleep-time agents process conversation transcripts and codebase context, write `learned_context` into shared blocks, and consolidate or invalidate archival records.

Properties that fall out:

- **No latency cost.** Primary responses do not wait for memory ops.
- **Stronger model allowed.** The sleep-time agent can be a more expensive, slower model because it is not latency-constrained.
- **Natural consolidation window.** Dedup, summarize, invalidate contradicted facts when the user is not waiting.

The shape matches how humans work: you do the task, you sleep on it, the long-term memory settles overnight.

### Letta V1 and native reasoning

Letta V1 (`letta_v1_agent`, 2026) deprecates `send_message`/heartbeat and inline `Thought:` tokens in favor of native reasoning. The Responses API (OpenAI) and the Messages API with extended thinking (Anthropic) emit reasoning on a separate channel, passed through turns (encrypted across providers in production). The control loop is still ReAct. The thought trace is structural, not prompt-shaped.

### Where this pattern goes wrong

- **Block bloat.** Infinite `block_append` hits the limit fast. Wire a block summarizer before the write that pushes over the cap.
- **Silent drift.** Sleep-time agent rewrites a block and the primary agent never notices. Version blocks and surface diffs in the trace.
- **Poisoned consolidation.** Sleep-time agent processes attacker-reachable content into core. Lesson 27 applies to the sleep-time surface too.

## Build It

`code/main.py` implements:

- `Block` — id, label, value, limit, description.
- `BlockStore` — CRUD + `near_limit(label)` helper.
- Two scripted agents — `PrimaryAgent` serves a turn, `SleepTimeAgent` consolidates between turns.
- A trace that shows a three-turn conversation with block writes, plus a sleep-time pass that summarizes a block and invalidates a stale fact.

Run it:

```
python3 code/main.py
```

The transcript shows the split: primary turns are fast and produce raw writes; the sleep pass compacts and cleans up.

## Use It

- **Letta** (letta.com) for the reference implementation. Self-host or managed cloud.
- **Claude Agent SDK skills** as block-shaped knowledge — a skill is a named, versioned, retrievable block of instructions the agent loads on demand.
- **Custom builds** for teams that want control over the storage backend. Use the Letta API contract so you can migrate later.

## Exercises

1. Add a `block_summarize` tool that replaces the block value with a model-generated summary when `near_limit` returns true. Which trigger threshold minimizes both summarization calls and block overflow?
2. Implement sleep-time dedup over archival: two records whose text has >90% token overlap collapse to one. Do it only in the sleep pass, never on the critical path.
3. Version blocks. On every write record the old value and a diff. Expose `block_history(label)` so operators can debug "why did the agent forget X."
4. Treat sleep-time agents as untrusted writers. When they touch the Persona or Safety block, require a second-agent review before committing.
5. Port the example to use the Letta API (`letta_v1_agent`). What changes in the block schema, and how does native reasoning alter the trace shape?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory block | "Editable prompt section" | Typed, persistent, LLM-editable segment of core memory |
| Human block | "User memory" | Facts about the user, pinned in core |
| Persona block | "Agent identity" | Self-concept, tone, constraints, pinned in core |
| Sleep-time compute | "Async memory work" | Second agent doing consolidation off the critical path |
| Core / Recall / Archival | "Tiers" | Three-layer memory split: always-visible / conversation / external |
| Block limit | "Cap" | Character limit per block; forces summarization |
| Native reasoning | "Thinking channel" | Provider-level reasoning output, not prompt-level `Thought:` |
| Learned context | "Sleep output" | Facts the sleep-time agent writes into shared blocks |

## Further Reading

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the block pattern
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — async consolidation
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — native reasoning rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — the origin

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/08-memory-blocks-sleep-time-compute)

---

## Part 4 (ch278): Hybrid Memory: Vector + Graph + KV (Mem0)

> Mem0 (Chhikara et al., 2025) treats memory as three stores in parallel — vector for semantic similarity, KV for fast fact lookup, graph for entity-relationship reasoning. A scoring layer fuses the three on retrieval. This is the 2026 production standard for external memory.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Explain why a single store (vector only, graph only, KV only) is insufficient for agent memory.
- Name Mem0's three parallel stores and what each one optimizes for.
- Describe Mem0's fusion scoring — relevance, importance, recency — and why it is a weighted sum, not a hierarchy.
- Implement a toy three-store memory in stdlib with an `add()` that writes to all three and a `search()` that fuses results.

## The Problem

One store is wrong for one of three query classes:

- **Semantic similarity** — "what did we discuss about agent drift last week?" Vector wins; KV and graph miss.
- **Fact lookup** — "what is the user's phone number?" KV wins; vector is wasteful, graph is overkill.
- **Relationship reasoning** — "which customers share the same billing entity?" Graph wins; vector and KV cannot answer.

Production agents issue all three in one session. A single-store memory is always wrong for two of them. Mem0's contribution is wiring all three behind a single `add`/`search` surface with a scoring function that fuses them.

## The Concept

### Three stores in parallel

Mem0 (arXiv:2504.19413, April 2025) on `add(text, user_id, metadata)`:

1. Extract candidate facts from the text (an LLM-driven step).
2. Write each fact to the vector store (embedding) for semantic search.
3. Write each fact to the KV store keyed on (user_id, fact_type, entity) for O(1) lookup.
4. Write each fact to the graph store (Mem0g) as typed edges for relationship queries.

On `search(query, user_id)`:

1. Vector store returns top-k by embedding cosine.
2. KV store returns direct hits keyed on query-derived (user_id, type, entity).
3. Graph store returns subgraph reachable from query entities.
4. A scoring layer fuses the three.

### Fusion scoring

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — vector cosine, KV exact match, graph path weight.
- **Importance** — tagged at write time or learned (some facts matter more: names, IDs, policies).
- **Recency** — exponential decay over time since last write or read.

Weights are tuned per product. Higher `w_recency` for chat agents; higher `w_importance` for compliance agents; higher `w_relevance` for retrieval agents.

### Mem0g and temporal reasoning

Mem0g adds a conflict detector. When a new fact contradicts an existing edge, the existing edge is marked invalid but not deleted. Temporal queries ("what was the user's city in March?") traverse the valid-at-time subgraph.

This is the compliance-grade behavior Letta's invalidation pattern generalizes.

### Benchmark numbers

The Mem0 paper reports (2025):

- **LoCoMo** (long-form conversation memory): 91.6
- **LongMemEval** (long-horizon episodic memory): 93.4
- **BEAM 1M** (1M-token memory benchmark): 64.1

Comparison baselines (full-context 128k LLM, flat vector store, flat KV) all lose by 10+ points. Benchmarks alone don't justify choice — operational shape does — but the numbers show the fusion design is not a rounding error.

### Scope taxonomy

Mem0 splits memory by scope:

- **User memory** — persists across sessions, keyed on `user_id`.
- **Session memory** — persists within one thread.
- **Agent memory** — per-agent instance state.

Every write picks one scope. Retrieval can query across scopes with per-scope weights. Mixing scopes without thought is how you get "the assistant told Alice about Bob's project" incidents.

### Where this pattern goes wrong

- **Embedding drift.** Vector results that look right on the first hundred queries degrade as the corpus grows. Add periodic re-embedding of the top-N-used records.
- **KV schema creep.** `(user_id, type, entity)` looks simple until every team adds their own `type`. Audit the type set quarterly.
- **Graph explosion.** One noisy extractor adds 50 edges per message. Cap graph writes per `add` call; drop low-confidence edges.

## Build It

`code/main.py` implements the three-store pattern in stdlib:

- `VectorStore` — naive token-overlap similarity as an embedding stand-in.
- `KVStore` — dict keyed on `(user_id, fact_type, entity)`.
- `GraphStore` — typed edges (subject, relation, object, valid).
- `Mem0` — top-level facade with `add()`, `search()`, fusion scoring, and scope-aware retrieval.
- A worked trace on a multi-user, multi-session conversation.

Run it:

```
python3 code/main.py
```

The output shows three separate recall paths plus the fused top-k. Flip the scoring weights at the top of `main()` and watch the ranking change.

## Use It

- **Mem0 (Apache 2.0)** — production-ready. Self-host with Postgres + Qdrant + Neo4j, or use the managed cloud.
- **Letta** — three-tier core/recall/archival; bring your own vector and graph backends.
- **Zep** — commercial alternative with temporal KG and fact extraction.
- **Custom builds** — when you need exact control over the extractor (compliance) or fusion weights (voice agents where recency dominates).

## Exercises

1. Replace the toy vector similarity with a real embedding model (sentence-transformers, Ollama, OpenAI embeddings). Measure recall@10 on a synthetic long conversation. Does the ranking drift over 1000 writes?
2. Add a temporal query: `search(query, as_of=timestamp)`. Return only records valid at or before that time. Which store needs the most work?
3. Implement a conflict detector: if an incoming fact contradicts a graph edge, invalidate the old edge and log both. Test on "user lives in Berlin" -> "user lives in Lisbon."
4. Port the fusion scorer to include a `user_feedback` dimension (thumbs-up on retrieved records). How do you prevent gaming (the agent only returns records it already liked)?
5. Read the Mem0 docs (`docs.mem0.ai`). Port the toy to `mem0` client calls. Compare retrieval quality on the same 20 test queries.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hybrid memory | "Vector plus graph plus KV" | Three stores written in parallel, fused on retrieval |
| Fact extraction | "Memory ingestion" | LLM step that breaks text into (entity, relation, fact) tuples |
| Fusion scoring | "Relevance ranking" | Weighted sum of relevance, importance, recency |
| Scope | "Memory namespace" | user / session / agent — determines who sees what |
| Mem0g | "Memory graph" | Typed edges with temporal validity for relationship queries |
| Temporal invalidation | "Soft delete" | Mark contradicted edges invalid; never delete |
| Embedding drift | "Retrieval rot" | Vector quality degrades as corpus grows; re-embed periodically |

## Further Reading

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — the original paper
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — production API, SDKs, managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — the virtual-context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the three-tier sibling design

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/09-hybrid-memory-mem0)

---

## Part 5 (ch279): Skill Libraries and Lifelong Learning (Voyager)

> Voyager (Wang et al., TMLR 2024) treats executable code as a skill. Skills are named, retrievable, composable, and refined by environment feedback. This is the reference architecture for Claude Agent SDK skills, skillkit, and the 2026 skill-library pattern.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Name Voyager's three components — automatic curriculum, skill library, iterative prompting — and the role of each.
- Explain why Voyager makes the action space code, not primitive commands.
- Implement a stdlib skill library with registration, retrieval, composition, and failure-driven refinement.
- Map Voyager's pattern onto the 2026 Claude Agent SDK skills and the skillkit ecosystem.

## The Problem

Agents that rebuild every capability from scratch in every session do three things wrong:

1. **Waste tokens.** Every task re-elicits the same reasoning.
2. **Lose progress.** A correction learned in session A doesn't transfer to session B.
3. **Fail on long-horizon composition.** Complex tasks need capability hierarchies; one-shot prompts cannot express them.

Voyager's answer: treat each reusable capability as a named chunk of code stored in a library, retrievable by similarity, composable with other skills, and refined by execution feedback.

## The Concept

### Three components

Voyager (arXiv:2305.16291) structures an agent around:

1. **Automatic curriculum.** A curiosity-driven proposer picks the next task based on the agent's current skill set and environment state. Exploration is bottom-up.
2. **Skill library.** Each skill is executable code. New skills are added when a task succeeds. Skills are retrieved by query-to-description similarity.
3. **Iterative prompting mechanism.** On failure, the agent receives execution errors, environment feedback, and self-verification output, then refines the skill.

The Minecraft evaluation (Wang et al., 2024): 3.3x more unique items, 8.5x faster stone tools, 6.4x faster iron tools, 2.3x longer map traversal versus baselines. The numbers are Minecraft-specific, but the pattern transfers.

### Action space = code

Most agents emit primitive commands. Voyager emits JavaScript functions. A skill is:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Composed from sub-skills. Stored keyed on description and embedding. Retrieved as a program, not a prompt.

This is the 2026 Claude Agent SDK skill: a named, retrievable chunk of code plus instructions the agent loads on demand.

### Skill retrieval

New task "make a diamond pickaxe." Agent:

1. Embeds the task description.
2. Queries the skill library for top-k similar skills.
3. Retrieves `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` etc.
4. Composes the new skill from retrieved primitives + new logic.

This is the pattern MCP resources (Phase 13) and Agent SDK skills implement: retrieval over a knowledge/code surface, scoped to the current task.

### Iterative refinement

Voyager's feedback loop:

1. Agent writes a skill.
2. Skill runs against the environment.
3. One of three signals returns: `success`, `error` (with stack trace), `self-verification failure`.
4. Agent rewrites the skill using the signal as context.
5. Loop until success or max rounds.

This is Self-Refine (Lesson 05) applied to code generation with environment-grounded verification. CRITIC (Lesson 05) is the same pattern with external tools as the verifier.

### Curriculum and exploration

Voyager's curriculum module proposes tasks like "build a shelter near the lake" based on what the agent has and what it has not yet done. The proposer uses the environment state + skill inventory to pick a task just above current capability — the exploration sweet spot.

For production agents this translates to a "what's missing" operator: given the current skill library and a domain, what skills are we not yet covering? Teams typically implement this manually as curriculum review.

### Where this pattern goes wrong

- **Skill library rot.** Same skill added 10 times with slightly different descriptions. Add deduplication on write; retrieval returns only one.
- **Composed-skill drift.** Parent skill depends on a child that was refined. Version skills; a parent pinned to v1 doesn't magically pick up v3.
- **Retrieval quality.** Vector retrieval over skill descriptions degrades as the library grows past a few hundred. Supplement with tag filters and hard constraints ("only skills with `category=tooling`").

## Build It

`code/main.py` implements a stdlib skill library:

- `Skill` — name, description, code (as string), version, tags, dependencies.
- `SkillLibrary` — register, search (token overlap), compose (topological sort of deps), and refine (version bump on update).
- A scripted agent that registers three primitive skills, composes a fourth, hits a failure, and refines.

Run it:

```
python3 code/main.py
```

The trace shows library writes, retrieval, composition, a failed execution, and a v2 refinement — Voyager's loop end to end.

## Use It

- **Claude Agent SDK skills** (Anthropic) — the 2026 reference: each skill has a description, code, and instructions; loaded on demand during an agent session.
- **skillkit** (npm: skillkit) — cross-agent skill management for 32+ AI coding agents.
- **Custom skill libraries** — domain-specific (SQL skills for data agents, Terraform skills for infra agents). The Voyager pattern scales down.
- **OpenAI Agents SDK `tools`** — at the low end; each tool is a lightweight skill.

## Exercises

1. Add a dependency-cycle detector to `compose()`. What happens when skill A depends on B which depends on A? Error vs warning?
2. Implement per-skill version pinning. When a parent skill composes child `crafting@1`, a refinement to `crafting@2` must not silently upgrade the parent.
3. Replace token-overlap retrieval with sentence-transformers embeddings (or a BM25 stdlib impl). Measure retrieval@5 on a 50-skill toy library.
4. Add a "curriculum" agent: given the current library and a domain description, propose 5 missing skills. Call it weekly.
5. Read Anthropic's Claude Agent SDK skill docs. Port the toy library to the SDK's skill schema. What changes about discoverability?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Skill | "Reusable capability" | Named chunk of code + description, retrievable by similarity |
| Skill library | "Agent memory of how-to" | Persistent store of skills, searchable and composable |
| Curriculum | "Task proposer" | Bottom-up goal generator driven by current capability gap |
| Composition | "Skill DAG" | Skills invoking skills; topologically sorted on execution |
| Iterative refinement | "Self-correcting loop" | Env feedback + errors + self-verification fold back into the next version |
| Action-space-as-code | "Programmatic actions" | Emit functions, not primitive commands, for temporally extended behavior |
| Dedup on write | "Skill collapse" | Near-duplicate descriptions collapse to one canonical skill |

## Further Reading

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — the original skill-library paper
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — skills as the 2026 productization
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills and subagents in practice
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — the refinement loop underneath Voyager

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/10-skill-libraries-voyager)
