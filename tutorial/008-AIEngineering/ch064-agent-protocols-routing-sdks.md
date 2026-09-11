# A2A, OTel, Routing, SDKs & Tool Capstone

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch265): A2A — Agent-to-Agent Protocol

> MCP is agent-to-tool. A2A is agent-to-agent — an open protocol for letting opaque agents built on different frameworks collaborate. Released by Google in April 2025, donated to the Linux Foundation in June 2025, v1.0 in April 2026.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06, 08
**Time:** ~75 minutes

## Learning Objectives
- Distinguish agent-to-tool (MCP) from agent-to-agent (A2A) use cases
- Publish an Agent Card at `/.well-known/agent.json` with skills and endpoint metadata
- Walk the Task lifecycle (submitted → working → input-required → completed/failed/canceled/rejected)
- Use Messages with Parts (text, file, data) and Artifacts as outputs

## The Problem

A customer-service agent needs to delegate to a writer agent. Pre-A2A: custom REST API (every pairing is a one-off), shared codebase (requires same framework), or MCP (doesn't fit — MCP is for tools, not opaque agent collaboration).

## The Concept

### Agent Card

Every A2A-compliant agent publishes at `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "skills": [{"id": "summarize_paper", "name": "Summarize a paper",
    "description": "Read a paper PDF and produce a 3-paragraph summary.",
    "inputModes": ["text", "file"], "outputModes": ["text", "artifact"]}],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

### Task lifecycle

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

### Messages and Parts

```json
{"role": "user", "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
]}
```

### Artifacts

Outputs are named, typed artifacts:

```json
{"name": "summary", "parts": [{"type": "text", "text": "..."}], "mimeType": "text/markdown"}
```

### Two transport bindings

1. **JSON-RPC over HTTP** — `/a2a` endpoint, POST for requests, optional SSE for streaming
2. **gRPC** — for enterprise environments

### Opacity preservation

The called agent's internal state is opaque. The caller sees task state and artifacts — never the chain-of-thought, tool calls, or sub-agent delegation. Enables competitors to collaborate without revealing internals.

### Relationship to MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Use MCP for specific tools, A2A for delegating whole tasks. Many systems use both.

## Use It

`code/main.py` implements a minimal A2A harness: a research agent publishes its card, a writer agent receives `tasks/send`, transitions through working → input_required → working → completed, and returns a text artifact.

## Exercises

1. Trace the full Task lifecycle including the input-required clarification pause.
2. Add a signed Agent Card with HMAC verification.
3. Implement task streaming with SSE for incremental artifact chunks.
4. Design an A2A agent wrapping an MCP server; note opacitiy trade-offs.
5. Read the A2A v1.0 announcement and identify one unimplemented feature.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| A2A | Agent-to-Agent protocol for opaque agent collaboration |
| Agent Card | `/.well-known/agent.json` with skills and endpoint |
| Skill | A named callable unit the agent supports |
| Task | Unit of delegation with lifecycle and final artifact |
| Part | Typed chunk: `text` / `file` / `data` in a message |
| Artifact | Named, typed output returned on completion |
| Opacity | Black-box collaboration — internals hidden from caller |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/19-a2a-protocol)

---

## Part 2 (ch266): OpenTelemetry GenAI — Tracing Tool Calls End-to-End

> An agent calls five tools, three MCP servers, and two sub-agents. You need one trace across all of it. The OpenTelemetry GenAI semantic conventions (stable in v1.37+) are the 2026 standard, natively supported by Datadog, Langfuse, Arize Phoenix, OpenLLMetry, and AgentOps.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07, 08
**Time:** ~75 minutes

## Learning Objectives
- Name the required OTel GenAI attributes for LLM and tool-execution spans
- Build a trace hierarchy covering agent loop, LLM call, tool call, and MCP dispatch
- Decide what content to capture (opt-in) vs redact (defaults)
- Emit spans to a local collector without rewriting tool code

## The Problem

"Agent sometimes takes 30 seconds, sometimes 3 seconds." No traces. Logs show the LLM call but not the tool dispatch, MCP round-trip, or sub-agent. Without end-to-end tracing, you can't find the cold-start MCP server that occasionally hangs.

## The Concept

### Span hierarchy

```
agent.invoke_agent (top, INTERNAL)
 ├── llm.chat (CLIENT)
 ├── tool.execute (INTERNAL)
 │    └── mcp.call (CLIENT)
 ├── llm.chat (CLIENT)
 └── subagent.invoke (INTERNAL)
```

One trace id across everything. Parent-child links via `parentSpanId`.

### Required attributes

```python
# LLM span
span.set_attribute("gen_ai.operation.name", "chat")
span.set_attribute("gen_ai.provider.name", "openai")
span.set_attribute("gen_ai.request.model", "gpt-4o")
span.set_attribute("gen_ai.response.model", "gpt-4o-2024-08-06")
span.set_attribute("gen_ai.usage.input_tokens", 150)
span.set_attribute("gen_ai.usage.output_tokens", 42)

# Tool span
span.set_attribute("gen_ai.tool.name", "get_weather")
span.set_attribute("gen_ai.tool.call.id", "call_abc123")

# Agent span
span.set_attribute("gen_ai.agent.name", "research-agent")
```

### Span kinds

- `SpanKind.CLIENT` for calls crossing process boundaries (LLM provider, MCP server)
- `SpanKind.INTERNAL` for agent's own loop steps and tool execution

### Opt-in content capture

By default, spans carry metrics and timing — not prompts/completions. Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` to include content. Review carefully before enabling in production.

### Events on spans

Token-level events: `gen_ai.content.prompt` (input messages), `gen_ai.content.completion` (output messages), `gen_ai.content.tool_call` (tool call as recorded).

### Propagation across MCP

Inject W3C `traceparent` header into requests. Streamable HTTP supports standard headers. For stdio, include `traceparent` in `_meta` of every JSON-RPC call until the spec formalizes it.

### Metrics

Alongside spans: `gen_ai.client.token.usage` (histogram), `gen_ai.client.operation.duration` (histogram), `gen_ai.tool.execution.duration` (histogram).

## Use It

`code/main.py` emits OTel-shaped spans to stdout for an agent that calls an LLM, dispatches two tools, and makes one MCP round-trip. Focus is on span shape and attribute set, not exporters.

## Exercises

1. Count spans and identify CLIENT vs INTERNAL.
2. Turn on content capture and confirm event attributes appear.
3. Add `gen_ai.tool.execution.duration` histogram metric.
4. Propagate `traceparent` from agent span into MCP `_meta`.
5. Identify one attribute from the OTel semconv spec not emitted in this code and add it.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| OTel | Open standard for traces, metrics, logs |
| GenAI semconv | Stable attribute names for LLM/tool/agent spans |
| `gen_ai.*` | The attribute namespace for GenAI operations |
| Span | Timed operation with start, end, and attributes |
| SpanKind | CLIENT / SERVER / INTERNAL direction hints |
| OTLP | Wire format for exporting to backends |
| traceparent | W3C header propagating trace context across services |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/20-opentelemetry-genai)

---

## Part 3 (ch267): LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> Provider lock-in is expensive. Different tool-calling workloads suit different models. Routing gateways give one API surface, retries, failover, cost tracking, and guardrails.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02, 17
**Time:** ~45 minutes

## Learning Objectives
- Distinguish self-hosted, managed, and production-grade routing options
- Implement a fallback chain that retries on provider failures in priority order
- Track per-request cost and token usage across providers
- Decide between LiteLLM, OpenRouter, and Portkey for production constraints

## The Problem

Scenarios: cost (Sonnet 3x Haiku — route accordingly), failover (OpenAI down → auto fallback to Anthropic), latency (live chat needs fast TTFt, batch doesn't), compliance (EU users stay in EU regions), experimentation (A/B test models).

## The Concept

### OpenAI-compatible proxy

Everyone speaks OpenAI shape. The gateway exposes `/v1/chat/completions` and internally proxies to any backend. The client doesn't care.

### Model aliases

```python
ROUTES = {
    "our_smart_model": [
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet"},
        {"provider": "google", "model": "gemini-1.5-pro"},
    ],
    "our_fast_model": [
        {"provider": "anthropic", "model": "claude-3-5-haiku"},
        {"provider": "openai", "model": "gpt-4o-mini"},
    ]
}
```

### Fallback chains

```python
def route(alias):
    for entry in ROUTES[alias]:
        try:
            return call_provider(entry["provider"], entry["model"], ...)
        except ProviderError:
            continue  # fall back to next in chain
    raise NoProviderAvailable
```

### Guardrails

Gateway-level: PII redaction (regex or ML before sending prompts), policy violation rejection, output filters for leak scrubbing.

### Self-hosted vs managed

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source Python | Managed SaaS | Open source + managed |
| Setup | Deploy proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your keys | Credits | Your keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Full control | Rapid prototyping | Compliance |

### Cost tracking

```python
PRICING = {
    ("openai", "gpt-4o"): {"input": 2.50/1M, "output": 10.00/1M},
    ("anthropic", "claude-3-5-sonnet"): {"input": 3.00/1M, "output": 15.00/1M},
}
cost = tokens_in * PRICING[provider, model]["input"] + tokens_out * PRICING[provider, model]["output"]
```

### Routing strategies

- Static priority: first in list, fall back on error
- Load balancing: round-robin or weighted
- Cost-aware: pick cheapest model meeting SLA
- Latency-aware: pick fastest in last N minutes
- Task-aware: prompt classifier routes by category

## Use It

`code/main.py` implements a routing gateway in ~150 lines: accepts OpenAI-shaped requests, runs priority fallback chain, tracks per-request cost, and applies PII redaction on inputs.

## Exercises

1. Trigger the outage scenario and confirm fallback with correct cost attribution.
2. Add semantic caching: SHA256 prompt key with instant cache hits.
3. Add a prompt classifier routing "code" vs "summarize" to different aliases.
4. Design per-team budgets with monthly spend caps.
5. Read LiteLLM, OpenRouter, and Portkey docs; name one unique feature each ships.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Routing gateway | One-API-surface layer in front of many providers |
| OpenAI-compatible | Accepts `/v1/chat/completions`, translates to any backend |
| Model alias | Name your code uses; gateway maps to concrete model |
| Fallback chain | Ordered provider list attempted on failure |
| Semantic caching | Embedding-based cache; near-duplicates share a hit |
| Guardrails | Input/output filters: PII redaction, policy enforcement |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/21-llm-routing-layer)

---

## Part 4 (ch268): Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP says "what tools exist." Skills say "how to do a task." The 2026 stack layers both. Anthropic's Agent Skills ship as SKILL.md with progressive disclosure. OpenAI's Apps SDK is MCP plus widget metadata. AGENTS.md sits at the repo root as project-level agent context.

**Type:** Learn
**Languages:** Python (stdlib, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07
**Time:** ~45 minutes

## Learning Objectives
- Distinguish the three layers: AGENTS.md (project context), SKILL.md (reusable know-how), MCP (tools)
- Write a SKILL.md with YAML frontmatter and progressive disclosure
- Load skills filesystem-style into an agent runtime
- Compose a skill with an MCP server and an AGENTS.md for cross-agent portability

## The Problem

An engineer distills a release-notes-writing workflow. They want to use it from Claude Code, Cursor, and Codex CLI. Pre-2026: copy the workflow three times. AGENTS.md + SKILL.md fix this.

## The Concept

### AGENTS.md

One file at repo root. Every coding agent in 2026 supports it.

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

### SKILL.md format

Anthropic's Agent Skills (open standard, December 2025):

```markdown
---
name: release-notes-writer
description: Write changelog entry for latest merged PRs.
---

# Release notes writer

1. List PRs merged since last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR: `- <title> (#<num>)`.
4. Draft in CHANGELOG.md.

## Notes
- Never include commits without a PR.
- Skip "chore" entries from public changelog.
```

### Progressive disclosure

Skills reference sub-resources fetched only when needed:

```
skills/release-notes-writer/
  SKILL.md
  style-guide.md
  template.md
  scripts/generate.sh
```

SKILL.md says "see style-guide.md." Agent pulls it only when the skill runs, avoiding prompt bloat.

### Filesystem discovery

Agent runtimes scan: `~/.anthropic/skills/*/SKILL.md`, `./skills/*/SKILL.md`, `~/.claude/skills/*/SKILL.md`.

### The three-layer stack

| Layer | File | Loaded when | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | Project conventions |
| SKILL.md | skills/ directory | skill invoked | Reusable workflow |
| MCP server | external process | tools needed | Callable actions |

### Cross-agent portability via SkillKit

Tools like SkillKit translate a single SKILL.md into the native format of 32+ AI agents. One source of truth, many consumers.

### OpenAI Apps SDK

Launched October 2025. Built directly on MCP: an MCP server (tools, resources, prompts) plus widget metadata for ChatGPT's UI, plus optional `ui://` resources.

## Use It

`code/main.py` ships a stdlib SKILL.md parser and loader. It discovers skills under `./skills/`, parses YAML frontmatter plus markdown body, and simulates an agent loop invoking a skill by name.

## Exercises

1. Add a second skill under `skills/` and confirm the loader picks it up.
2. Write an AGENTS.md for this course repo.
3. Port a multi-step workflow from your team's docs into a SKILL.md.
4. Translate the skill into Cursor's and Codex's native formats.
5. Read the Anthropic Agent Skills blog post; identify one SDK feature the loader doesn't cover.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| SKILL.md | YAML frontmatter + markdown body, loaded by agent runtime |
| AGENTS.md | Repo-root project conventions, read on session start |
| Progressive disclosure | Lazy-load sub-resources from skill directory |
| Frontmatter | YAML metadata block in `---` delimiters |
| Claude Agent SDK | Anthropic's skill runtime (`@anthropic-ai/claude-agent-sdk`) |
| OpenAI Apps SDK | MCP + widget metadata for ChatGPT UI |
| SkillKit | Cross-agent translator covering 32+ agent formats |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/22-skills-and-agent-sdks)

---

## Part 5 (ch269): Capstone — Build a Complete Tool Ecosystem

> Phase 13 taught every piece. This capstone wires them into one production-shaped system: an MCP server with tools + resources + prompts + tasks + UI, OAuth 2.1 at the edge, an RBAC gateway, a multi-server client, an A2A sub-agent call, OTel tracing into a collector, tool-poisoning detection in CI, and an AGENTS.md + SKILL.md bundle.

**Type:** Build
**Languages:** Python (stdlib, end-to-end ecosystem harness)
**Prerequisites:** Phase 13 · 01 through 21
**Time:** ~120 minutes

## Learning Objectives
- Compose an MCP server exposing tools, resources, prompts, tasks, and a `ui://` app
- Front the server with an OAuth 2.1 gateway enforcing RBAC and pinned hashes
- Write a multi-server client that traces with OTel GenAI attributes end-to-end
- Delegate part of a workload to an A2A sub-agent; verify opacity is preserved
- Package the whole stack with AGENTS.md + SKILL.md

## The Problem

Ship the "research and report" system: search arXiv via MCP, delegate paper summarization to a writer agent via A2A, aggregate results, render an interactive report as `ui://`, log every step to OTel.

## The Concept

### Architecture

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                       |
                                                       +- MCP tool: arxiv_search (pure)
                                                       +- MCP resource: notes://recent
                                                       +- MCP prompt: /research_topic
                                                       +- MCP task: generate_report (long)
                                                       +- MCP Apps UI: ui://report/current
                                                       +- A2A call: writer-agent (tasks/send)
                                                       |
                                                       +- OTel GenAI spans
```

### Trace hierarchy

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (returns ui:// resource)
 └── llm.chat (final synthesis)
```

### Security posture

- OAuth 2.1 + PKCE with resource indicator pinning to gateway
- Gateway holds upstream credentials; user never sees them
- RBAC: `alice` has `research:read/write`, `bob` has `research:read` only
- Pinned description manifest: rejects servers with changed tool hashes
- Rule of Two: no tool combines untrusted input, sensitive data, and consequential action

### Rendering

The final `generate_report` task returns content blocks plus a `ui://report/current` resource. The host renders an interactive dashboard with sorted paper list, citation counts, and a button calling `host.callTool('summarize_paper', {arxiv_id})`.

### Packaging

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # top-level workflow
  servers/
    research-mcp/               # MCP server
  agents/
    writer/                     # A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

### What each lesson contributed

| Lesson | Capstone usage |
|--------|----------------|
| 01-05 | Tool interface, provider-portability, parallel calls, schemas, linting |
| 06-10 | MCP primitives, server, client, transports, resources + prompts |
| 11-14 | Sampling, roots + elicitation, async tasks, `ui://` apps |
| 15-17 | Tool poisoning, OAuth 2.1, gateway + registry |
| 18 | A2A sub-agent delegation |
| 19 | OTel GenAI tracing |
| 20 | Routing gateway for LLM layer |
| 21 | SKILL.md + AGENTS.md packaging |

## Use It

`code/main.py` stitches all previous lessons' patterns into one runnable demo. All stdlib, all in-process. Runs the full research-and-report flow: gateway handshake, OAuth 2.1, tools/list merge, task-augmented report, A2A call, `ui://` resource, OTel spans.

## Exercises

1. Run the demo and count how many primitives from Phase 13 it touches.
2. Add a second backend MCP server (e.g. `bibliography`) and verify namespace merging.
3. Replace the fake A2A writer with a real subprocess (use Lesson 19 harness).
4. Add PII redaction in the routing gateway between orchestrator and LLM.
5. Write an AGENTS.md for a teammate — under 5 minutes to read, everything needed to drive the capstone.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Capstone | End-to-end system using every Phase 13 primitive |
| Research and report | Search, summarize, render pattern |
| Trace hierarchy | Single trace id across every hop |
| Gateway-issued token | Client sees only gateway's token; gateway holds upstream creds |
| Opacity boundary | A2A sub-agent's reasoning invisible to orchestrator |
| Three-layer stack | AGENTS.md + SKILL.md + MCP |
| Defense-in-depth | Pinned hashes, OAuth, RBAC, Rule of Two, audit log |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/23-capstone-tool-ecosystem)
