# Capstone — Build a Complete Tool Ecosystem

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
