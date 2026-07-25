# A2A — Agent-to-Agent Protocol

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
