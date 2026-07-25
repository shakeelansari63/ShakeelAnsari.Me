# MCP Fundamentals — Primitives, Lifecycle, JSON-RPC Base

> Every integration before MCP was a one-off. The Model Context Protocol standardizes discovery and invocation so any client can speak to any server. The 2025-11-25 spec names six primitives, a three-phase lifecycle, and a JSON-RPC 2.0 wire format.

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05
**Time:** ~45 minutes

## Learning Objectives
- Name all six MCP primitives (tools, resources, prompts on server; roots, sampling, elicitation on client)
- Walk the three-phase lifecycle (initialize, operation, shutdown)
- Parse and emit JSON-RPC 2.0 request, response, and notification envelopes
- Explain what capability negotiation at `initialize` is and what breaks without it

## The Problem

Before MCP, every tool-using agent had its own protocol. A team that built a "Postgres query" tool wrote it three times. MCP fixes this by standardizing the wire format. 300+ clients by April 2026, 110M monthly SDK downloads, 10,000+ public servers.

## The Concept

### Three server primitives

1. **Tools** — callable actions (the four-step loop from Lesson 01).
2. **Resources** — exposed data, URI-addressable (`file://`, `db://`, custom schemes).
3. **Prompts** — reusable templates, slash-commands in the host UI.

### Three client primitives

4. **Roots** — set of URIs the server may touch.
5. **Sampling** — server requests the client's LLM for completions.
6. **Elicitation** — server asks the user for structured input mid-flight.

### Wire format: JSON-RPC 2.0

- Requests: `{jsonrpc: "2.0", id, method, params}`
- Responses: `{jsonrpc: "2.0", id, result | error}`
- Notifications: `{jsonrpc: "2.0", method, params}` (no id, no response)

Key methods: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `sampling/createMessage`, `notifications/tools/list_changed`.

### Three-phase lifecycle

**Initialize**: Client sends `initialize` with capabilities. Server responds with capabilities and spec version. Client sends `notifications/initialized`.

**Operation**: Bidirectional. Client calls `tools/list`, `tools/call`. Server may send `sampling/createMessage` or notifications.

**Shutdown**: Either side closes the transport. No structured shutdown method.

### Capability negotiation

```json
// Server declares
{"tools": {"listChanged": true}, "resources": {"subscribe": true, "listChanged": true}}

// Client declares
{"roots": {"listChanged": true}, "sampling": {}, "elicitation": {}}
```

If the client doesn't declare `sampling`, the server must not call `sampling/createMessage`. Symmetric for server capabilities.

## Use It

`code/main.py` ships a minimal JSON-RPC 2.0 parser and walks the `initialize` → `tools/list` → `tools/call` → shutdown sequence, printing every message.

## Exercises

1. Identify the capability negotiation line and describe what changes if the server omits `tools.listChanged`.
2. Extend the parser to handle `notifications/progress`.
3. Read the MCP spec and identify the capability flag most servers do not need.
4. Sketch where a "cron job" feature would belong among the six primitives.
5. Parse a real MCP session log and count request vs response vs notification messages.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| MCP | Open protocol for model-to-tool discovery and invocation |
| Server primitive | tools (actions), resources (data), prompts (templates) |
| Client primitive | roots (scope), sampling (LLM callbacks), elicitation (user input) |
| JSON-RPC 2.0 | Symmetric request/response/notification envelopes |
| `initialize` handshake | Capability negotiation to prevent ecosystem drift |
| Content block | Typed result: `{type: "text" | "image" | "resource" | "ui_resource"}` |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/06-mcp-fundamentals)
