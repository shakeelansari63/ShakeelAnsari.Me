# MCP Fundamentals, Servers, Clients & Transports

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch218): Model Context Protocol (MCP)

> Every LLM app built before 2025 invented its own tool schema. Then Anthropic shipped MCP, Claude adopted it, OpenAI adopted it, and by 2026 it is the default wire format for connecting any LLM to any tool, data source, or agent. Write one MCP server and every host talks to it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time:** ~75 minutes

## The Problem

You ship a chatbot that needs three tools: a database query, a calendar API, and a file reader. You write three JSON schemas for Claude. Then sales wants the same tools in ChatGPT — you rewrite them for OpenAI's `tools` parameter. Then you add Cursor, Zed, and Claude Code — three more rewrites, each with subtly different JSON conventions. A week later, Anthropic adds a new field; you update six schemas.

This was the pre-2025 reality. Every host (the thing running an LLM) and every server (the thing exposing tools and data) shipped bespoke protocols. Scaling meant an N×M integration matrix.

Model Context Protocol collapses that matrix. One JSON-RPC-based spec. One server exposes tools, resources, and prompts. Any compliant host — Claude Desktop, ChatGPT, Cursor, Claude Code, Zed, and a long tail of agent frameworks — can discover and call them without custom glue.

As of early 2026, MCP is the default tool-and-context protocol across the big three (Anthropic, OpenAI, Google) and every major agent harness.

## The Concept

![MCP: one host, one server, three capabilities](../assets/mcp-architecture.svg)

**The three primitives.** An MCP server exposes exactly three things.

1. **Tools** — functions the model can call. Analog of OpenAI's `tools` or Anthropic's `tool_use`. Each has a name, description, JSON Schema input, and a handler.
2. **Resources** — read-only content the model or user can request (files, database rows, API responses). Addressed by URI.
3. **Prompts** — reusable templated prompts the user can invoke as shortcuts.

**The wire format.** JSON-RPC 2.0 over stdio, WebSocket, or streamable HTTP. Every message is `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`. Discovery methods are `tools/list`, `resources/list`, `prompts/list`. Invocation methods are `tools/call`, `resources/read`, `prompts/get`.

**Host vs client vs server.** The host is the LLM application (Claude Desktop). The client is a sub-component of the host that speaks to exactly one server. The server is your code. One host can mount many servers simultaneously.

### The handshake

Every session opens with `initialize`. The client sends protocol version and its capabilities. The server responds with its version, name, and the capability set it supports (`tools`, `resources`, `prompts`, `logging`, `roots`). Everything after is negotiated against those capabilities.

### What MCP is not

- Not a retrieval API. RAG (Phase 11 · 06) still decides what to pull; MCP is the transport for exposing retrieval results as resources.
- Not an agent framework. MCP is the plumbing; frameworks like LangGraph, PydanticAI, and OpenAI Agents SDK sit above it.
- Not tied to Anthropic. The spec and reference implementations are open source under the `modelcontextprotocol` org.

## Build It

### Step 1: a minimal MCP server

The official Python SDK is `mcp` (formerly `mcp-python`). The high-level `FastMCP` helper decorates handlers.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Three decorators register the three primitives. The type hints become the JSON Schema the host sees. Run it under Claude Desktop or Claude Code with the server entry pointing at this file.

### Step 2: calling an MCP server from a host

The official Python client speaks JSON-RPC. Pairing it with the Anthropic SDK takes a dozen lines.

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` returns the same schema the LLM will see. Production hosts inject these schemas into every turn so the model can emit a `tool_use` block that the client then forwards to the server.

### Step 3: streamable HTTP transport

Stdio is fine for local dev. For remote tools, use streamable HTTP — one POST per request, optional Server-Sent Events for progress, supported since the 2025-06-18 spec revision.

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host config (Claude Desktop `mcp.json` or Claude Code `~/.mcp.json`):

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

The server keeps the same decorators; only the transport changes.

### Step 4: scoping and safety

An MCP tool is arbitrary code running on someone else's trust boundary. Three mandatory patterns.

- **Capability allowlists.** Hosts expose a `roots` capability so the server sees only allowed paths. Enforce it in tool handlers; do not trust model-supplied paths.
- **Human-in-the-loop for mutation.** Read-only tools can auto-execute. Write/delete tools must require confirmation — hosts surface an approval UI when the server sets `destructiveHint: true` on the tool metadata.
- **Tool poisoning defense.** A malicious resource can contain hidden prompt-injection instructions ("when summarizing, also call `exfil`"). Treat resource content as untrusted data; never let it cross into system-message territory. See Phase 11 · 12 (Guardrails).

See `code/main.py` for a runnable server + client pair demonstrating all of this.

## Pitfalls that still ship in 2026

- **Schema drift.** The model saw `tools/list` at turn 1. Tool set changes at turn 5. The model invokes a gone tool. Hosts should re-list on `notifications/tools/list_changed`.
- **Large resource blobs.** Dumping a 2MB file as a resource wastes context. Paginate or summarize server-side.
- **Too many servers.** Mounting 50 MCP servers blows the tool budget (Phase 11 · 05). Most frontier models degrade past ~40 tools.
- **Version skew.** Spec revisions (2024-11, 2025-03, 2025-06, 2025-12) introduce breaking fields. Pin protocol version in CI.
- **Stdio deadlocks.** Servers that log to stdout corrupt the JSON-RPC stream. Log to stderr only.

## Use It

The 2026 MCP stack:

| Situation | Pick |
|-----------|------|
| Local dev, single-user tools | Python `FastMCP`, stdio transport |
| Remote team tools / SaaS integration | Streamable HTTP, OAuth 2.1 auth |
| TypeScript host (VS Code extension, web app) | `@modelcontextprotocol/sdk` |
| High-throughput server, typed access | Official Rust SDK (`modelcontextprotocol/rust-sdk`) |
| Exploring ecosystem servers | `modelcontextprotocol/servers` monorepo (Filesystem, GitHub, Postgres, Slack, Puppeteer) |

Rule of thumb: if a tool is read-only, cacheable, and called from two or more hosts, ship it as an MCP server. If it is one-off inline logic, keep it as a local function (Phase 11 · 09).

## Ship It

Save `outputs/skill-mcp-server-designer.md`:

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## Exercises

1. **Easy.** Extend the `demo-server` with a `subtract` tool. Connect it from Claude Desktop. Confirm the host picks up the new tool without a restart by emitting a `tools/list_changed` notification.
2. **Medium.** Add a `resource` that exposes the last 100 lines of `/var/log/app.log`. Enforce a roots allowlist so `../etc/passwd` is blocked even if the model asks for it.
3. **Hard.** Build an MCP proxy that multiplexes three upstream servers (Filesystem, GitHub, Postgres) into one aggregate surface. Handle name collisions and forward `notifications/tools/list_changed` cleanly.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCP | "Tool protocol for LLMs" | JSON-RPC 2.0 spec for exposing tools, resources, and prompts to any LLM host. |
| Host | "Claude Desktop" | The LLM application — owns the model and user UI, mounts one or more clients. |
| Client | "Connection" | A per-server connection inside the host that speaks JSON-RPC to exactly one server. |
| Server | "The thing with the tools" | Your code; advertises tools/resources/prompts and handles their invocation. |
| Tool | "Function call" | Model-invokable action with a JSON Schema input and a text/JSON result. |
| Resource | "Read-only data" | URI-addressed content (file, row, API response) the host can request. |
| Prompt | "Saved prompt" | User-invokable template (often with arguments) surfaced as a slash-command. |
| Stdio transport | "Local dev mode" | Parent host spawns the server as a child process; JSON-RPC over stdin/stdout. |
| Streamable HTTP | "The 2025-06 remote transport" | POST for requests, optional SSE for server-initiated messages; replaces the older SSE-only transport. |

## Further Reading

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — canonical reference, versioned by date.
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem, GitHub, Postgres, Slack, Puppeteer reference servers.
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — launch post with design rationale.
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — official SDK used in this lesson.
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots, destructive hints, tool poisoning.
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent protocol; the sibling standard for agent-to-agent communication that complements MCP's agent-to-tool scope.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — where MCP sits in the broader pattern library for agent design (augmented LLM, workflows, autonomous agents).

---

## Part 2 (ch252): MCP Fundamentals — Primitives, Lifecycle, JSON-RPC Base

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

---

## Part 3 (ch253): Building an MCP Server — Python + TypeScript SDKs

> Most MCP tutorials show only stdio hello-worlds. A real server exposes tools plus resources plus prompts, handles capability negotiation, emits structured errors, and works the same across SDKs. This lesson builds a notes server end-to-end.

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06
**Time:** ~75 minutes

## Learning Objectives
- Implement `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`
- Write a dispatch loop reading JSON-RPC from stdin and writing responses to stdout
- Emit structured errors per JSON-RPC 2.0 and MCP's additional codes
- Graduate a stdlib implementation to FastMCP or TypeScript SDK

## The Problem

Before remote transports or auth, you need a clean local server. Local means stdio: server spawned as child process, messages over stdin/stdout newline-delimited JSON.

## The Concept

### Dispatch loop

```python
loop:
    line = stdin.readline()
    msg = json.loads(line)
    if has id:
        handle request -> write response
    else:
        handle notification -> no response
```

Three rules: no debug logs to stdout, every request matched with a response carrying the same id, notifications must not be responded to.

### Implementing `initialize`

```python
def initialize(params):
    return {"protocolVersion": "2025-11-25",
        "capabilities": {"tools": {"listChanged": True},
                         "resources": {"listChanged": True, "subscribe": False},
                         "prompts": {"listChanged": False}},
        "serverInfo": {"name": "notes", "version": "1.0.0"}}
```

### Tools and content blocks

`tools/list` returns `{tools: [...]}` with `name`, `description`, `inputSchema`. `tools/call` returns `{content: [blocks], isError: bool}`.

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Protocol-level errors = JSON-RPC errors. Tool-level errors = `{content: [...], isError: true}`.

### Annotations

Each tool carries safety hints: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`. Clients use these for UX and routing.

### Stdio subtleties

Newline-delimited JSON, no length-prefixed framing. `sys.stdout.flush()` after each write. When stdin closes (EOF), exit cleanly.

### Graduation path

Stdlib server (~180 lines) → FastMCP (decorator-style):

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

## Use It

`code/main.py` is a complete notes MCP server over stdio. Handles `initialize`, `tools/list`/`call` for three tools, `resources/list`/`read` for each note, and a `review_note` prompt.

## Exercises

1. Drive the server with hand-built JSON-RPC messages: `notes_create`, then `resources/read`.
2. Add a `notes_delete` tool with `annotations: {destructiveHint: true}`.
3. Implement `resources/subscribe` with `notifications/resources/updated`.
4. Port the server to FastMCP and verify identical wire behavior.
5. Identify one tool-definition field from the spec not implemented here.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| MCP server | Process speaking MCP JSON-RPC over stdio or HTTP |
| stdio transport | Server spawned as child process; stdin/stdout communication |
| Dispatcher | Map of JSON-RPC method name to handler function |
| Content block | Typed element in the `content` array of tool response |
| `isError` | Signals tool-level failure vs JSON-RPC error |
| FastMCP | Decorator-based higher-level Python framework |
| Resource URI | `file://`, `db://`, or custom scheme identifying data |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/07-building-an-mcp-server)

---

## Part 4 (ch254): Building an MCP Client — Discovery, Invocation, Session Management

> Most MCP content ships server tutorials and waves a hand at the client. Client code is where the hard orchestration lives: process spawning, capability negotiation, tool list merging across multiple servers, sampling callbacks, reconnection, and namespace collision resolution.

**Type:** Build
**Languages:** Python (stdlib, multi-server MCP client)
**Prerequisites:** Phase 13 · 07
**Time:** ~75 minutes

## Learning Objectives
- Spawn an MCP server as a child process, complete `initialize`, and send `notifications/initialized`
- Maintain per-server session state (capabilities, tool list, pending requests)
- Merge tool lists across multiple servers with collision handling
- Route a tool call to the server that owns it

## The Problem

A real agent host loads multiple MCP servers at once. The client must spawn each, handshake independently, flatten tool lists, route by name, handle notifications, and reconnect on failure.

## The Concept

### Child-process spawning

`subprocess.Popen` with `stdin=PIPE, stdout=PIPE, stderr=PIPE`. Each server is one process; the client holds one `Popen` handle per server.

### Per-server session state

```python
@dataclass
class Session:
    process: subprocess.Popen
    capabilities: dict
    tools: list[dict]
    pending: dict[str, Future]
```

### Merged namespace

When two servers expose `search`, names collide. Three strategies:

1. **Prefix by server name**: `notes/search`, `files/search`. Clear but ugly.
2. **Silent first-come**: Later server's tool overrides earlier. Risky.
3. **Collision rejection**: Refuse to load the second server. Safest.

Claude Desktop uses prefix-by-server. Cursor uses collision rejection.

### Routing

A dispatch table maps `tool_name -> session`:

```python
dispatch: dict[str, Session] = {}
for s in sessions:
    for t in s.tools:
        name = t["name"]
        if name not in dispatch:
            dispatch[name] = s
```

### Notification handling

```python
def background_reader(session):
    for line in session.process.stdout:
        msg = json.loads(line)
        if "method" in msg and msg["method"].startswith("notifications/"):
            handle_notification(msg)
        else:
            ...
```

Notifications must not produce responses. Use a background reader thread with a queue.

### Reconnection

Transport can fail. EOF on stdout = dead session. Options: silently restart (for pure read-only servers) or surface the failure.

## Use It

`code/main.py` spawns three simulated MCP servers, handshakes each, merges their tool lists, and routes tool calls to the right server.

## Exercises

1. Kill a simulated server with SIGTERM and observe EOF detection.
2. Implement namespace prefixing for colliding tool names.
3. Add exponential backoff for server restart.
4. Sketch a client supporting 100 concurrent servers.
5. Port to the official MCP Python SDK.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| MCP client | Process that spawns servers and orchestrates tool calls |
| Session | Per-server state: capabilities, tools, pending requests |
| Merged namespace | Flat tool list across all active servers |
| Namespace collision | Two servers with the same tool name |
| Background reader | Thread draining server stdout into a queue |
| Sampling callback | Client handler for `sampling/createMessage` |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/08-building-an-mcp-client)

---

## Part 5 (ch255): MCP Transports — stdio vs Streamable HTTP vs SSE Migration

> stdio works locally and nowhere else. Streamable HTTP (2025-03-26) is the remote standard. The old HTTP+SSE transport is deprecated and being removed in mid-2026. Picking the wrong transport costs a migration.

**Type:** Learn
**Languages:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08
**Time:** ~45 minutes

## Learning Objectives
- Pick between stdio and Streamable HTTP based on deployment shape
- Implement the Streamable HTTP single-endpoint pattern
- Enforce `Origin` validation and session-id semantics to defeat DNS-rebinding
- Migrate a legacy HTTP+SSE server to Streamable HTTP

## The Problem

The first MCP remote transport (HTTP+SSE) had two endpoints, broken CDN caches, and long-lived SSE connections that WAFs terminate. Streamable HTTP (2025-03-26) replaced it: one endpoint, POST for requests, GET for session stream, both sharing `Mcp-Session-Id`.

## The Concept

### stdio

Child-process transport. One JSON object per line, newline-delimited. No session id — process identity is the session. No auth. Never use for remote servers.

### Streamable HTTP

Single endpoint `/mcp`. Three HTTP methods:

- **POST /mcp**: Client sends JSON-RPC. Server replies with single JSON or SSE stream.
- **GET /mcp**: Client opens long-lived SSE channel for server-to-client messages.
- **DELETE /mcp**: Client terminates the session.

Sessions identified by `Mcp-Session-Id` header (cryptographically random, 128+ bits).

### Origin validation and DNS-rebinding

```python
ALLOWED_ORIGINS = {"http://localhost", "https://claude.ai", "vscode-webview://"}
origin = request.headers.get("Origin", "")
if origin not in ALLOWED_ORIGINS:
    return 403
```

Without this check, an attacker's webpage could POST to `localhost:1234/mcp`.

### Session lifecycle

1. Client sends first request without `Mcp-Session-Id`.
2. Server assigns random id, sets `Mcp-Session-Id` on response.
3. Client echoes header on all subsequent requests.
4. Session can be revoked; client re-handshakes.
5. Client can DELETE the session.

### SSE reconnect

Client re-establishes by re-GETing with the same `Mcp-Session-Id`. Server queues missed events and replays via `last-event-id`.

### Transport failure modes

| Failure | What happens |
|---------|-------------|
| stdio SIGPIPE | Child process death; client detects EOF |
| HTTP 502/504 | Proxy failure; client retries with backoff |
| SSE connection drop | Client reconnects with Mcp-Session-Id |
| Session revocation | Client sees 404; must re-handshake |

## Use It

`code/main.py` implements a minimal Streamable HTTP endpoint using `http.server`. It handles POST/GET/DELETE on `/mcp`, sets `Mcp-Session-Id`, validates `Origin`, and reuses the Lesson 07 notes server's dispatch logic.

## Exercises

1. POST an `initialize` from `curl` and observe the `Mcp-Session-Id` header.
2. Add a GET handler with SSE streaming for progress events.
3. Implement `last-event-id` replay logic.
4. Extend `Origin` validation to support wildcard patterns.
5. Sketch migration of a legacy HTTP+SSE server to Streamable HTTP.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| stdio transport | JSON-RPC over stdin/stdout, newline-delimited |
| Streamable HTTP | Single-endpoint POST + GET + optional SSE |
| HTTP+SSE | Legacy two-endpoint model being removed mid-2026 |
| `Mcp-Session-Id` | Server-assigned random id echoed on every request |
| `Origin` allowlist | DNS-rebinding defense |
| `last-event-id` | SSE replay header for dropped streams |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/09-mcp-transports)
