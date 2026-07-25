# Building an MCP Server — Python + TypeScript SDKs

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
