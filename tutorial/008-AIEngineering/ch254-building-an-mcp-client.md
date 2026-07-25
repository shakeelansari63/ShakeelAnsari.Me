# Building an MCP Client — Discovery, Invocation, Session Management

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
