# MCP Transports — stdio vs Streamable HTTP vs SSE Migration

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
