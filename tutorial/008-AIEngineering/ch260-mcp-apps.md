# MCP Apps — Interactive UI Resources via `ui://`

> Text-only tool output caps what agents can show. MCP Apps (SEP-1724, January 2026) let a tool return sandboxed interactive HTML rendered inline in Claude Desktop, ChatGPT, Cursor, Goose, and VS Code.

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites:** Phase 13 · 07, 10
**Time:** ~75 minutes

## Learning Objectives
- Return a `ui://` resource from a tool call with the correct MIME and metadata
- Declare a tool's associated UI with `_meta.ui.resourceUri`, CSP, and permissions
- Implement the iframe sandbox postMessage JSON-RPC for UI-to-host communication
- Apply CSP and permissions-policy defaults that defend against UI-originated attacks

## The Problem

A 2025-era `visualize_timeline` tool returns "Here are 14 notes organized chronologically." Users want the interactive timeline. MCP Apps standardizes the contract: a tool result contains a `ui://` resource with MIME `text/html;profile=mcp-app`. The host renders it in a sandboxed iframe.

## The Concept

### The `ui://` resource scheme

```json
{"content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {"ui": {"resourceUri": "ui://notes/timeline",
                    "csp": {"defaultSrc": "'self'", "scriptSrc": "'self' 'unsafe-inline'"},
                    "permissions": []}}}
```

The host calls `resources/read` on `ui://notes/timeline` and gets HTML.

### Iframe sandbox

Rendered in a sandboxed `<iframe>` with:
- `sandbox="allow-scripts allow-same-origin"`
- Server-declared CSP applied via response headers
- No cookies or localStorage from the host's origin

### postMessage protocol

```javascript
// iframe to host
window.parent.postMessage({
  jsonrpc: "2.0", id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14" } }
}, "https://host.example.com");

// Always pin targetOrigin and validate event.origin on receive
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

Available host methods: `host.callTool`, `host.readResource`, `host.getPrompt`, `host.close`.

### Permissions

`_meta.ui.permissions` requests extra capabilities: `camera`, `microphone`, `geolocation`, `network:*`. Each is a prompt the user sees before the UI renders.

### `ui/initialize` handshake

After iframe loads, it sends `ui/initialize` with theme, locale, sessionId. Host responds with capabilities and a session token.

### Security risks

- **Prompt-injection via UI**: malicious UI text looks like system messages
- **Exfiltration via `connectSrc`**: if CSP allows `*`, UI can send data anywhere
- **Clickjacking**: UI overlays host chrome

## Use It

`code/main.py` extends the notes server with a `visualize_timeline` tool returning a `ui://notes/timeline` resource. The HTML is stdlib-templated with an SVG timeline. postMessage is documented but inert in this stdlib demo.

## Exercises

1. Inspect the emitted HTML and sketch the postMessage contract for `host.callTool`.
2. Tighten CSP: remove `'unsafe-inline'` and use nonce-based script policy.
3. Add a `ui://notes/editor` with a form that calls `host.callTool("notes_update", ...)`.
4. Audit the UI's attack surface: what does iframe sandbox defend and miss?
5. Read SEP-1724 and identify one capability not used in this implementation.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| MCP Apps | SEP-1724 extension for interactive UI resources (2026-01-26) |
| `ui://` | App URI scheme for UI bundles |
| `text/html;profile=mcp-app` | Content-type for MCP App HTML |
| Iframe sandbox | Browser sandboxing with CSP and permissions |
| postMessage JSON-RPC | Tiny JSON-RPC-over-postMessage dialect for host calls |
| `_meta.ui` | Metadata linking a tool result to a UI resource |
| `ui/initialize` | First postMessage handshake from UI to host |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/14-mcp-apps)
