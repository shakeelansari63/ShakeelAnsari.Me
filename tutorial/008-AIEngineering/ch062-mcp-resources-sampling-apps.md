# MCP Resources, Sampling, Roots, Tasks & Apps

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch256): MCP Resources and Prompts — Context Exposure Beyond Tools

> Tools get 90 percent of MCP attention. The other two server primitives solve different problems. Resources expose data for reading; prompts expose reusable templates as slash-commands.

**Type:** Build
**Languages:** Python (stdlib, resource + prompt handler)
**Prerequisites:** Phase 13 · 07
**Time:** ~45 minutes

## Learning Objectives
- Decide between exposing a capability as a tool, a resource, or a prompt
- Implement `resources/list`, `resources/read`, `resources/subscribe`
- Implement `prompts/list` and `prompts/get` with argument templates
- Recognize when hosts surface prompts as slash-commands vs auto-injected context

## The Problem

A naive notes server exposes everything as tools: `notes_read`, `notes_list`, `notes_search`. This forces model-driven tool calls for every data access, blocks subscription/streaming to host UI, and prevents client UIs from surfacing the data.

## The Concept

### Decision rule

| Capability | Primitive |
|------------|-----------|
| User wants to search, filter, or transform data | tool |
| User wants host to include this data as context | resource |
| User wants a templated workflow they can re-run | prompt |

### Resources

`resources/list` returns `{resources: [{uri, name, mimeType}]}`. `resources/read` takes `{uri}` and returns `{contents: [{uri, mimeType, text | blob}]}`.

URIs can be anything: `file://`, `postgres://query/...`, `notes://note-14`, `memory://...`.

### Resource subscriptions

Declare `{resources: {subscribe: true}}` in capabilities. Client calls `resources/subscribe {uri}`. Server sends `notifications/resources/updated` when the resource changes.

### Prompts

`prompts/list` returns `{prompts: [{name, description, arguments}]}`. `prompts/get` takes `{name, arguments}` and returns `{description, messages: [{role, content}]}`.

A prompt is a template that fills to a message list. Example: `code_review` takes `file_path` and returns a three-message sequence.

Hosts like Claude Desktop, VS Code, and Cursor expose prompts as slash-commands. The user types `/code_review` and picks arguments.

### The "list changed" notification

Both resources and prompts emit `notifications/list_changed` when the set mutates. The client re-calls `list` to pick up additions.

### Dynamic resources

A resource URI can compute content dynamically: `notes://recent` returns the latest five notes on every read. If the client can cache by URI, the URI must be stable; if one-shot, include a timestamp.

## Use It

`code/main.py` extends the notes server from Lesson 07 with per-note resources, a `review_note` prompt, a file-watcher simulation emitting `notifications/resources/updated`, and a `notes://recent` dynamic resource.

## Exercises

1. Trigger a note edit and verify `notifications/resources/updated` fires.
2. Add a `resources/list_changed` emitter when a new note is created.
3. Design three prompts for a GitHub MCP server.
4. Reclassify an existing Lesson 07 tool as resource + tool pair.
5. Read the spec and identify the rarely-populated field in `resources/read`.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Resource | URI-addressable content the host can read |
| Resource URI | Scheme-prefixed identifier (`file://`, `notes://`) |
| `resources/subscribe` | Client opt-in for server-push updates on a URI |
| Resource template | Parameterized URI pattern with completion hints |
| Prompt | Named multi-message template with argument slots |
| Slash-command UX | Host surfaces prompts as commands starting with `/` |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/10-mcp-resources-and-prompts)

---

## Part 2 (ch257): MCP Sampling — Server-Requested LLM Completions and Agent Loops

> Most MCP servers are dumb executors. Sampling lets a server flip direction: it asks the client's LLM to make a decision. This enables server-hosted agent loops without the server owning any model credentials.

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07, 10
**Time:** ~75 minutes

## Learning Objectives
- Explain what `sampling/createMessage` solves (server-hosted loops without server-side API keys)
- Implement a server that asks the client to sample over a multi-turn prompt
- Use `modelPreferences` (cost/speed/intelligence priorities) to guide client model selection
- Build a `summarize_repo` tool that internally iterates via sampling

## The Problem

A code-summarization MCP server: Option A calls its own LLM (needs API key, expensive), Option B returns raw content (fragile), Option C asks the client's LLM via `sampling/createMessage` — algorithm on server, billing on client.

## The Concept

### `sampling/createMessage` request

```json
{
  "jsonrpc": "2.0", "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {"costPriority": 0.3, "speedPriority": 0.2, "intelligencePriority": 0.5},
    "maxTokens": 1024
  }
}
```

### `modelPreferences`

Three floats summing to 1.0: `costPriority` (cheaper models), `speedPriority` (faster), `intelligencePriority` (more capable). Plus `hints: [{"name": "claude-3-5-sonnet"}]`.

### Sampling with tools (SEP-1577)

The sampling request can include a `tools` array. The client runs a full tool-calling loop using those tools. Experimental through Q1 2026.

### Human-in-the-loop

The client MUST show the user what the server is asking before running the sample. Claude Desktop, VS Code, and Cursor surface sampling requests as confirmation dialogs.

### Server-hosted loops example

```python
# Step 1: Walk repo structure
# Step 2: Call sampling/createMessage — "Pick five files most likely to describe this repo"
# Step 3: Read those files
# Step 4: Call sampling/createMessage — "Summarize the repo in 3 paragraphs"
# Step 5: Return summary as tools/call result
```

The server never touches an LLM API. The client's user pays for completions.

### Safety risks

- **Covert sampling**: tool always calls sampling with hidden prompts.
- **Resource theft**: server forces client to spend LLM budget.
- **Loop bombs**: server calls sampling in tight loop. Clients MUST rate-limit.

## Use It

`code/main.py` ships a sampling harness. A simulated `summarize_repo` tool invokes two sampling rounds, and the fake client returns canned responses. Rate limiter caps total sampling calls per tool invocation.

## Exercises

1. Change `max_samples_per_tool` to 2 and observe the rate-limit cut-off.
2. Implement the SEP-1577 tool-in-sampling variant.
3. Add human-in-the-loop confirmation before first sampling call.
4. Add per-user rate limiter keyed by client session.
5. Design a `summarize_pdf` tool using sampling with different `intelligencePriority` settings.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Sampling | Server asks client's model for a completion |
| `sampling/createMessage` | JSON-RPC method for sampling requests |
| `modelPreferences` | Cost/speed/intelligence weights plus name hints |
| SEP-1577 | Tools inside sampling for server-hosted ReAct loops |
| Loop bomb | Runaway sampling loop; client must rate-limit |
| Covert sampling | Malicious server hides intent in sampling prompts |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/11-mcp-sampling)

---

## Part 3 (ch258): Roots and Elicitation — Scoping and Mid-Flight User Input

> Hard-coded paths break the moment a user opens a different project. Pre-filled tool arguments break when the user under-specifies. Roots scope the server to a user-controlled set of URIs; elicitation pauses mid-tool-call to ask the user for structured input.

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07
**Time:** ~45 minutes

## Learning Objectives
- Declare `roots` and respond to `notifications/roots/list_changed`
- Restrict server file operations to URIs inside the declared root set
- Use `elicitation/create` to ask the user for confirmation or structured input mid-tool-call
- Choose between form-mode and URL-mode elicitation

## The Problem

Broken path assumptions: server hard-codes `~/notes` but user has notes in `~/Documents/Notes`. Missing arguments: "delete the old TPS report" matches three notes. Roots fix the first; elicitation fixes the second.

## The Concept

### Roots

Client declares roots at `initialize`:

```json
{"capabilities": {"roots": {"listChanged": true}}}
```

Server calls `roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Servers MUST treat roots as the boundary. Any file read/write outside is rejected. When roots change, client sends `notifications/roots/list_changed`.

### Elicitation: form mode

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {"type": "string", "enum": ["note-3", "note-7", "note-14"]},
        "confirm": {"type": "boolean"}},
      "required": ["note_id", "confirm"]}}
}
```

Three outcomes: `accept` (user filled), `decline` (user closed), `cancel` (user aborted tool call).

### Elicitation: URL mode (SEP-1036, experimental)

Instead of a schema, the server sends a URL for OAuth flows, payment authorization, or document signing. Client opens in browser, waits for completion.

### When elicitation is right

- User confirmation before destructive actions
- Disambiguation (pick one of N matches)
- First-run setup (API keys, directories)
- OAuth flows (URL mode)

### When elicitation is wrong

- Arguments the model could ask for in prose (use re-prompt)
- High-frequency calls (interrupts conversation)
- Anything the server could validate after the fact

## Use It

`code/main.py` extends the notes server with roots handling, a `notes_delete` tool that uses elicitation to disambiguate, a `notes_setup` tool with URL-mode elicitation, and a boundary check that rejects out-of-root operations.

## Exercises

1. Trigger the disambiguation path and confirm the simulated user answer routes back.
2. Add a `notes_archive` tool requiring elicitation confirmation every time.
3. Implement URL-mode elicitation for a first-run OAuth flow.
4. Handle `notifications/roots/list_changed` by atomically re-reading roots.
5. Read SEP-1036 and identify one open question about URL-mode callbacks.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Root | Consent boundary URI the client allows the server to touch |
| `roots/list` | Server asks client for the current root set |
| Elicitation | Server-initiated request for structured user input mid-call |
| `elicitation/create` | JSON-RPC method for elicitation requests |
| Form mode | Flat JSON Schema rendered as a form in the client UI |
| URL mode | SEP-1036 experimental; opens a URL and waits |
| Disambiguation | Pick one of N matching candidates |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/12-mcp-roots-and-elicitation)

---

## Part 4 (ch259): Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work

> Real agent work takes minutes to hours: CI runs, deep-research synthesis, batch exports. Synchronous tool calls drop connections, time out, or block the UI. SEP-1686 adds a Tasks primitive: any request can become a task, and the result can be fetched later.

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07, 09
**Time:** ~75 minutes

## Learning Objectives
- Identify when to promote a tool from synchronous to task-augmented (>30 seconds)
- Walk the task lifecycle: `working` → `input_required` → `completed` / `failed` / `cancelled`
- Persist task state so crashes don't lose in-flight work
- Poll `tasks/status` and fetch `tasks/result` correctly

## The Problem

A `generate_report` tool takes three minutes. Options under synchronous model: hold connection (drops), return placeholder (breaks MCP uniformity), fire-and-forget (no result). SEP-1686 adds a fourth: task augmentation.

## The Concept

### Task augmentation

A request becomes a task via `params._meta.task.required: true`. Server responds immediately with a task id:

```json
{"jsonrpc": "2.0", "id": 1, "result": {"_meta": {"task": {"id": "tsk_9f7b...", "state": "working", "ttl": 900000}}}}
```

### Per-tool opt-in

```python
taskSupport: "forbidden"  # always sync (fast tools)
taskSupport: "optional"   # client may request task mode
taskSupport: "required"   # client MUST use task mode
```

### States

```
working -> input_required -> working (loop via elicitation)
working -> completed
working -> failed
working -> cancelled
```

### Methods

- `tasks/status {taskId}` — returns current state and progress
- `tasks/result {taskId}` — returns completed payload or 404
- `tasks/cancel {taskId}` — idempotent
- `notifications/tasks/updated` — server pushes state changes

### Durable state

The spec requires servers to persist task state. The harness uses filesystem storage:

```python
# Task state persisted to /tmp/lesson-13-tasks/<id>.json
{
    "id": "tsk_9f7b...",
    "state": "working",
    "progress": 0.4,
    "ttl": 900000
}
```

### Crash recovery

On restart: load persisted states, mark any `working` tasks as `failed` with `CRASH_RECOVERY`, preserve `completed`/`failed`/`cancelled` for their TTL.

### Async tasks plus sampling

A task can itself call `sampling/createMessage`. Long-running research tasks work by sampling the client's model as needed while showing `working` with progress updates.

## Use It

`code/main.py` implements a durable task store (filesystem-backed) and a `generate_report` tool that runs in a background thread. Clients poll `tasks/status`, cancel mid-run, and crash recovery is simulated.

## Exercises

1. Kick off `generate_report`, poll status, then fetch the result.
2. Cancel mid-run and verify `cancelled` state.
3. Simulate crash recovery and observe `CRASH_RECOVERY`.
4. Extend the store to SQLite.
5. Read the MCP 2026 roadmap and identify the one Tasks-related open issue most likely to affect SDK API design.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Task | Long-running request augmented with `_meta.task` |
| SEP-1686 | Spec Evolution Proposal adding Tasks in 2025-11-25 |
| `_meta.task` | Per-request metadata containing id, state, ttl |
| taskSupport | `forbidden` / `optional` / `required` per tool |
| `tasks/status` | Poll method for current state and progress |
| `tasks/result` | Fetch completed payload or 404 |
| ttl | Milliseconds the server promises to retain state |
| Durable store | Filesystem/SQLite/Redis persistence for crash safety |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/13-mcp-async-tasks)

---

## Part 5 (ch260): MCP Apps — Interactive UI Resources via `ui://`

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
