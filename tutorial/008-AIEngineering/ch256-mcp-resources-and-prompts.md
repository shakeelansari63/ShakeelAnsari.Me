# MCP Resources and Prompts — Context Exposure Beyond Tools

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
