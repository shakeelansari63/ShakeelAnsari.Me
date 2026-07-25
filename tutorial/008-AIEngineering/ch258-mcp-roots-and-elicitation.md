# Roots and Elicitation — Scoping and Mid-Flight User Input

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
