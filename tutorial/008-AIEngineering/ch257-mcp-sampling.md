# MCP Sampling — Server-Requested LLM Completions and Agent Loops

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
