# A2A — The Agent-to-Agent Protocol

> Google announced A2A in April 2025; by April 2026 the spec is at https://a2a-protocol.org/latest/specification/ and 150+ organizations back it. A2A is the horizontal complement to MCP: where MCP is vertical (agent ↔ tools), A2A is peer-to-peer (agent ↔ agent).

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Your agent needs to call another agent on another system. How? You can expose an HTTP endpoint, define a bespoke JSON schema, and hope the other side speaks it. Every pair of agents becomes a custom integration.

A2A is the universal wire protocol for that call. Standard discovery, standard task model, standard transport, standard artifacts.

## Concept

### The four elements

**Agent Card.** A JSON document at `/.well-known/agent.json` describing the agent: name, skills, endpoints, supported modalities, auth requirements.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": { "tasks": "https://agent.example.com/tasks" },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task.** The unit of work. An async, stateful object with a lifecycle: `submitted → working → completed / failed / canceled`.

**Artifact.** The result type produced by a task. Text, structured JSON, image, video, audio.

**Opaque lifecycle.** A2A does not prescribe *how* the remote agent solves the task. The client sees state transitions and artifacts.

### The MCP/A2A split

- **MCP**: agent ↔ tool. The agent reads/writes via JSON-RPC to a tool server.
- **A2A**: agent ↔ agent. Peer protocol; both sides are agents with their own reasoning.

Production multi-agent systems use both.

### Discovery flow

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts
```

### Auth

A2A supports three common patterns: Bearer token (OAuth2), mTLS, signed requests (HMAC).

### 150+ organizations by April 2026

Enterprise adoption drove A2A scale. Google Cloud shipped Vertex AI Agent Builder A2A support; Microsoft Agent Framework supports it; most major frameworks ship A2A adapters.

### Where A2A wins

- Cross-organization calls, heterogeneous frameworks, typed artifacts, long-running tasks.

### Where A2A struggles

- Latency-sensitive micro-calls, tight-coupled in-process agents, small teams.

## Build It

`code/main.py` implements an A2A-minimal server and client using `http.server` and JSON. The server exposes `/.well-known/agent.json`, accepts `POST /tasks`, manages task state, and returns artifacts on `GET /tasks/{id}`. The client fetches the Agent Card, submits a task, polls until completion, and reads the artifact.

```
python3 code/main.py
```

## Ship It

Checklist:
- **Pin the spec version.** A2A is still evolving.
- **Idempotent task creation.** Duplicate submissions should produce one task.
- **Artifact schemas.** Declare what shapes the agent returns.
- **Rate limits + auth.** A2A is public-facing.
- **Dead-letter for failed tasks.** Inspect patterns over time.

## Exercises

1. Run `code/main.py`. Confirm the client discovers the server and receives the correct artifact.
2. Add a second skill to the server (e.g., "summarize"). Update the Agent Card.
3. Implement an SSE streaming endpoint: `/tasks/{id}/events` that emits state changes.
4. Read the A2A spec. Identify three things the spec mandates that this demo does not implement.
5. Compare A2A (Agent Card discovery) to MCP (server-side capability listing via `listTools`).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-agent" | Peer protocol for agents to call other agents across systems. |
| Agent Card | "The agent's business card" | JSON at `/.well-known/agent.json`. |
| Task | "The unit of work" | Async stateful object with a lifecycle. |
| Artifact | "The result" | Typed output: text, structured JSON, image, video, audio. |
| Opaque lifecycle | "How it's solved is the agent's business" | Client sees state transitions; server is free to choose framework. |
| Discovery | "Finding the agent" | `GET /.well-known/agent.json` returns the card. |
| MCP vs A2A | "Tools vs peers" | MCP: vertical agent ↔ tool. A2A: horizontal agent ↔ agent. |

## Further Reading

- [A2A specification](https://a2a-protocol.org/latest/specification/)
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [A2A GitHub repo](https://github.com/a2aproject/A2A)
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/12-a2a-protocol)
