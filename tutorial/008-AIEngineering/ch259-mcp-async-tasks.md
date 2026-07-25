# Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work

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
