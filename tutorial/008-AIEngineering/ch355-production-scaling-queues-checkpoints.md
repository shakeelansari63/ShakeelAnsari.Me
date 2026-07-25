# Production Scaling — Queues, Checkpoints, Durability

> Scaling multi-agent systems to thousands of concurrent runs requires **durable execution**. LangGraph's runtime writes a checkpoint after each super-step keyed by `thread_id`. **MegaAgent** ran a per-agent producer-consumer queue with three states. **Fiber/async** beats thread-per-job for LLM streaming. Counterpoint: Ashpreet Bedi's "Scaling Agentic Software" argues for **FastAPI + Postgres + nothing else** until load proves otherwise.

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

A prototype multi-agent system works on one laptop with three agents in an in-memory event loop. You move to production: agents run for hours, worker processes crash, peak load is 10x average, you need exactly-once semantics for charging.

The in-memory event loop does none of these. You need a durable execution layer underneath.

## Concept

### Durable execution, the pattern

A durable-execution engine persists the full program state after each "super-step." On crash:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Requirements: serializable state, deterministic resume, idempotent side effects.

### LangGraph's runtime

Each agent has a `thread_id`; state is a typed dict; each super-step writes a row to the checkpoints table. On resume, the runtime replays from the last checkpoint. Agents can `interrupt()` waiting for human input.

### MegaAgent's per-agent queue

arXiv:2408.09955: thousands of concurrent agents. Each agent has state ∈ {Idle, Processing, Response}, an in_queue, and an out_queue. Two-layer coordination: intra-group chat + inter-group admin chat.

### Async vs thread-per-job

LLM calls are I/O-bound. Threads cost ~1MB RAM each — at 10,000 concurrent calls, that is 10GB for stacks. Fibers (Python `asyncio`) cooperatively yield on I/O.

### Bedi's counterpoint

FastAPI + Postgres. Each agent run is a row; state updated in-place with optimistic concurrency. Background jobs via `pg_notify`. For loads under ~100 concurrent agent-runs, this is often all you need.

### Exactly-once semantics

- Dedup key per run.
- Outbox pattern: side effects write to a table first, then a separate process executes them.
- Compensating transactions.

### Rainbow deployment

Multiple versions of the agent runtime run concurrently so long-running agents do not have to be killed on every code deploy.

## Build It

`code/main.py` implements:
- `CheckpointStore` — SQLite-backed checkpoint log with thread-id keys.
- `run_with_checkpoint(agent, thread_id)` — simulates a crash mid-run; a second worker resumes from last checkpoint.
- `AgentQueue` — per-agent Idle / Processing / Response state machine.
- `demo_async_vs_threads()` — runs 500 concurrent simulated "LLM calls" via asyncio and via threads.

```
python3 code/main.py
```

Expected output: checkpoint resume succeeds after simulated crash; async handles 500 concurrent calls in < 1s.

## Ship It

- **Start simple (Bedi's rule).** FastAPI + Postgres until you measure it failing.
- **Instrument everything before optimizing.**
- **Outbox pattern for side effects.**
- **Rainbow deploys.** Never kill in-flight agent runs during deploys.
- **Adopt durable-execution engines when** you hit specific problems.
- **Async for the I/O layer.** Threads only for CPU-bound post-processing.

## Exercises

1. Run `code/main.py`. Confirm checkpoint resume works; measure async vs thread concurrency difference.
2. Implement an outbox table: every tool call writes to outbox first.
3. Simulate a rainbow deploy: two concurrent runtime versions.
4. Read LangGraph's runtime doc. Identify which features would take the longest to replicate in FastAPI + Postgres.
5. Read MegaAgent Section 3. Sketch the two-layer coordination mapped to a message queue.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Durable execution | "Persist the program state" | Engine writes state after each super-step. |
| Super-step | "Transactional boundary" | Unit of work between checkpoints. |
| thread_id | "Agent run identifier" | Key that binds checkpoints and resume logic. |
| Idempotency | "Safe to retry" | Repeating a side effect produces the same result. |
| Outbox pattern | "Decouple side effects" | Write intent to a table; a separate executor performs. |
| At-least-once delivery | "Possible duplicates" | Message queue semantics; dedup key makes consumer effective-once. |
| Rainbow deploy | "Overlapping versions" | Multiple runtime versions concurrent during long-running workloads. |
| Async fiber | "Cooperative yielding" | User-mode concurrency; cheap compared to threads. |
| Checkpoint | "State snapshot" | Serialized state at a super-step boundary. |

## Further Reading

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents)
- [MegaAgent](https://arxiv.org/abs/2408.09955)
- [Matrix](https://arxiv.org/abs/2511.21686)
- [Temporal docs](https://docs.temporal.io/)
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints)
