# Parallel / Swarm / Networked Architectures

> Contrast with supervisor: no central decider. Agents read a shared event bus, pick up work asynchronously, write results back. The tradeoff is explicit: determinism and traceability for scalability. Swarm fits tasks with many independent sub-problems; it does not fit tasks that need a single coherent plan.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Supervisor scales to a few workers. What about hundreds? The supervisor itself becomes the bottleneck: every decision about who does what funnels through one agent. One slow plan step stalls the whole system.

Swarm architectures flip the design. Instead of a central planner dispatching work, workers pick work off a shared queue. No orchestrator; the system scales until the queue does.

## Concept

### The shape

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

No orchestrator. Each worker repeats: pull a task, process, write result.

### When swarm fits

- **Many independent tasks.** Scraping, transforming, classifying.
- **Variable-duration work.** If some tasks take 100ms and others take 10s, a swarm balances load automatically.
- **Throughput over determinism.** You care about total completion time, not strict ordering.

### When swarm fails

- **Ordered workflows.** If step 3 needs step 2's output, a swarm risks step 3 firing before step 2 is done.
- **Global-plan tasks.** Complex research questions benefit from a planner.
- **Debugging.** With no central log and asynchronous work, reproducing a bug is expensive.

### Matrix (arXiv:2511.21686)

Matrix takes swarm to its natural conclusion: both control flow and data flow are serialized messages on distributed queues. No central coordinator. Fault tolerance comes from message durability. Coordination becomes "what message topic does this agent subscribe to?" rather than "which agent does the supervisor pick next?"

### LangGraph's Swarm Architecture

LangGraph explicitly describes "Swarm Architecture" as one of the multi-agent patterns: agents are nodes, but edges form a directed graph with cycles and any node can be activated from the pool.

### Failure mode: starvation and hot-spotting

If all workers pull the fastest-available task, long-running tasks never get picked until they are the only ones left. Mitigations: priority queues with explicit aging, worker specialization, back-pressure.

## Build It

`code/main.py` implements a swarm of 4 worker threads pulling from a shared `queue.Queue`. Tasks have variable durations. The demo contrasts:
- **Sequential baseline:** one worker processes all tasks serially.
- **Fixed assignment:** each task pre-assigned to a specific worker.
- **Swarm:** workers pull from a shared queue.

Swarm balances load automatically; fixed assignment leaves fast workers idle when their assigned task is slow.

```
python3 code/main.py
```

## Ship It

Checklist:
- **Priority queue with aging.** Prevent long-task starvation.
- **Worker idempotency.** A task may be pulled more than once if a worker crashes mid-run.
- **Durable queue.** Use Kafka, Redis Streams, or a database-backed queue for production.
- **Observability per task.** Every task has a trace ID.
- **Back-pressure.** If the queue grows faster than workers drain it, slow the producer.

## Exercises

1. Run `code/main.py`. How much faster is swarm than sequential on the variable-duration workload?
2. Add a priority queue variant (use `queue.PriorityQueue`). Observe whether low-priority tasks ever starve.
3. Implement a hot-spot detector: log when any worker processes 3x more tasks than the slowest worker.
4. Read the Matrix paper (arXiv:2511.21686) abstract and Section 3. Identify one specific tradeoff Matrix accepts.
5. Convert the swarm demo to use a `queue.Queue` of (task_type, payload) tuples, with workers subscribing only to specific types.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Swarm architecture | "Decentralized agents" | Workers pull from shared queue; no central orchestrator. |
| Event bus | "Agents subscribe to topics" | Message broker that routes tasks to workers by type or content. |
| Starvation | "Task never runs" | Low-priority task never gets picked. |
| Hot-spotting | "One worker drowns" | Load imbalance where one worker gets most tasks. |
| Back-pressure | "Slow down the producer" | Mechanism that signals upstream to stop producing when the queue fills up. |
| Idempotent worker | "Safe to re-run" | A task processed twice produces the same result. |
| Durable queue | "Survives crashes" | Queue backed by disk or replicated storage. |
| Matrix framework | "Full message-passing swarm" | Both data and control flow are serialized messages on distributed queues. |

## Further Reading

- [LangGraph — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686)
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system)
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/09-parallel-swarm-networks)
