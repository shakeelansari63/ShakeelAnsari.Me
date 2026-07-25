# Supervisor / Orchestrator-Worker Pattern

> One lead agent plans and delegates; specialized workers execute in parallel contexts and report back. This is the pattern behind Anthropic's Research system (Claude Opus 4 as lead, Sonnet 4 as subagents), measured at +90.2% over single-agent Opus 4 on internal research evals.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Research is the prototypical task that single-agent systems fail. You ask "what changed in multi-agent systems between 2023 and 2026?" A single agent reads five papers sequentially, fills half its context with their text, and then has to reason about all of them together. It forgets the first paper by the time it reaches the fifth. It cannot parallelize.

The supervisor pattern fixes this: one lead agent plans the search, delegates each sub-question to a worker, and synthesizes. Each worker gets its own 200k-token window for a narrow question. The lead never sees the raw papers — only the worker summaries.

Anthropic's production Research system reports +90.2% on internal research evals vs a single Opus 4. The same post notes that 80% of the BrowseComp variance is explained by *token usage alone*. Fresh context per subagent is the main mechanism.

## Concept

### The pattern

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

The lead never reads the raw materials. The workers never see each other's work until the lead synthesizes. Each arrow is a handoff with a narrow artifact.

### Why it wins

1. **Fresh context per subagent.** A worker exploring "FIPA-ACL heritage" does not carry the 40k tokens the lead spent planning.
2. **Specialization via prompt.** The lead's prompt is "decompose and synthesize," not "research." Each worker's prompt is narrow.
3. **Parallelism.** Workers run concurrently. Wall-clock time is roughly `max(worker_times) + plan + synthesis`, not `sum(worker_times)`.

### Engineering lessons (Anthropic 2025)

- **Scale effort to query complexity.** Simple queries: one agent, 3-10 tool calls. Complex queries: 10+ agents.
- **Broad then narrow.** Decompose into broad sub-questions first, then spawn more workers per sub-question if needed.
- **Rainbow deployments.** Agents are long-running and stateful. Traditional blue-green does not work. Rainbow: gradual rollout of new versions while old ones drain.
- **Token usage dominates.** Multi-agent is ~15x the tokens of single-agent. Only run it when the task value justifies the cost.

### The LangGraph turn

LangGraph originally shipped a `langgraph-supervisor` library with `create_supervisor`. In 2025 LangChain moved the recommendation to implementing the supervisor pattern via tool-calling directly, because tool calls give more control over what the supervisor sees.

### The failure modes

- **Lead hallucinates the plan.** If the lead generates sub-questions that do not decompose the real question, workers do precise research on the wrong target.
- **Workers over-explore.** Without explicit scope boundaries, workers drift beyond their assigned sub-question.
- **Synthesis conflicts.** Two workers return contradictory facts. The lead must either re-ask or note the disagreement explicitly.

### When supervisor is wrong

- **Sequential tasks.** If step 2 needs step 1's output, parallelism buys nothing. Use a pipeline.
- **Simple queries.** Single-agent handles them faster and cheaper.
- **Strict determinism.** Static graphs are better when audit/replay matter more than adaptability.

## Build It

`code/main.py` implements a supervisor of three parallel workers using `threading`. The lead decomposes a query into sub-questions, workers run concurrently on each sub-question, and the lead synthesizes. No real LLMs — the workers are scripted to simulate fetch-and-summarize.

Key structure:
- `Lead.plan(query)` splits a query into 3 sub-questions.
- `Worker.run(sub_q)` returns a fake summary.
- `Lead.run(query)` kicks off workers in threads, joins, and synthesizes.

```
python3 code/main.py
```

Output shows the plan, the parallel worker traces with start/end timestamps, and the final synthesis. Three 0.3-second workers run in ~0.35 seconds, not 0.9.

## Ship It

Checklist before deploying a supervisor pattern:
- **Model pairing.** Lead on a reasoning-tier model (Opus class). Workers on a faster, cheaper model (Sonnet, `o4-mini`).
- **Worker timeout.** Any worker that exceeds 2x median runtime gets killed.
- **Token cap per worker.** Hard limit prevents a runaway worker from blowing the budget.
- **Observability.** Trace the lead's plan, each worker's tool calls, and the synthesis.
- **Rainbow rollout.** Stateful long-running agents need gradual version transition, not hot swap.

## Exercises

1. Run `code/main.py`, then modify the lead to spawn 5 workers instead of 3. Observe the wall-clock effect.
2. Implement a worker timeout: kill any worker that runs longer than 0.5 seconds and have the lead synthesize the remaining results.
3. Add a conflict-detection step to the lead's synthesis: if two workers return contradictory answers, the lead notes the disagreement rather than picking one.
4. Read Anthropic's Research-system engineering post. List three practices that this toy demo would need to adopt to run in production.
5. Compare LangGraph's `create_supervisor` (legacy) vs the new tool-calling recommendation.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor | "Lead agent" | An orchestrator agent that plans, delegates, and synthesizes. |
| Worker | "Subagent" | A focused agent invoked by the supervisor with narrow scope. |
| Orchestrator-worker | "Supervisor pattern" | Same thing, different name. |
| Fresh context | "Clean window" | A worker's context starts from its system prompt and assigned question. |
| Rainbow deployment | "Gradual rollout" | Long-running stateful agents need versioned drain-and-replace. |
| Token dominance | "Context is the variable" | 80% of research-eval variance comes from total tokens used. |
| Scale effort | "Match agent count to complexity" | Lead estimates query difficulty, spawns workers accordingly. |
| Synthesis conflict | "Workers disagree" | Lead must surface disagreement, not silently pick one. |

## Further Reading

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor)
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern)
