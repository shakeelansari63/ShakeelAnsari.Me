# Hierarchical Architecture and Its Failure Mode

> Hierarchical is supervisor nested. Manager agents over sub-managers over workers. CrewAI `Process.hierarchical` is the textbook version: a `manager_llm` dynamically delegates tasks and validates outputs. It is the natural pattern when the task is a real org chart. It is also the pattern most likely to collapse into managerial looping.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 minutes

## Problem

Once the supervisor pattern clicks, the natural next step is "what if the workers are themselves supervisors?" Teams have sub-teams; companies have departments of departments. Hierarchical architectures mirror that.

The issue: LLM managers are not the same as human managers. A human manager has stable priors about what their reports know. An LLM manager re-reasons the org every turn from whatever is in its context. Tiny drift in that context, and the whole tree misallocates work.

## Concept

### The shape

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Every internal node plans, delegates, and synthesizes. Only leaves do work.

### Where it shines

- **Clear org mapping.** If the real task is departmental, the hierarchy is explicit.
- **Local summarization.** Each sub-manager synthesizes its team's output before the top manager sees it.

### Where it breaks

1. **Task assignment error.** The manager hallucinates a decomposition and delegates to the wrong sub-manager. The error only surfaces at the top synthesis.
2. **Output misinterpretation.** Sub-manager returns "unable to verify claim X." Top manager summarizes as "claim X not confirmed." Meaning drifts at every level.
3. **Consensus loops.** Two sub-managers disagree; top manager asks them to reconcile; they re-delegate down; workers re-run; loop.

### The deciding question

Sequential (linear pipeline) vs hierarchical: does your task actually have independent sub-teams, or is it one linear flow pretending to be a tree?

### CrewAI's implementation

`Process.hierarchical` wires a manager LLM over specialist crews. The manager receives the top-level task, assigns subtasks to crews, evaluates crew outputs, and decides whether to accept, re-delegate, or iterate.

### LangGraph's implementation

LangGraph uses nested `create_supervisor` calls. The inner supervisor has its own graph; the outer supervisor treats the inner graph as an opaque node.

## Build It

`code/main.py` runs a 3-level hierarchy:
- top manager: splits a task into "engineering" and "legal" branches,
- engineering sub-manager: splits into "frontend" and "backend" workers,
- legal sub-manager: one worker.

Demo contrasts happy path (everyone agrees) against a **perturbed path** where the top manager's decomposition mislabels "legal" as "finance" and watches the error cascade.

```
python3 code/main.py
```

## Ship It

If you ship hierarchical:
- **Cap tree depth at 2.** Three levels already hides most errors from observability.
- **Explicit reconciliation budget.** Set max rounds before the top manager must commit. Usually 2.
- **Provenance on every synthesis.** Each node's summary must cite which leaf outputs produced it.
- **Alert on decomposition drift.** Log the manager's decomposition per step; diff against the user query.

## Exercises

1. Run `code/main.py` and compare happy vs perturbed. How many levels of manager hand-off does it take before the top output fully diverges from the user's question?
2. Add a third level (top → sub → sub-sub → worker). Measure how often the perturbed path corrects itself vs fully diverges as depth grows.
3. Implement a "canary" worker at each sub-manager that is always asked the original user question unchanged.
4. Read CrewAI's `Process.hierarchical` docs. Identify one concrete guardrail CrewAI applies.
5. Compare nested LangGraph supervisors to CrewAI hierarchical. Which makes reconciliation loops cheaper to detect?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hierarchical | "Org chart pattern" | Supervisors over supervisors; only leaves do work. |
| Manager LLM | "The boss" | The LLM that decomposes, assigns, and validates at an internal node. |
| Decomposition drift | "The boss lost the plot" | Top manager's split no longer covers the original question. |
| Reconciliation loop | "Endless meetings" | Sub-managers disagree; top re-delegates; workers re-run; loop until budget exhausted. |
| Depth-2 ceiling | "Don't go deeper than 2 levels" | Empirical guardrail: 3+ levels collapses observability. |
| Canary question | "Ground truth at every level" | A worker that is always asked the original query unchanged. |
| Provenance chain | "Who said what" | Trace from each synthesis back to the leaf outputs that produced it. |

## Further Reading

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction)
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor)
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/06-hierarchical-architecture)
