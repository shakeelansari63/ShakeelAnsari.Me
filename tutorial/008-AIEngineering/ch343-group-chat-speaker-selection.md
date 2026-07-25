# Group Chat and Speaker Selection

> AutoGen GroupChat and AG2 GroupChat share one conversation across N agents; a selector function (LLM, round-robin, or custom) picks who speaks next. This is the archetype of emergent multi-agent conversation — agents do not know their role in a static graph, they just react to the shared pool.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Static graphs (LangGraph) are great when the workflow is known. Real conversations are not static: sometimes the coder asks the reviewer, sometimes the researcher, sometimes the writer. Hardcoding every possible handoff produces an edge explosion. You want agents reacting to a shared pool, with some function deciding who talks next.

## Concept

### The shape

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Every agent sees every message. A selector function is invoked at each turn to pick who speaks next.

### The three selector flavors

**Round-robin.** Fixed cycle. Deterministic. Scales linearly in N but ignores context.

**LLM-selected.** A call to an LLM that reads the recent pool and returns the best next speaker. Context-aware but slow.

**Custom.** A Python function with whatever logic you want.

### The ConversableAgent API

```
agent = ConversableAgent(name="coder", system_message="You write Python.", llm_config={...})
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` holds the selector. When an agent completes a turn, the manager calls the selector, which returns the next agent.

### Termination

Three common patterns: max rounds, "TERMINATE" token, goal-reached check.

### The AutoGen → AG2 split and Microsoft Agent Framework merge

In early 2025, Microsoft began a major rewrite of AutoGen (v0.4) around an event-driven actor model. The community forked AutoGen v0.2's GroupChat semantics as AG2. In February 2026, Microsoft announced AutoGen would go to maintenance mode, merging into Microsoft Agent Framework. AG2 is the preferred upstream for v0.2-compatible code.

### When GroupChat fits

- **Emergent conversations.** You do not want to pre-wire every possible next-speaker.
- **Role-mixing tasks.** Coder asks researcher, researcher asks archivist, archivist asks coder back.
- **Exploratory problem-solving.** Think "brainstorm meeting," not "assembly line."

### When it fails

- **Strict determinism.** The LLM selector can be inconsistent.
- **Sycophancy cascades.** Agents defer to whoever spoke most confidently.
- **Context bloat.** Every agent reads every message.
- **Hot speakers.** One agent dominates the conversation.

## Build It

`code/main.py` implements a GroupChat from scratch in stdlib. Three agents (coder, reviewer, manager), round-robin and LLM-selected variants, and a termination on a `TERMINATE` token.

```
python3 code/main.py
```

## Ship It

Checklist:
- **Max rounds cap.** Always. 10-20 for typical tasks.
- **Speaker-balance metric.** Track turns per agent; alert when imbalance exceeds a threshold.
- **Termination token.** `TERMINATE` or a dedicated verifier agent.
- **Projection or scoped memory.** After ~10 messages, consider giving each agent only a scoped view.
- **Selector logging.** For LLM-selected variants, log both the selector's input and its choice.

## Exercises

1. Run `code/main.py`. Compare the conversation under round-robin vs LLM-selected.
2. Add a "max-speaks-per-agent" rule in the selector.
3. Implement a goal-reached termination: stop when the reviewer returns "approved."
4. Read the AutoGen stable docs on GroupChat. Identify the default selector used by `GroupChatManager`.
5. Read the AG2 repo and compare its v0.2 GroupChat to the v0.4 event-driven version.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GroupChat | "Agents in one chat room" | Shared message pool + selector function. |
| Speaker selection | "Who talks next" | The function that picks the next agent. |
| GroupChatManager | "The meeting host" | AutoGen component that owns the selector. |
| ConversableAgent | "The base agent" | AutoGen base class. |
| Termination token | "The 'stop' word" | Sentinel string that ends the chat. |
| Hot speaker | "One agent dominates" | Failure mode where the selector keeps picking the same agent. |
| Context bloat | "Pool grows unbounded" | Each agent reads every prior message. |
| Projection | "Scoped view" | Role-specific view into the shared pool. |

## Further Reading

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)
- [AG2 repo](https://github.com/ag2ai/ag2)
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/)
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection)
