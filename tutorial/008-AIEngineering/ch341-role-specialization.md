# Role Specialization — Planner, Critic, Executor, Verifier

> The most common multi-agent decomposition in 2026: one agent plans, one executes, one critiques or verifies. MetaGPT formalizes this as SOPs encoded into role prompts. ChatDev chains designer, programmer, reviewer, tester through a "chat chain." The verifier is load-bearing: Cemri et al. (MAST) show every multi-agent failure can be traced to missing or broken verification.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 minutes

## Problem

Generic multi-agent systems produce generic output. Three coders in a group chat write three flavors of the same mediocre code. The fix is not more agents — it is *different* agents. Assign distinct roles. Give the critic tools the planner does not have. Give the verifier an objective test suite.

## Concept

### The four canonical roles

**Planner.** Reads the goal, produces a step list or a spec. Tools: knowledge retrieval, docs. Output: structured plan.

**Executor.** Reads one plan step at a time, produces the artifact. Tools: the actual work tools. Output: the artifact.

**Critic.** Reads the executor's output against the planner's intent. Tools: read-only access. Output: accept/reject with reasons.

**Verifier.** Reads the artifact and runs a deterministic check. Tools: test runner, type checker, schema validator. Output: pass/fail with evidence.

Critic is subjective, opinionated, often LLM-based. Verifier is objective, deterministic, often code-based. They are not the same role.

### MetaGPT's SOP pattern

MetaGPT (arXiv:2308.00352) encodes software engineering SOPs as role prompts:
- **Product Manager** writes the PRD.
- **Architect** produces the system design.
- **Project Manager** splits tasks.
- **Engineer** implements.
- **QA Engineer** runs tests.

Each role has a strict input/output schema. The `Code = SOP(Team)` formulation — deterministic SOPs turn a team of LLMs into a predictable pipeline.

### ChatDev's communicative dehallucination

ChatDev adds: when an executor needs a specific detail that was not in the plan, it explicitly asks the designer before continuing. This prevents the classic LLM failure of plausibly inventing the detail.

### Why verifier matters most

Cemri et al. (MAST) traced 1642 multi-agent execution failures. 21.3% were verification gaps — the system shipped an answer no one had checked. PwC reported (CrewAI deployments, 2025) that adding a structured validation loop moved accuracy from 10% to 70%.

### Critic vs verifier

- A critic is an LLM reviewing an artifact for quality. Subjective. Can be fooled by plausible prose.
- A verifier is a deterministic program running on the artifact. Objective. Gives pass/fail with evidence.

Use both.

### The anti-pattern

Every role in your system is an LLM and every role's output is "looks good to me." Classic MAST failure mode. Add at least one verifier whose pass/fail is decided by code, not by an LLM.

## Build It

`code/main.py` implements a 4-role pipeline building a simple Python function:
- **Planner** produces a spec.
- **Executor** generates a code string.
- **Critic** (LLM-simulated) flags obvious issues.
- **Verifier** runs the generated code in a sandbox (`exec`) against a test case.

Demo runs twice: once where the executor produces correct code (critic + verifier both pass), once where the executor produces off-spec code (critic misses the bug because it looks plausible, verifier catches it because the test fails).

```
python3 code/main.py
```

## Ship It

Checklist:
- **At least one deterministic verifier.** Never all-LLM.
- **Explicit I/O schema per role.** The planner returns a spec, not prose.
- **Communicative dehallucination.** Executor must ask the planner when info is missing.
- **Critic/verifier ordering.** Run critic first (cheap), verifier second (slow).
- **Loop budget.** Max 2 critic-executor revision rounds before escalating to human.

## Exercises

1. Run `code/main.py` and observe how the verifier catches the bug the critic missed. Add a static-analysis check as an additional verifier.
2. Add a 5th role: "requirements analyst" that translates user wish into planner-ready spec.
3. Read MetaGPT Section 3 ("Agents"). List the input/output schema of each of MetaGPT's 5 roles.
4. Read ChatDev's chat-chain diagram (arXiv:2307.07924 Figure 3). Identify where communicative dehallucination breaks a loop that would otherwise be infinite.
5. PwC's 7x accuracy gain came from verification loops. Hypothesize three tasks where adding a verifier would not help.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Role specialization | "Different agents, different jobs" | Distinct system prompts tuned for planner/executor/critic/verifier roles. |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT's framing: strict I/O schemas per role. |
| Communicative dehallucination | "Ask before inventing" | ChatDev pattern: executor asks planner when a detail is missing. |
| Critic | "LLM reviewer" | Subjective, opinionated reviewer. |
| Verifier | "Deterministic check" | Code-based pass/fail. Test runner, type checker. |
| Verification gap | "No one checked" | 21.3% of MAST failures. |
| Revision loop | "Critic sends it back" | Critic rejection triggers executor re-run with feedback. |
| All-LLM anti-pattern | "Looks good to me" | Every role is an LLM, no deterministic check. |

## Further Reading

- [Hong et al. — MetaGPT](https://arxiv.org/abs/2308.00352)
- [Qian et al. — ChatDev](https://arxiv.org/abs/2307.07924)
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/08-role-specialization)
