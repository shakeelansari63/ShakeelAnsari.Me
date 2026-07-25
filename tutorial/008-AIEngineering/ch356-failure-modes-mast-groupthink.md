# Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors

> The reference taxonomy for 2026 is **MAST** (Cemri et al., NeurIPS 2025), derived from 1642 execution traces showing **41–86.7% failure rate**. Three root categories: **Specification Problems** (41.77%), **Coordination Failures** (36.94%), **Verification Gaps** (21.30%). The **Groupthink** family adds: monoculture collapse, conformity bias, deficient theory of mind, mixed-motive dynamics, cascading reliability failures.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 minutes

## Problem

Multi-agent systems fail 41-86.7% of the time on real tasks. That is not debuggable by "just add more agents." The failures have structural causes. The MAST taxonomy gives you the categories.

## Concept

### MAST categories

**Specification Problems (41.77% of failures).** The agent's task was not defined tightly enough. Examples: role ambiguity, task underspecified, success criteria implicit.

Mitigations: explicit role contracts, acceptance tests per task, pre-flight spec check.

**Coordination Failures (36.94%).** Communication or state breakdowns. Examples: two agents update shared state without synchronization, message lost, state drift.

Mitigations: versioned shared state with optimistic concurrency, explicit acknowledgment for critical messages, periodic state-sync checkpoints.

**Verification Gaps (21.30%).** No independent check on outputs. Examples: one agent claims success, chain of agents each trusts the prior's output.

Mitigations: independent verifier agent, explicit handoff contract, outcome logging.

### Groupthink family

Five related failures when agents homogenize or mimic each other:

**Monoculture collapse.** Same base model → correlated errors.

**Conformity bias.** Agents adjust toward the loudest or most-confident peer.

**Deficient ToM.** Agents fail to model each other's beliefs.

**Mixed-motive dynamics.** Agents with partially-aligned incentives drift toward compromise-middle.

**Cascading reliability failures.** One component's error triggers errors in dependent components.

### Cascading example — the retry storm

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load → starts timing out
   ↓
every order retries inventory check → 10x normal load → cluster goes down
```

The fix: **circuit breakers**. When downstream error rate exceeds threshold, short-circuit with cached or default results.

### Memory poisoning (revisited)

One agent's hallucination becomes shared-memory fact. Gradual accuracy decay is the symptom. Mitigation: append-only log, provenance, unwritable verifier.

### STRATUS — specialized agents for failure detection

STRATUS (NeurIPS 2025) reports 1.5x mitigation-success improvement with:
- **Detection agent.** Watches for symptom patterns.
- **Diagnosis agent.** Infers likely root cause from the MAST taxonomy.
- **Validation agent.** Checks that symptoms clear after mitigation.

### The failure-mode audit

1. Trace sample — collect ~1000 real execution traces.
2. Categorize — map to MAST + Groupthink categories.
3. Compute failure-by-category rate.
4. Rank mitigations.
5. Pick 2-3 mitigations; implement; re-audit next quarter.

### Slow failures

Some failures are slow (memory poisoning, monoculture drift, role ambiguity) and expensive to detect. Instrument slow-failure proxies: agreement rate, retry rate, output-length distribution.

## Build It

`code/main.py` implements:
- `FailureTaxonomy` — categorizes simulated incidents.
- `CircuitBreaker` — classic pattern; opens when error rate exceeds threshold.
- `RetryStormSimulator` — shows the cascading failure.
- `DetectionAgent` — scripted STRATUS-style symptom matcher.

```
python3 code/main.py
```

Expected output: retry storm with no circuit breaker blows up; with circuit breaker, capped at threshold; detection agent flags the pattern.

## Ship It

- **MAST audit per quarter.** Not annual.
- **Circuit breakers everywhere.** Default open threshold at 5-10% error rate.
- **Golden datasets.** Small, high-quality, hand-audited.
- **STRATUS trio.** Detection + Diagnosis + Validation agents monitoring production.
- **Failure budget.** Explicit SLO for failure rate by category.

## Exercises

1. Run `code/main.py`. Confirm the circuit breaker caps the retry storm.
2. Implement a slow-failure proxy: agreement rate across 3 parallel agents.
3. Read Cemri et al. Pick one of their 7 MAS systems and map its top 3 failure categories.
4. Read the Groupthink paper (arXiv:2508.05687). Identify which of the five patterns is hardest to detect.
5. Design a STRATUS-style detection-diagnosis-validation trio for a system you know.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MAST | "The 2026 taxonomy" | Cemri 2025; 3 root categories + 14 sub-types. |
| Specification Problem | "Role ambiguity" | Task or role under-defined. |
| Coordination Failure | "State drift" | Communication or sync breakdown. |
| Verification Gap | "No one checked" | Outputs accepted without independent validation. |
| Groupthink family | "Homogeneity failures" | Monoculture, conformity, deficient ToM, mixed-motive, cascading. |
| Monoculture collapse | "Same model, same hallucinations" | Correlated errors from shared base model. |
| Retry storm | "Cascading error amplification" | One failure triggers retries which amplify load downstream. |
| Circuit breaker | "Fail fast on error rate" | Open when error rate exceeds threshold. |
| STRATUS | "Incident response trio" | Detection + diagnosis + validation agents. |
| Memory poisoning | "Hallucinations propagate" | Shared-memory fact tainted. |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687)
- [STRATUS — NeurIPS 2025](https://neurips.cc/)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/)
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink)
