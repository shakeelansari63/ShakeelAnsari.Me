# Consensus and Byzantine Fault Tolerance for Agents

> Classical distributed-systems BFT meets stochastic LLMs. In 2025-2026 three research directions emerged: **CP-WBFT** (arXiv:2511.10400) weighs each vote by a confidence probe; **DecentLLMs** (arXiv:2507.14928) goes leaderless with geometric-median aggregation; **WBFT** (arXiv:2505.05103) combines weighted voting with Hierarchical Structure Clustering.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

You have N LLM agents each producing an answer. They disagree. Majority vote picks the wrong one because two agents are correlated (same base model, same training data, same failure modes).

Now add a deceptive agent: it lies on purpose. Or a sycophantic agent: it agrees with whoever spoke last. Classical BFT handles arbitrary bit-flipping but not "three honest agents share a hallucination because they share training data."

## Concept

### What classical BFT gives you

Practical Byzantine Fault Tolerance (Castro & Liskov, OSDI 1999) tolerates `f < n/3` Byzantine nodes. Three phases (pre-prepare, prepare, commit). The guarantees assume independent faults, truly honest honest nodes, and a ground-truth answer. LLM agents violate all three.

### The three LLM-specific attacks

**Byzantine lie.** One agent outputs a deliberately wrong answer. Classical BFT handles this if `f < n/3`.

**Sycophantic conformity.** One agent reads others' answers before voting and aligns with whoever spoke last. Classical BFT does not prevent this.

**Correlated-error monoculture.** Three agents share a base model. They hallucinate the same wrong answer. Classical BFT does not help.

### The 2025-2026 responses

**CP-WBFT** — Confidence-Probed Weighted BFT. Each voter attaches a confidence probe. Vote weights scale with confidence. Mitigates sycophantic conformity.

**DecentLLMs** — Leaderless. Workers propose in parallel, evaluators score proposals, final answer is the geometric median. Robust when `f < n/2`. Mitigates Byzantine lies and correlated errors.

**WBFT** — Weighted BFT with Hierarchical Structure Clustering. Cluster agents into Core and Edge; Core must achieve consensus first. Mitigates scalability issues.

### "Can AI Agents Agree?" (arXiv:2603.01213)

Even with no adversaries, LLM agents disagree on scalar questions at rates above 30% on many benchmarks. A single deceptive agent can pull the consensus 40+ percentage points off the honest baseline.

### The core protocol, stripped down

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C: w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

## Build It

`code/main.py` implements:
- `MajorityVote` — classical plurality.
- `CPWBFT` — confidence-weighted voting with semantic clustering.
- `DecentLLMs` — geometric-median aggregation on scored proposals.
- `Scenario` — runs each aggregator under three attack patterns (byzantine, sycophancy, monoculture).

```
python3 code/main.py
```

Expected output: a table of (attack, aggregator) -> final answer. Plurality fails the monoculture case. CPWBFT's confidence weighting mitigates sycophancy.

## Ship It

- **Attack-test with at least the three patterns** above.
- **Log every minority cluster** with provenance.
- **Enforce bounded rounds.** No "keep debating until agreement."
- **Separate agreement from correctness.** Verifier is independent of the ensemble.
- **Monitor the agreement rate.** Sharp rise = conformity bias; sharp fall = model drift.

## Exercises

1. Run `code/main.py`. Confirm plurality fails the monoculture attack but CPWBFT partially mitigates it.
2. Add a fourth attack pattern: **silent abstention** — one agent refuses to answer.
3. Swap semantic clustering from string canonicalization to embedding-similarity.
4. Read CP-WBFT (arXiv:2511.10400). Implement the confidence-probe calibration step.
5. Read "Can AI Agents Agree?" (arXiv:2603.01213). Reproduce a simplified scalar-agreement experiment.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| BFT | "Byzantine fault tolerance" | Castro-Liskov 1999 protocol for consensus with `f < n/3` arbitrary faults. |
| Byzantine | "Any bad behavior" | A node that can lie, drop messages, fail silently. |
| Confidence probe | "How sure are you?" | Self-reported or calibrator-predicted probability attached to a vote. |
| Semantic clustering | "Same answer, different words" | Grouping equivalent answers before counting votes. |
| Geometric median | "Robust center" | Robust to outliers, unlike the mean. |
| Monoculture | "Same model, same failures" | Correlated errors when agents share training data or base model. |
| Sycophantic conformity | "Agreeing with the loud voice" | An agent's vote biases toward whoever spoke first/loudest. |
| Core/Edge | "Hierarchical BFT" | WBFT split: small Core consensus first, Edge nodes follow. |

## Further Reading

- [Castro & Liskov — Practical Byzantine Fault Tolerance](https://pmg.csail.mit.edu/papers/osdi99.pdf)
- [CP-WBFT](https://arxiv.org/abs/2511.10400)
- [DecentLLMs](https://arxiv.org/abs/2507.14928)
- [WBFT](https://arxiv.org/abs/2505.05103)
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/14-consensus-and-bft)
