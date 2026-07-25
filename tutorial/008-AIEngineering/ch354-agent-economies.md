# Agent Economies, Token Incentives, Reputation

> Long-horizon autonomous agents need economic agency. The emerging **5-layer stack** is: **DePIN** (physical compute) → **Identity** (W3C DIDs + reputation capital) → **Cognition** (RAG + MCP) → **Settlement** (account abstraction) → **Governance** (Agentic DAOs). Production agent-incentive networks include Bittensor, Fetch.ai / ASI Alliance, and Gonka.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Multi-agent systems get complicated when agents produce value jointly but need to be rewarded individually. Classical mechanisms — equal split, last-contributor-takes-all — are unfair or gameable.

Beyond credit attribution, the field has turned to actual economic agents: Bittensor TAO rewards mining compute, Fetch.ai/ASI rewards ASI-1 Mini LLM usage with FET tokens, Gonka reallocates transformer proof-of-work toward productive AI tasks.

## Concept

### The 5-layer agent-economy stack

1. **DePIN (physical compute).** Decentralized infrastructure that rents GPU, storage, bandwidth.
2. **Identity.** W3C DIDs give each agent a durable ID. Reputation accrues to the DID.
3. **Cognition.** The agent's reasoning loop: LLM + RAG + MCP.
4. **Settlement.** Account abstraction (ERC-4337) lets agents pay gas from their own balances.
5. **Governance.** Agentic DAOs where humans and agents vote on protocol changes.

### Bittensor, Fetch.ai, Gonka — what runs

**Bittensor (TAO).** Subnets are specialized tasks. Miners submit model outputs. Validators rank them; stake-weighted scoring distributes TAO rewards. Pay for task-specific output quality, not compute used.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM runs on Fetch.ai's network; users pay FET tokens for inference.

**Gonka.** Transformer proof-of-work: the "work" is forward passes of a transformer.

### Shapley-value credit attribution

Three agents collaborate on a task. The output scores 0.8. Who contributed what?

Shapley value: the unique credit allocation satisfying four axioms (efficiency, symmetry, linearity, null). For agent `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

For N=3, 6 permutations. For N=10, 3.6M — so in practice you sample.

### Second-price auction for aggregation

N agents each propose a completion; each has a private value. The auctioneer picks the highest-value proposal and pays the *second-highest* value. Under monotone aggregation, this is truthful.

### Reputation capital

A DID-bound reputation score from confirmed contributions:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

### Where the economics falls apart

- Price oracle manipulation, Sybil attacks, verification cost, regulatory overhang.

## Build It

`code/main.py` implements:
- `shapley(value_fn, agents)` — exact Shapley computation by enumeration for small N.
- `second_price_auction(bids)` — truthful mechanism.
- `Reputation` — DID-bound reputation with exponential decay and slashing.
- Demo 1: three agents collaborate, exact Shapley attributes credit.
- Demo 2: five agents bid for a task slot; second-price auction.
- Demo 3: 100 rounds of task assignment with rep-weighted routing.

```
python3 code/main.py
```

## Ship It

- **Start with reputation, not tokens.**
- **Verify before you reward.** Self-reported quality accrues sybil games.
- **Shapley-sample, not Shapley-exact.** Sample 100-1000 orderings.
- **Cap decay factor and floor reputation.**
- **Audit mechanisms adversarially.**

## Exercises

1. Run `code/main.py`. Confirm Shapley values sum to total value.
2. Implement Shapley sampling (Monte Carlo over K orderings).
3. Implement a coalition-forming step before the auction.
4. Read the Google Research mechanism-design post. Identify one assumption that breaks truthfulness.
5. Read AAMAS 2025 decentralized LaMAS. Implement their Shapley step over 10 agents.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | Token-incentivized compute/storage/bandwidth. |
| DID | "Decentralized identifier" | W3C spec for portable IDs. |
| ERC-4337 | "Account abstraction" | Contract accounts that can sponsor gas. |
| Shapley value | "Fair credit attribution" | Unique allocation satisfying efficiency, symmetry, linearity, null. |
| Second-price auction | "Vickrey auction" | Truthful mechanism: winner pays second-highest bid. |
| Reputation capital | "Accumulated quality score" | DID-bound score from confirmed contributions. |
| Agentic DAO | "Agents + humans govern" | DAO with agent voters as first-class. |

## Further Reading

- [The Agent Economy](https://arxiv.org/abs/2602.14219)
- [Google Research — Mechanism design for LLMs](https://research.google/blog/mechanism-design-for-large-language-models/)
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf)
- [Bittensor TAO documentation](https://docs.bittensor.com/)
- [Fetch.ai / ASI Alliance](https://fetch.ai/)
- [W3C DIDs spec](https://www.w3.org/TR/did-core/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/21-agent-economies)
