# Negotiation and Bargaining

> Agents negotiate resources, prices, task allocations, and terms. The 2026 benchmark set is clear: NegotiationArena shows LLMs can improve payoffs ~20% via persona manipulation; OG-Narrator pushed deal rate from 26.67% to 88.88%; chain-of-thought-concealing agents win by hiding reasoning from counterparts.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Two agents need to agree on a price. Left to themselves with pure language prompts, LLMs close deals at surprisingly low rates (~27% on tightly-parameterized bargains). Scale does not fix it: GPT-4 is not structurally better at bargaining than GPT-3.5.

The root issue: LLMs conflate two jobs — deciding the offer and narrating the offer. OG-Narrator separated these: a deterministic offer generator computes numeric moves; the LLM only narrates. Deal rate jumps to ~89%.

## Concept

### Contract Net, in one paragraph

Smith's 1980 Contract Net Protocol: a **manager** broadcasts a **call for proposals (cfp)**; **bidders** respond with **propose** messages containing their offers; the manager picks a winner and sends **accept-proposal** to the winner and **reject-proposal** to the losers.

### Why OG-Narrator wins

OG-Narrator decomposition:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

Deal rate jumps because prices stay in the bargaining zone, anchors are strategic, and the LLM does what it is good at: writing.

### NegotiationArena findings

- LLMs can improve payoffs ~20% by adopting personas ("I am desperate to sell this by Friday").
- Fair/cooperative agents are exploited by adversarial ones.
- Symmetric pair-ups converge to inequitable outcomes on about 40% of scenarios.

### Chain-of-thought concealment

Winners in the Large-Scale Autonomous Negotiation Competition (~180k negotiations) concealed their reasoning from counterparts. Engineering takeaway: separate private-scratchpad context from public-message context.

### Bhattacharya et al. 2025 — model rankings

On Harvard Negotiation Project metrics: Llama-3 most-effective, Claude-3 most-aggressive, GPT-4 fairest.

### The narration-vs-mechanism rule

> Let the LLM narrate. Do not let the LLM compute the offer.

## Build It

`code/main.py` implements:
- `ContractNetManager`, `ContractNetTask`, `Bid` — manager + bidders.
- `og_narrator_bargain(state, rng)` — OG-Narrator buyer: deterministic Zeuthen-style concession.
- `seller_response(state, rng)` — deterministic seller counter-offer policy.
- `naive_llm_bargain(state, rng)` — simulates an all-LLM bargainer.
- Measurement: deal rate over 1000 trials.

```
python3 code/main.py
```

Expected output: naive-LLM deal rate ~65-75%; OG-Narrator deal rate ~85-95%.

## Ship It

- **Separate scratchpad.** Private state never reaches the counterpart's context.
- **Deterministic offer generation.** Prices, quantities, ETAs: compute, do not prompt.
- **Validate all incoming offers** against a schema.
- **Bound rounds.** 3-5 rounds maximum; escalate to mediator on deadlock.
- **Measure deal rate and payoff variance** continuously.
- **Log all rejected proposals** with the deterministic rationale.

## Exercises

1. Run `code/main.py`. Confirm OG-Narrator beats naive-LLM on deal rate.
2. Implement persona-based payoff improvement — the buyer adopts a "desperate to buy this week" persona.
3. Implement chain-of-thought concealment with a private scratchpad.
4. Extend Contract Net to N-bidder auction with reserve price.
5. Read Bhattacharya et al. 2025. Implement two bargainers with different styles (aggressive vs fair).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Contract Net | "Task market" | Smith 1980, FIPA 1996. cfp + propose + accept/reject. |
| ZOPA | "Zone of possible agreement" | Overlap between buyer's max and seller's min. |
| BATNA | "Best alternative to a negotiated agreement" | Your fallback if this deal fails. |
| OG-Narrator | "Offer generator + narrator" | Decomposition: deterministic offer, LLM narration. |
| Zeuthen strategy | "Risk-minimizing concession" | Classical offer-generator that concedes based on risk limits. |
| CoT concealment | "Hide your reasoning" | Private scratchpads; public channel shows offer only. |
| Persona manipulation | "Emotional posturing" | ~20% payoff gain from desperation/urgency personas. |

## Further Reading

- [NegotiationArena](https://arxiv.org/abs/2402.05863)
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813)
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416)
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf)
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/16-negotiation-bargaining)
