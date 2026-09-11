# A2A, Blackboards, Consensus & Negotiation

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch345): A2A — The Agent-to-Agent Protocol

> Google announced A2A in April 2025; by April 2026 the spec is at https://a2a-protocol.org/latest/specification/ and 150+ organizations back it. A2A is the horizontal complement to MCP: where MCP is vertical (agent ↔ tools), A2A is peer-to-peer (agent ↔ agent).

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Your agent needs to call another agent on another system. How? You can expose an HTTP endpoint, define a bespoke JSON schema, and hope the other side speaks it. Every pair of agents becomes a custom integration.

A2A is the universal wire protocol for that call. Standard discovery, standard task model, standard transport, standard artifacts.

## Concept

### The four elements

**Agent Card.** A JSON document at `/.well-known/agent.json` describing the agent: name, skills, endpoints, supported modalities, auth requirements.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": { "tasks": "https://agent.example.com/tasks" },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task.** The unit of work. An async, stateful object with a lifecycle: `submitted → working → completed / failed / canceled`.

**Artifact.** The result type produced by a task. Text, structured JSON, image, video, audio.

**Opaque lifecycle.** A2A does not prescribe *how* the remote agent solves the task. The client sees state transitions and artifacts.

### The MCP/A2A split

- **MCP**: agent ↔ tool. The agent reads/writes via JSON-RPC to a tool server.
- **A2A**: agent ↔ agent. Peer protocol; both sides are agents with their own reasoning.

Production multi-agent systems use both.

### Discovery flow

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts
```

### Auth

A2A supports three common patterns: Bearer token (OAuth2), mTLS, signed requests (HMAC).

### 150+ organizations by April 2026

Enterprise adoption drove A2A scale. Google Cloud shipped Vertex AI Agent Builder A2A support; Microsoft Agent Framework supports it; most major frameworks ship A2A adapters.

### Where A2A wins

- Cross-organization calls, heterogeneous frameworks, typed artifacts, long-running tasks.

### Where A2A struggles

- Latency-sensitive micro-calls, tight-coupled in-process agents, small teams.

## Build It

`code/main.py` implements an A2A-minimal server and client using `http.server` and JSON. The server exposes `/.well-known/agent.json`, accepts `POST /tasks`, manages task state, and returns artifacts on `GET /tasks/{id}`. The client fetches the Agent Card, submits a task, polls until completion, and reads the artifact.

```
python3 code/main.py
```

## Ship It

Checklist:
- **Pin the spec version.** A2A is still evolving.
- **Idempotent task creation.** Duplicate submissions should produce one task.
- **Artifact schemas.** Declare what shapes the agent returns.
- **Rate limits + auth.** A2A is public-facing.
- **Dead-letter for failed tasks.** Inspect patterns over time.

## Exercises

1. Run `code/main.py`. Confirm the client discovers the server and receives the correct artifact.
2. Add a second skill to the server (e.g., "summarize"). Update the Agent Card.
3. Implement an SSE streaming endpoint: `/tasks/{id}/events` that emits state changes.
4. Read the A2A spec. Identify three things the spec mandates that this demo does not implement.
5. Compare A2A (Agent Card discovery) to MCP (server-side capability listing via `listTools`).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-agent" | Peer protocol for agents to call other agents across systems. |
| Agent Card | "The agent's business card" | JSON at `/.well-known/agent.json`. |
| Task | "The unit of work" | Async stateful object with a lifecycle. |
| Artifact | "The result" | Typed output: text, structured JSON, image, video, audio. |
| Opaque lifecycle | "How it's solved is the agent's business" | Client sees state transitions; server is free to choose framework. |
| Discovery | "Finding the agent" | `GET /.well-known/agent.json` returns the card. |
| MCP vs A2A | "Tools vs peers" | MCP: vertical agent ↔ tool. A2A: horizontal agent ↔ agent. |

## Further Reading

- [A2A specification](https://a2a-protocol.org/latest/specification/)
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [A2A GitHub repo](https://github.com/a2aproject/A2A)
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/12-a2a-protocol)

---

## Part 2 (ch346): Shared Memory and Blackboard Patterns

> Two approaches coexist in 2026 multi-agent systems: the **message pool** (everyone sees everyone's messages) and the **blackboard with subscription** (agents subscribe to relevant events). Both are the only stateful part of a multi-agent system — which means both are where the interesting bugs live. The reference failure mode is **memory poisoning**.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Multi-agent systems need a place for agents to share facts. When one of the agents hallucinates and writes the hallucination to shared state, every downstream agent that reads that state adopts the hallucination as fact. By the time the human notices, the reasoning chain is five steps deep.

This is memory poisoning. It is the second-most-documented failure family in the MAST taxonomy and it is structural: any shared-memory design without provenance and an unwritable verifier will exhibit it eventually.

## Concept

### The two main topologies

**Full message pool.** Every agent reads every message. AutoGen GroupChat and MetaGPT use this. Simple, transparent, inspectable, but does not scale past ~10 agents.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription.** Agents declare interest in topics; the substrate routes only relevant messages. CA-MCP and Matrix use this. Scales further, but requires upfront schema design.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### Memory poisoning, in one scenario

Three agents work on a research task. Agent A hallucinates a decimal (42% instead of 4.2%). Agent B reads and amplifies it. Agent C recommends adoption based on the inflated number. The final report cites a 42% number that never existed.

No agent crashed. No test failed. The hallucination crossed from one agent's context into every downstream agent's reasoning via shared state.

### Three mitigations

1. **Attribute provenance on every write.** Every entry records who wrote it, when, what source they cited.
2. **Version writes; treat them as append-only.** A correction is a new entry that supersedes the old.
3. **Keep at least one agent that cannot write to shared state.** A read-only verifier agent samples entries, re-fetches sources, and flags inconsistencies.

### Blackboard precedent (Hayes-Roth, 1985)

The blackboard pattern predates LLM agents by four decades. Hayes-Roth described specialist Knowledge Sources that observe a global blackboard, contribute partial solutions, and trigger other sources.

### Projection vs full view

A pure blackboard gives every subscriber the same projection (topic-scoped). A more aggressive design is **per-agent projection**: each agent gets a view customized to its role. LangGraph's state reducers are the canonical implementation.

### The unwritable verifier

The most load-bearing mitigation is the read-only verifier. It shares state with the team (reads the blackboard), has no write handle to shared state, independently fetches sources, and routes outputs to a human or separate decision agent.

## Build It

`code/main.py` implements both topologies in stdlib Python plus a toy poisoning attack and the three mitigations.

- `MessagePool` — thread-safe append-only log with full read-out.
- `Blackboard` — topic-keyed pub/sub with per-agent subscriptions.
- `ProvenanceEntry` — every write records (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — runs a three-agent research task where agent A hallucinates a decimal.
- `Verifier` — a read-only agent that re-fetches sources and flags inconsistencies.

Expected output:
- Run 1 (no verifier): the hallucinated 42% propagates to the final report.
- Run 2 (with verifier): the verifier flags the inconsistency.

```
python3 code/main.py
```

## Ship It

For any shared-memory design:
- Record provenance on every write.
- Make the log append-only.
- Deploy at least one read-only verifier agent with independent source access.
- Route verifier output to a separate channel, not back into the shared pool.
- Log the ratio of writes that are supersessions.

## Exercises

1. Run `code/main.py`. Confirm run 1 propagates the hallucination and run 2 catches it.
2. Add a second hallucination: agent B invents a dataset size. The verifier should catch both.
3. Switch the full pool to a blackboard with topic partitions. Which poisoning scenarios does topic partitioning make harder?
4. Read Hayes-Roth (1985, "A Blackboard Architecture for Control"). Identify two control patterns not discussed here.
5. Read CA-MCP (arXiv:2601.11595). Map its Shared Context Store to either the MessagePool or Blackboard class.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Message pool | "Shared chat history" | Append-only log that every agent reads. |
| Blackboard | "Shared workspace" | Topic-keyed pub/sub. Agents subscribe to relevant topics. |
| Provenance | "Who wrote what" | Metadata on each write: writer, timestamp, prompt, sources. |
| Memory poisoning | "Hallucinations spreading" | One agent's error enters shared state, downstream agents adopt it as fact. |
| Append-only | "No in-place updates" | Corrections are new entries that supersede. |
| Unwritable verifier | "Independent auditor" | Read-only agent that re-fetches sources and flags inconsistencies. |
| Projection | "Scoped view" | Per-agent view computed from global state. |
| Knowledge Source | "Specialist agent" | Hayes-Roth's 1985 term for a blackboard participant. |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595)
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686)
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/13-shared-memory-blackboard)

---

## Part 3 (ch347): Consensus and Byzantine Fault Tolerance for Agents

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

---

## Part 4 (ch349): Negotiation and Bargaining

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
