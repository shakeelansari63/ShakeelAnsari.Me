# Heritage of FIPA-ACL and Speech Acts

> Before MCP, before A2A, there was FIPA-ACL. In 2000 the IEEE Foundation for Intelligent Physical Agents ratified an agent communication language with twenty performatives, two content languages, and a set of interaction protocols — contract net, subscribe/notify, request-when. It faded from industry because the ontology overhead was too heavy for the web, but the LLM revival of multi-agent systems is quietly reimplementing the same ideas without the formal semantics: JSON contracts stand in for performatives, natural language stands in for ontologies.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## Problem

The 2026 agent-protocol landscape is busy: MCP for tools, A2A for agents, ACP for enterprise audit, ANP for decentralized trust, NLIP for natural-language content, plus CA-MCP and two dozen research proposals. Each spec announces itself as foundational.

The honest read is that most of them are rediscovering a very specific twenty-year-old decision tree. Speech-act theory from Austin (1962) and Searle (1969) gave us "utterances are actions." KQML (1993) turned that into a wire protocol. FIPA-ACL (ratified 2000) produced the reference standardization: twenty performatives, content languages SL0/SL1, interaction protocols for contract-net and subscribe-notify. JADE and JACK were the Java reference platforms. The effort faded around 2010 because the ontology overhead was too heavy and the web was winning.

When you look at MCP's `tools/call`, A2A's task lifecycle, or CA-MCP's shared context store, you are looking at a softer, JSON-native rehash of FIPA decisions. Knowing the heritage tells you two things: which new "innovations" are actually reinventions, and which old failure modes the new specs will rediscover.

## Concept

### Speech acts, in one paragraph

Austin noticed that some sentences do not describe the world — they change it. "I promise." "I request." "I declare." He called these performative utterances. Searle formalized five categories: assertive, directive, commissive, expressive, declarative. KQML (Finin et al., 1993) made this operational for software agents: a message is a performative (the action) plus content (what the action is about). FIPA-ACL cleaned up KQML's gaps and standardized around twenty performatives.

### The twenty FIPA performatives (partial list)

| Performative | Intent |
|---|---|
| `inform` | "I tell you P is true" |
| `request` | "I ask you to do X" |
| `query-if` | "Is P true?" |
| `query-ref` | "What is the value of X?" |
| `propose` | "I propose we do X" |
| `accept-proposal` | "I accept the proposal" |
| `reject-proposal` | "I reject the proposal" |
| `agree` | "I agree to do X" |
| `refuse` | "I refuse to do X" |
| `confirm` | "I confirm P is true" |
| `disconfirm` | "I deny P" |
| `not-understood` | "Your message did not parse" |
| `cfp` | "Call for proposals on X" |
| `subscribe` | "Notify me when X changes" |
| `cancel` | "Cancel the ongoing X" |
| `failure` | "I tried X and failed" |

### Canonical FIPA-ACL message

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Seven fields carry the protocol envelope; one field (`content`) carries the payload.

### The two legacy platforms

**JADE** (Java Agent DEvelopment framework, 1999–2020s) was the most-used FIPA-compliant runtime. Agents extended a base class, exchanged ACL messages, ran inside containers, and coordinated using "behaviors."

**JACK** (Agent Oriented Software, commercial) emphasized BDI (Belief-Desire-Intention) reasoning on top of FIPA messages.

Both declined once the web stack ate multi-agent use cases. MCP and A2A are the runtime "containers" of 2026.

### Why FIPA faded

- **Ontology overhead.** FIPA required a shared ontology to parse `content`. Agreeing on ontologies is a years-long standards process. The web just used HTTP + JSON.
- **Formal semantics nobody used.** SL (Semantic Language) gave rigorous truth conditions, but most production systems used free-form content and ignored the formalism.
- **Tooling lock-in.** JADE was Java-only; JACK was commercial. Polyglot teams routed around both.
- **The internet won the stack.** REST, then JSON-RPC, then gRPC replaced ACL's transport.

### The LLM revival is FIPA-lite

Compare a FIPA `request` to an MCP `tools/call`:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Same envelope, different syntax. Both carry: who, whom, intent, payload, correlation id.

### The trade-off, stated plainly

What FIPA gave you and modern specs drop: formal semantics, a canonical catalog of performatives, and decades of interaction-protocol patterns with known correctness properties.

What modern specs give you and FIPA did not: JSON-native payloads, natural-language content that LLMs can interpret, web-stack transport (HTTP, SSE, WebSocket), and capability discovery via self-describing documents.

### Interaction protocols worth porting

1. **Contract Net Protocol (CNP).** Manager issues `cfp` (call for proposals); bidders respond with `propose`; manager accepts/rejects. This is the canonical task-market pattern.
2. **Subscribe/Notify.** Subscriber sends `subscribe`; publisher sends `inform` whenever the topic changes.
3. **Request-When.** "Do X when condition Y holds." Delayed-action with pre-conditions.

### What breaks when you drop the ontology

Without a shared ontology, agents infer meaning from natural-language content. The documented 2026 failure mode is **semantic drift**: two agents use the same word ("customer") for subtly different concepts. Mitigations: JSON Schema on `content`, typed artifacts, explicit performative in the envelope.

### The 2026 specs, mapped to speech-act heritage

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | explicit intent, correlation id | formal semantics, ontology |
| MCP `resources/read` | `query-ref` | explicit intent, correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle, state transitions | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

## Build It

`code/main.py` implements a pure-stdlib FIPA-ACL translator. It encodes and decodes the canonical ACL envelope and shows how every MCP / A2A message shape reduces to the same seven fields. The demo encodes five MCP-style and A2A-style messages as FIPA-ACL, decodes FIPA-ACL back to the modern equivalent, and runs a toy Contract Net negotiation between one manager and three bidders using `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

```
python3 code/main.py
```

## Use It

`outputs/skill-fipa-mapper.md` is a skill that reads any agent-protocol spec and produces the FIPA-ACL mapping.

## Ship It

Do not bring FIPA-ACL back. Bring back its checklist:

- What is the intent primitive (performative) of each message?
- Is there a correlation id for request-response and cancellation?
- Is there an explicit content language?
- Are interaction protocols first-class, or are you re-implementing contract-net from scratch?
- What happens when two agents disagree about content meaning (semantic drift)?

## Exercises

1. Run `code/main.py`. Observe the round-trip encoding. Identify which FIPA performative corresponds to `tools/call`, `resources/read`, and A2A task creation.
2. Extend the contract-net demo with a `cancel` performative that lets the manager withdraw the task mid-bid.
3. Read FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) sections 4.1–4.3. Pick one performative not covered in this lesson and describe its modern JSON-RPC analog.
4. Read Liu et al., arXiv:2505.02279. For each of MCP, A2A, ACP, ANP, list the FIPA performative families they keep and drop.
5. Design a minimal JSON-Schema for the `content` field of a `request` performative in your own system.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speech act | "An utterance that does something" | Austin/Searle: utterances as actions. The theoretical parent of ACL. |
| FIPA | "That old XML thing" | IEEE Foundation for Intelligent Physical Agents. Standardized ACL in 2000. |
| ACL | "Agent Communication Language" | FIPA's envelope format: performative + content + metadata. |
| Performative | "The verb" | The intent class of a message: `inform`, `request`, `propose`, `cfp`, etc. |
| KQML | "FIPA's predecessor" | Knowledge Query and Manipulation Language (1993). |
| Ontology | "Shared vocabulary" | A formal definition of the concepts the content language talks about. |
| Contract Net | "Task market" | Manager issues cfp; bidders propose; manager accepts. |
| Interaction protocol | "Pattern of messages" | A sequence of performatives with known correctness. |

## Further Reading

- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1)
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/)
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [A2A specification](https://a2a-protocol.org/latest/specification/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/02-fipa-acl-heritage)
