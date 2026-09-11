# Voice Agents, OTel & Observability Dashboards

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch291): Voice Agents: Pipecat and LiveKit

> Voice agents are a first-class production category in 2026. Pipecat gives you a Python frame-based pipeline (VAD → STT → LLM → TTS → transport). LiveKit Agents bridges AI models to users over WebRTC. Production latency targets land at 450–600ms end-to-end for premium stacks.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## Learning Objectives

- Describe Pipecat's frame-based pipeline: DOWNSTREAM (source→sink) and UPSTREAM (control).
- Name the canonical voice pipeline stages and which transports Pipecat supports.
- Explain LiveKit Agents' two voice agent classes (MultimodalAgent, VoicePipelineAgent) and when each fits.
- Summarize 2026 production latency expectations and how they drive architecture choices.

## The Problem

Voice agents are not a text loop with TTS bolted on. Latency budgets are brutal (~600ms), partial audio is the default, turn detection is a model, and transports range from telephony SIP to WebRTC. Either you build a frame-based pipeline (Pipecat) or you lean on a platform (LiveKit).

## The Concept

### Pipecat (pipecat-ai/pipecat)

- Python frame-based pipeline framework.
- `Frame` → `FrameProcessor` chain.
- Two flow directions:
  - **DOWNSTREAM** — source → sink (audio in, TTS out).
  - **UPSTREAM** — feedback and control (cancellation, metrics, barge-in).
- `PipelineTask` manages lifecycle with events (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) and observers for metrics/tracing/RTVI.

Typical pipeline:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows adds structured conversations (state machines). Pipecat Cloud is the managed runtime.

### LiveKit Agents (livekit/agents)

- Bridges AI models to users over WebRTC.
- Key concepts: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Two voice agent classes:
  - **MultimodalAgent** — direct audio via OpenAI Realtime or equivalent.
  - **VoicePipelineAgent** — STT → LLM → TTS cascade; gives text-level control.
- Semantic turn detection via a transformer model.
- Native MCP integration.
- Telephony via SIP.
- 50+ models with no API keys via LiveKit Inference; 200+ more via plugins.

### Commercial platforms

Vapi (~450–600ms on an optimized premium stack) and Retell (~600ms end-to-end across 180 test calls) build on top of these. Pick a platform when you want a managed voice stack without a WebRTC team.

### Where this pattern goes wrong

- **No barge-in handling.** User interrupts; agent keeps talking. Requires UPSTREAM cancel frames in Pipecat, equivalent in LiveKit.
- **STT confidence ignored.** Low-confidence transcripts fed to the LLM as if gospel. Gate on confidence or request confirmation.
- **TTS mid-sentence cutoff.** When the pipeline cancels mid-utterance, TTS needs to know or cut audio.
- **Latency budget ignored.** Every component adds 50–200ms. Sum your chain before shipping.

### Typical 2026 latencies

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

End-to-end 450–600ms is premium. 800–1200ms is common. Anything > 1500ms feels broken.

## Build It

`code/main.py` is a frame-based toy pipeline with:

- `Frame` types (audio, transcript, text, tts_audio, control).
- `Processor` interface with `process(frame)`.
- A five-stage pipeline (VAD → STT → LLM → TTS → transport) as scripted processors.
- An UPSTREAM cancel frame to demonstrate barge-in.

Run it:

```
python3 code/main.py
```

The trace shows normal flow and a barge-in cancel that stops TTS mid-utterance.

## Use It

- **Pipecat** for full control — custom processors, Python-first, pluggable providers.
- **LiveKit Agents** for WebRTC-first deployments and telephony.
- **Vapi / Retell** for hosted voice agents without a WebRTC team.
- **OpenAI Realtime / Gemini Live** for direct audio-in/audio-out (MultimodalAgent).

## Exercises

1. Add a metrics observer to your toy pipeline: count frames per stage per second. Where does latency accumulate?
2. Implement confidence-gated STT: below threshold, request "could you repeat that?"
3. Add semantic turn detection: simple rule — if transcript ends with "?", end of turn.
4. Read Pipecat's transport docs. Swap the stdlib transport for the SmallWebRTCTransport config (stub).
5. Measure an OpenAI Realtime vs STT+LLM+TTS cascade on the same query. What latency cost does text-level control carry?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Frame | "Event" | Typed unit of data in the pipeline (audio, transcript, text, control) |
| Processor | "Pipeline stage" | Handler with process(frame) |
| DOWNSTREAM | "Forward flow" | Source to sink: audio in, speech out |
| UPSTREAM | "Feedback flow" | Control: cancel, metrics, barge-in |
| VAD | "Voice activity detection" | Detects when user is speaking |
| Semantic turn detection | "Smart end-of-turn" | Model-based decision that the user is done |
| MultimodalAgent | "Direct audio agent" | Audio in, audio out; no text in the middle |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS; text-level control |

## Further Reading

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline, processors, transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitives
- [Vapi](https://vapi.ai/) — managed voice platform
- [Retell AI](https://www.retellai.com/) — managed voice, latency-benchmarked

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/22-voice-agents-pipecat-livekit)

---

## Part 2 (ch292): OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry's GenAI SIG (launched April 2024) defines the standard schema for agent telemetry. Span names, attributes, and content-capture rules converge across vendors so agent traces mean the same thing in Datadog, Grafana, Jaeger, and Honeycomb.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## Learning Objectives

- Name the GenAI span categories: model/client, agent, tool.
- Distinguish `invoke_agent` CLIENT vs INTERNAL spans and when each applies.
- List the top-level GenAI attributes: provider name, request model, data-source ID.
- Explain the content-capture contract: opt-in, `OTEL_SEMCONV_STABILITY_OPT_IN`, external-reference recommendation.

## The Problem

Every vendor invents their own span names. Ops teams end up building per-framework dashboards. OpenTelemetry's GenAI SIG fixes this by defining one standard the whole ecosystem targets.

## The Concept

### Span categories

1. **Model / client spans.** Cover raw LLM calls. Emitted by provider SDKs (Anthropic, OpenAI, Bedrock) and framework model adapters.
2. **Agent spans.** `create_agent` (when the agent is constructed) and `invoke_agent` (when it runs).
3. **Tool spans.** One per tool invocation; connected to the agent span by parent-child relation.

### Agent span naming

- Span name: `invoke_agent {gen_ai.agent.name}` if named; fallback to `invoke_agent`.
- Span kind:
  - **CLIENT** — for remote agent services (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — for in-process agent frameworks (LangChain, CrewAI, local ReAct).

### Key attributes

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — the model ID.
- `gen_ai.response.model` — the resolved model (may differ from request due to routing).
- `gen_ai.agent.name` — agent identifier.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — for RAG: which corpus or store was consulted.

Technology-specific conventions exist for Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Content capture

The default rule: instrumentations SHOULD NOT capture inputs/outputs by default. Capture is opt-in via:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Recommended production pattern: store content externally (S3, your log store), record references on spans (pointer IDs, not prose). This is the Lesson 27 content-poisoning defense wired into observability.

### Stability

Most conventions are experimental as of March 2026. Opt in to the stable preview with:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ maps GenAI attributes natively into its LLM Observability schema. Other backends (Grafana, Honeycomb, Jaeger) support the raw attributes.

### Where this pattern goes wrong

- **Capturing full prompts in spans.** PII, secrets, customer data in traces that ops can read. Store externally.
- **No `gen_ai.provider.name`.** Multi-provider dashboards break when attribution is missing.
- **Spans without parent links.** Orphaned tool spans. Always propagate context.
- **Not setting stability opt-in.** Your attributes may get renamed on backend upgrade.

## Build It

`code/main.py` implements a stdlib span emitter matching GenAI conventions:

- `Span` with GenAI attribute schema.
- `Tracer` with `start_span`, nested contexts.
- A scripted agent run that emits: `create_agent`, `invoke_agent` (INTERNAL), per-tool spans, `chat` spans for LLM calls.
- A content-capture mode that stores prompts externally and records IDs on spans.

Run it:

```
python3 code/main.py
```

Output: a span tree with all required GenAI attributes, and an "external store" showing the opt-in content references.

## Use It

- **Datadog LLM Observability** (v1.37+) maps attributes natively.
- **Langfuse / Phoenix / Opik** (Lesson 24) — auto-instrument the ecosystem.
- **Jaeger / Honeycomb / Grafana Tempo** — raw OTel traces; build dashboards from GenAI attributes.
- **Self-hosted** — run the OTel Collector with a GenAI processor.

## Exercises

1. Instrument your Lesson 01 ReAct loop with `invoke_agent` (INTERNAL) + per-tool spans. Send to a Jaeger instance.
2. Add content capture in "references only" mode: prompts to SQLite, span attributes carry only row IDs.
3. Read the spec for `gen_ai.data_source.id`. Wire it into your Lesson 09 Mem0 search.
4. Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` and verify your attributes don't get renamed by the collector.
5. Build a dashboard: "which tool errors correlate with which models" from GenAI attributes alone.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI group" | OTel working group defining the schema |
| invoke_agent | "Agent span" | Name of the span representing an agent run |
| CLIENT span | "Remote call" | Span for a call to a remote agent service |
| INTERNAL span | "In-process" | Span for an in-process agent run |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | Which corpus/store a retrieval hit |
| Content capture | "Prompt logging" | Opt-in capture of messages; store externally in prod |
| Stability opt-in | "Preview mode" | Env var to pin experimental conventions |

## Further Reading

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans built in
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/23-otel-genai-conventions)

---

## Part 3 (ch293): Agent Observability: Langfuse, Phoenix, Opik

> Three open-source agent observability platforms dominate 2026. Langfuse (MIT) — 6M+ installs/month, tracing + prompt management + evals + session replay. Arize Phoenix (Elastic 2.0) — deep agent-specific evals, RAG relevancy, OpenInference auto-instrumentation. Comet Opik (Apache 2.0) — automated prompt optimization, guardrails, LLM-judge hallucination detection.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## Learning Objectives

- Name the three top open-source agent observability platforms and their licenses.
- Distinguish what each one is strongest at: Langfuse (prompt mgmt + sessions), Phoenix (RAG + auto-instrumentation), Opik (optimization + guardrails).
- Explain why 89% of organizations report having agent observability in place by 2026.
- Implement a stdlib trace-to-dashboard pipeline with LLM-judge evaluation.

## The Problem

OTel GenAI (Lesson 23) gives you the schema. You still need the platform that ingests spans, runs evaluations, stores prompt versions, and surfaces regressions. The three contenders each emphasize different parts of the lifecycle.

## The Concept

### Langfuse (MIT)

- 6M+ SDK installs/month, 19k+ GitHub stars.
- Features: tracing, prompt management with versioning + playground, evaluations (LLM-as-judge, user feedback, custom), session replays.
- June 2025: formerly commercial modules (LLM-as-a-judge, annotation queues, prompt experiments, Playground) open-sourced under MIT.
- Strongest for: end-to-end observability with tight prompt-management loop.

### Arize Phoenix (Elastic License 2.0)

- Deeper agent-specific evaluation: trace clustering, anomaly detection, retrieval relevancy for RAG.
- Native OpenInference auto-instrumentation.
- Pairs with managed Arize AX for production.
- No prompt versioning — positioned as a drift/behavioral-regression tool alongside broader platforms.
- Strongest for: RAG relevancy, behavioral drift, anomaly detection.

### Comet Opik (Apache 2.0)

- Automated prompt optimization through A/B experiments.
- Guardrails (PII redaction, topical constraints).
- LLM-judge hallucination detection.
- Benchmark from Comet's own measurement: Opik logs + evals in 23.44s vs Langfuse 327.15s (~14x gap) — take vendor benchmarks as directional.
- Strongest for: optimization loop, automated experimentation, guardrail enforcement.

### Industry data

Per Maxim (2026 field analysis): 89% of organizations have agent observability in place; quality issues are the top production barrier (32% of respondents cite them).

### Picking one

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### Where this pattern goes wrong

- **No eval strategy.** Tracing without evaluation is just expensive logging.
- **Self-rolled LLM-judge without grounding.** CRITIC pattern (Lesson 05) applies — judges need external tools for factual verification.
- **Prompt versions not tied to traces.** When prod regresses, you cannot bisect to the prompt that caused it.

## Build It

`code/main.py` implements a stdlib trace collector + LLM-judge evaluator:

- Ingest GenAI-shaped spans.
- Group by session, tag failed runs (guardrail trips, low-confidence evals).
- A scripted LLM-judge that scores agent responses on a rubric.
- A dashboard-like summary: failure rate, top failure reasons, eval score distribution.

Run it:

```
python3 code/main.py
```

Output: per-session eval scores and failure categorization matching what Langfuse/Phoenix/Opik would show.

## Use It

- **Langfuse** self-hosted or cloud; wire via OTel or their SDK.
- **Arize Phoenix** self-hosted; auto-instrument OpenInference.
- **Comet Opik** self-hosted or cloud; automated optimization loop.
- **Datadog LLM Observability** for mixed ops+ML teams that already run Datadog.

## Exercises

1. Export a week of OTel traces to Langfuse cloud (free tier). Which sessions failed? Why?
2. Write an LLM-judge rubric for your domain (factual correctness, tone, scope adherence). Test on 50 traces.
3. Compare Langfuse prompt versioning against Phoenix's trace clustering. Which tells you what broke faster?
4. Read Opik's guardrail docs. Wire a PII redaction guardrail to one of your agent runs.
5. Benchmark the three on your corpus. Ignore vendor-published numbers; measure your own.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tracing | "Spans collector" | Ingest OTel / SDK spans; index by session |
| Prompt management | "Prompt CMS" | Versioned prompts tied to traces |
| LLM-as-judge | "Automated eval" | Separate LLM scores agent output against a rubric |
| Session replay | "Trace playback" | Step through past runs for debugging |
| RAG relevancy | "Retrieval quality" | Does the retrieved context match the query |
| Trace clustering | "Behavioral grouping" | Cluster similar runs for drift detection |
| Guardrail enforcement | "Policy at log time" | PII/toxicity/scope checks on logged content |

## Further Reading

- [Langfuse docs](https://langfuse.com/) — tracing, evals, prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the schema all three consume

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/14-agent-engineering/24-agent-observability-platforms)

---

## Part 4 (ch427): LLM Observability & Eval Dashboard

> Langfuse went open-core. Arize Phoenix published the 2026 GenAI semconv mappings. Helicone and Braintrust both doubled down on per-user cost attribution. Traceloop's OpenLLMetry became the de-facto SDK instrumentation. The production shape is ClickHouse for traces, Postgres for metadata, Next.js for UI, and a small army of eval jobs (DeepEval, RAGAS, LLM-judge) running over sampled traces. Build one self-hosted, ingest from at least four SDK families, and demonstrate catching an injected regression in under five minutes.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Time:** 25 hours

## Problem

Every AI team running production traffic in 2026 keeps an observability plane alongside the model. Cost attribution. Hallucination detection. Drift monitoring. Jailbreak signal. SLO dashboards. PII leak alerts. The open-source references — Langfuse, Phoenix, OpenLLMetry — converged on OpenTelemetry GenAI semantic conventions as the ingest schema. You can now instrument OpenAI, Anthropic, Google, LangChain, LlamaIndex, and vLLM with one SDK and ship compatible spans.

You will build a self-hosted dashboard that ingests from at least four SDK families, runs a small set of eval jobs over sampled traces, detects drift, and alerts. The measurement bar: given a deliberately injected regression (a prompt that starts producing PII), the dashboard catches it and fires an alert in under five minutes.

## Concept

Ingest is OTLP HTTP. The SDK produces GenAI-semconv spans: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Spans land in ClickHouse for columnar analytics; metadata (users, sessions, apps) lands in Postgres.

Evals run as batch jobs over sampled traces. DeepEval scores faithfulness, toxicity, and answer relevance. RAGAS scores retrieval metrics when the trace carries retrieval context. Custom LLM-judges run domain-specific checks (PII leak, off-policy response). Eval runs write back to the same ClickHouse as eval spans linked to the parent trace.

Drift detection watches embedding-space distributions over time (PSI or KL divergence on prompt embeddings) plus eval-score trends. Alerts feed Prometheus Alertmanager and then Slack / PagerDuty. The UI is Next.js 15 with Recharts.

## Architecture

```mermaid
graph TD
    apps[production apps: OpenAI / Anthropic / Google / LangChain / LlamaIndex / vLLM] --> otel[OpenTelemetry SDK with GenAI semconv]
    otel --> collector[collector (OTLP HTTP)]
    collector --> click[ClickHouse (spans)]
    collector --> pg[Postgres (metadata)]
    collector --> s3[S3 archive (raw events)]
    click --> eval[eval jobs: DeepEval / RAGAS / LLM-judge]
    click --> drift[drift detector: PSI / KL]
    click --> prom[Prometheus metrics]
    prom --> alert[Alertmanager -> Slack / PagerDuty]
    click --> ui[Next.js 15 dashboard]
```

## Stack

- Ingest: OpenTelemetry SDKs + GenAI semantic conventions; OTLP HTTP transport
- Collector: OpenTelemetry Collector with tail-sampling processor (for cost control)
- Storage: ClickHouse for spans, Postgres for metadata, S3 for raw event archive
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix evaluator pack, custom LLM-judge
- Drift: PSI / KL on pooled prompt embeddings (sentence-transformers) weekly
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDKs supported out of the box: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## Build It

1. **Collector config.** OpenTelemetry Collector with the OTLP HTTP receiver, a tail-sampler keeping 100% of errored traces and 10% of successes, and exporters to ClickHouse and S3.

2. **ClickHouse schema.** Table `spans` with columns mirroring GenAI semconv: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, plus JSON bag for long payloads. Add secondary indexes by user_id and app_id.

3. **SDK coverage test.** Write a small client app using each SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) with OpenLLMetry auto-instrument. Verify each produces canonical GenAI spans that land in ClickHouse.

4. **Eval jobs.** A scheduled job reads last-15-min sampled traces and runs DeepEval faithfulness, toxicity, and answer relevance. Outputs are eval spans linked to the parent trace.

5. **Custom LLM-judge.** A PII-leak judge: given a response, call a guard LLM to score likelihood of PII leak. High-score responses land in a triage queue.

6. **Drift detection.** Weekly job computes PSI between this week's pooled prompt embeddings and the trailing 4-week baseline. If PSI above threshold, alert.

7. **Dashboard.** Next.js 15 with pages: overview (spans/sec, cost/user, p95 latency), traces (search + waterfall), evals (faithfulness trend, toxicity), drift (PSI over time), alerts.

8. **Alerting chain.** Prometheus exporter reads eval score aggregates and latency percentiles; Alertmanager routes to Slack for warnings and PagerDuty for critical breaches.

9. **Regression probe.** Inject a bug: the evaluated chatbot starts leaking fake SSNs 1% of the time. Measure MTTR: from bug deployed to Slack alert.

## Use It

```console
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Ship It

`outputs/skill-llm-observability.md` is the deliverable. Given an LLM application, the dashboard ingests its traces, runs evals, alerts on drift, and surfaces cost/user breakdown in Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Exercises

1. Add custom instrumentation for the Haystack framework. Verify canonical spans land in ClickHouse with faithful `gen_ai.*` attributes.

2. Swap DeepEval for Phoenix evaluators on the same traces. Measure score drift between the two eval engines.

3. Sharpen the drift detector: compute PSI per app-id rather than globally. Show per-app drift trails.

4. Add a "user impact" page: cost-per-user and failure-rate-per-user with sparklines.

5. Build a tail-sampling policy that keeps 100% of traces with toxicity > 0.5 plus a 10% stratified sample of the rest. Measure sampling bias introduced.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Further Reading

- [Langfuse](https://github.com/langfuse/langfuse)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Helicone](https://www.helicone.ai)
- [Braintrust](https://www.braintrust.dev)
- [ClickHouse documentation](https://clickhouse.com/docs)
- [DeepEval](https://github.com/confident-ai/deepeval)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/11-llm-observability-dashboard)

---

## Part 5 (ch419): Real-Time Voice Assistant (ASR to LLM to TTS)

> A voice agent that feels right has end-to-end latency under 800ms, knows when you have stopped talking, handles barge-in, and can call a tool without stalling. Retell, Vapi, LiveKit Agents, and Pipecat all hit this bar in 2026. They do it with the same shape: a streaming ASR, a turn-detector, a streaming LLM, and a streaming TTS, all wired through WebRTC with aggressive latency budgets at every hop. Build one, measure WER and MOS and false-cutoff rate, and run it under packet loss.

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Time:** 30 hours

## Problem

Voice has been the fastest-moving AI UX category of 2025-2026. The technical ceiling dropped each quarter. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, and Pipecat 0.0.70 all put sub-800ms first-audio-out within reach. The bar is not latency alone. It is the interaction feel: not cutting the user off, not getting cut off, recovering from a mid-sentence interruption, calling a tool mid-conversation without stalling the audio, surviving jittery mobile networks.

You cannot get there by stitching three REST calls. The architecture is pipelined streaming end to end. Build it and the failure modes become visible: a VAD tuned for phone audio firing on background TV, a turn-detector waiting for punctuation that never comes, a TTS that buffers 400ms before emitting. The capstone is to fix these one at a time under load and publish a latency-and-quality report.

## Concept

The pipeline has five streaming stages: **audio in** (WebRTC from browser or PSTN), **ASR** (streaming partial transcripts from Deepgram Nova-3 or faster-whisper), **turn detection** (VAD plus a small turn-detector model that reads partial transcripts for completion cues), **LLM** (streaming tokens as soon as the turn is judged complete), **TTS** (streaming audio out within ~200ms of the first LLM token).

Three cross-cutting concerns. **Barge-in**: when the user starts speaking while the agent is speaking, the TTS cancels and the ASR picks up immediately. **Tool use**: mid-conversation function calls (weather, calendar) must run on a side channel without stalling the audio; the agent pre-fills an acknowledgement token ("one second...") if latency exceeds 300ms. **Backpressure**: under packet loss, partial transcripts are held, VAD raises the speech-gate threshold, and the agent avoids speaking over an unacknowledged message.

The measurement bar is quantitative. WER under 8% on the Hamming VAD benchmark at 15 dB SNR. First-audio-out p50 under 800ms on 100 measured calls. False-cutoff rate under 3%. MOS above 4.2 on TTS. 50 concurrent calls on a single g5.xlarge. These numbers are the deliverable.

## Architecture

```mermaid
graph TD
    client[browser / Twilio PSTN] --> edge[WebRTC / SIP edge]
    edge --> lk[LiveKit Agents 1.0]
    lk --> asr[ASR (Deepgram Nova-3 / Whisper-v3)]
    lk --> vad[VAD v5 (Silero)]
    lk --> turn[turn-detector (LiveKit)]
    lk --> side[side-channel tools]
    asr --> llm[LLM (streaming)]
    vad --> llm
    turn --> llm
    llm --> tts[TTS streaming (Cartesia / ElevenLabs)]
    tts --> audio[audio back to caller]
    llm --> otel[OpenTelemetry voice traces --> Langfuse]
```

## Stack

- Transport: LiveKit Agents 1.0 (WebRTC) plus Twilio PSTN gateway; Pipecat 0.0.70 as the alternate framework
- ASR: Deepgram Nova-3 (streaming, sub-300ms first partial) or faster-whisper Whisper-v3-turbo self-hosted
- VAD: Silero VAD v5 plus the LiveKit turn-detector (small transformer that reads partial transcripts)
- LLM: OpenAI GPT-4o-realtime for tight integration, Gemini 2.5 Flash Live, or cascaded Claude Haiku 4.5 (streaming completions, separate audio path)
- TTS: Cartesia Sonic-2 (lowest first-byte), ElevenLabs Flash v3, or open-source Orpheus for self-host
- Tools: FastMCP side-channel for weather/calendar/booking; agent pre-emits filler if tool takes >300ms
- Observability: OpenTelemetry voice spans, Langfuse voice traces with audio replay
- Deployment: single g5.xlarge (24GB VRAM) for self-hosted Whisper + Orpheus; hosted APIs for lowest latency

## Build It

1. **WebRTC session.** Stand up a LiveKit room and a web client that streams microphone audio. On the server, attach an agent worker that joins the room.

2. **ASR streaming.** Feed 20ms PCM frames to Deepgram Nova-3 (or faster-whisper on GPU). Subscribe to partial and final transcripts. Log per-partial latency.

3. **VAD and turn detector.** Run Silero VAD v5 on the frame stream. On speech-end event, fire the LiveKit turn-detector against the latest partial transcript. Only commit to "turn complete" when VAD says silence for 500ms and the turn-detector scores completion > 0.6.

4. **LLM stream.** On turn complete, start the LLM call with the running conversation plus the final transcript. Stream tokens out. At the first token, hand off to TTS.

5. **TTS stream.** Cartesia Sonic-2 streams audio chunks back. The first chunk must leave the server within 200ms of the first LLM token. Emit chunks to LiveKit room; client plays through WebRTC jitter buffer.

6. **Barge-in.** When VAD detects new user speech while TTS is playing, cancel the TTS stream immediately, drop the remaining LLM output, and re-arm the ASR. Publish a `tts_canceled` span.

7. **Tool side channel.** Register weather and calendar as function-calling tools. When invoked, fire the call concurrently; if it does not resolve within 300ms, have the LLM emit "one second, let me check" as a filler; resume once the tool returns.

8. **Eval harness.** Record 100 calls. Compute WER (against a held-out transcript), false-cutoff rate (TTS cancelled while user was mid-sentence), first-audio-out p50, TTS MOS (human or NISQA), and a jitter-loss test (drop 3% of packets).

9. **Load test.** Drive 50 concurrent calls on a single g5.xlarge with a synthetic caller. Measure sustained first-audio-out p95.

## Use It

```console
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Ship It

`outputs/skill-voice-agent.md` is the deliverable. Given a domain (customer support, scheduling, or kiosk), it stands up a LiveKit agent with the ASR/VAD/LLM/TTS pipeline tuned to the measurement bar. Rubric:

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | End-to-end latency | p50 first-audio-out under 800ms across 100 recorded calls |
| 20 | Turn-taking quality | False-cutoff rate under 3% on the Hamming VAD benchmark |
| 20 | Tool-use correctness | Mid-conversation tool calls that return the right data without stalling audio |
| 20 | Reliability under packet loss | WER and turn-taking stability with 3% packet drop injected |
| 15 | Eval harness completeness | Reproducible measurements with public config |
| **100** | | |

## Exercises

1. Swap Deepgram Nova-3 for faster-whisper v3 turbo on a g5.xlarge. Measure the latency and WER gap. Identify where CPU-vs-GPU decisions matter.

2. Add an interruption-arbitration policy: what does the agent do when the user barges in during a tool call? Compare three policies (hard cancel, finish-tool-then-stop, queue next turn).

3. Run an adversarial turn-detector test: give the user long pauses mid-sentence. Tune the VAD silence threshold and the turn-detector score threshold for lowest false-cutoff without blowing past 900ms.

4. Deploy the same agent on PSTN via Twilio. Compare PSTN first-audio-out to WebRTC. Explain the jitter-buffer and codec differences.

5. Add voice activity detection for non-English languages (Japanese, Spanish). Measure the Silero VAD v5 false-trigger rate versus language-specific fine-tunes.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Turn detection | "End of utterance" | Classifier that, given VAD silence and a partial transcript, decides the user is done speaking |
| Barge-in | "Interruption handling" | Canceling TTS mid-playback when VAD detects new user speech |
| First-audio-out | "Latency" | Time from user stops speaking to the first audio packet leaving the server |
| VAD | "Speech gate" | Model classifying audio frames as speech vs silence; Silero VAD v5 is the 2026 default |
| Jitter buffer | "Audio smoothing" | Client-side buffer that holds packets briefly to absorb network variance |
| Filler | "Acknowledgment token" | Short phrase the agent emits to avoid silence when a tool is slow |
| MOS | "Mean opinion score" | Perceptual speech quality rating; NISQA is the automated proxy |

## Further Reading

- [LiveKit Agents 1.0](https://github.com/livekit/agents)
- [Pipecat](https://github.com/pipecat-ai/pipecat)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs)
- [Silero VAD v5](https://github.com/snakers4/silero-vad)
- [Cartesia Sonic-2](https://docs.cartesia.ai)
- [Retell AI architecture](https://docs.retellai.com)
- [Vapi.ai production stack](https://docs.vapi.ai)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/03-realtime-voice-assistant)
