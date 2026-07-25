# OpenTelemetry GenAI — Tracing Tool Calls End-to-End

> An agent calls five tools, three MCP servers, and two sub-agents. You need one trace across all of it. The OpenTelemetry GenAI semantic conventions (stable in v1.37+) are the 2026 standard, natively supported by Datadog, Langfuse, Arize Phoenix, OpenLLMetry, and AgentOps.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07, 08
**Time:** ~75 minutes

## Learning Objectives
- Name the required OTel GenAI attributes for LLM and tool-execution spans
- Build a trace hierarchy covering agent loop, LLM call, tool call, and MCP dispatch
- Decide what content to capture (opt-in) vs redact (defaults)
- Emit spans to a local collector without rewriting tool code

## The Problem

"Agent sometimes takes 30 seconds, sometimes 3 seconds." No traces. Logs show the LLM call but not the tool dispatch, MCP round-trip, or sub-agent. Without end-to-end tracing, you can't find the cold-start MCP server that occasionally hangs.

## The Concept

### Span hierarchy

```
agent.invoke_agent (top, INTERNAL)
 ├── llm.chat (CLIENT)
 ├── tool.execute (INTERNAL)
 │    └── mcp.call (CLIENT)
 ├── llm.chat (CLIENT)
 └── subagent.invoke (INTERNAL)
```

One trace id across everything. Parent-child links via `parentSpanId`.

### Required attributes

```python
# LLM span
span.set_attribute("gen_ai.operation.name", "chat")
span.set_attribute("gen_ai.provider.name", "openai")
span.set_attribute("gen_ai.request.model", "gpt-4o")
span.set_attribute("gen_ai.response.model", "gpt-4o-2024-08-06")
span.set_attribute("gen_ai.usage.input_tokens", 150)
span.set_attribute("gen_ai.usage.output_tokens", 42)

# Tool span
span.set_attribute("gen_ai.tool.name", "get_weather")
span.set_attribute("gen_ai.tool.call.id", "call_abc123")

# Agent span
span.set_attribute("gen_ai.agent.name", "research-agent")
```

### Span kinds

- `SpanKind.CLIENT` for calls crossing process boundaries (LLM provider, MCP server)
- `SpanKind.INTERNAL` for agent's own loop steps and tool execution

### Opt-in content capture

By default, spans carry metrics and timing — not prompts/completions. Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` to include content. Review carefully before enabling in production.

### Events on spans

Token-level events: `gen_ai.content.prompt` (input messages), `gen_ai.content.completion` (output messages), `gen_ai.content.tool_call` (tool call as recorded).

### Propagation across MCP

Inject W3C `traceparent` header into requests. Streamable HTTP supports standard headers. For stdio, include `traceparent` in `_meta` of every JSON-RPC call until the spec formalizes it.

### Metrics

Alongside spans: `gen_ai.client.token.usage` (histogram), `gen_ai.client.operation.duration` (histogram), `gen_ai.tool.execution.duration` (histogram).

## Use It

`code/main.py` emits OTel-shaped spans to stdout for an agent that calls an LLM, dispatches two tools, and makes one MCP round-trip. Focus is on span shape and attribute set, not exporters.

## Exercises

1. Count spans and identify CLIENT vs INTERNAL.
2. Turn on content capture and confirm event attributes appear.
3. Add `gen_ai.tool.execution.duration` histogram metric.
4. Propagate `traceparent` from agent span into MCP `_meta`.
5. Identify one attribute from the OTel semconv spec not emitted in this code and add it.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| OTel | Open standard for traces, metrics, logs |
| GenAI semconv | Stable attribute names for LLM/tool/agent spans |
| `gen_ai.*` | The attribute namespace for GenAI operations |
| Span | Timed operation with start, end, and attributes |
| SpanKind | CLIENT / SERVER / INTERNAL direction hints |
| OTLP | Wire format for exporting to backends |
| traceparent | W3C header propagating trace context across services |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/20-opentelemetry-genai)
