# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

> The three frontier providers converged on the same tool-call loop in 2024 and then diverged on everything else. OpenAI uses `tools` and `tool_calls`. Anthropic uses `tool_use` and `tool_result` blocks. Gemini uses `functionDeclarations` and unique-id correlation. This lesson diffs the three side by side so code that ships on one provider does not break when you port it.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01
**Time:** ~75 minutes

## Learning Objectives
- State the three shape differences between OpenAI, Anthropic, and Gemini function-calling payloads
- Translate one tool declaration across all three provider formats
- Use `tool_choice` in each provider to force, forbid, or auto-pick tool calls
- Know per-provider hard limits and error signatures

## The Problem

Same loop, different field names, different nesting, different string-vs-object conventions, different correlation mechanisms. A team that writes a weather agent on OpenAI pays a two-day port to Anthropic and another day to Gemini just for the plumbing.

## The Concept

### Shape diffs

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | `tool_calls[]` on assistant message | `content[]` of type `tool_use` | `parts[]` of type `functionCall` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...` | `toolu_...` | UUID (Gemini 3+) |
| Result block | role `tool`, `tool_call_id` | `user` with `tool_result`, `tool_use_id` | `functionResponse` with matching `id` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Strict schema | `strict: true` | schema-is-schema | `responseSchema` at request level |

### Limits

- **OpenAI**: 128 tools, schema depth 5, argument string <= 8192 bytes. Strict mode forbids `$ref`, `oneOf`/`anyOf`/`allOf` overlap.
- **Anthropic**: 64 tools, schema depth practically unbounded. No strict flag; schema is contract.
- **Gemini**: 64 functions, OpenAPI 3.0 subset. Parallel calls with UUIDs since Gemini 3.

### The translator pattern

A canonical tool in your code gets translated to all three provider shapes:

```python
WEATHER = Tool(name="get_weather", description="Use when...",
    input_schema={"type": "object", "properties": {...}, "required": [...]})

def to_openai(tool): return {"type": "function", "function": {"name": tool.name, "parameters": tool.input_schema, "strict": True}}

def to_anthropic(tool): return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}

def to_gemini(tool): return {"functionDeclarations": [{"name": tool.name, "parameters": _gemini_schema(tool.input_schema)}]}
```

### Parsing responses

```python
def parse_openai(resp):
    msg = resp["choices"][0]["message"]
    return [Call(id=tc["id"], name=tc["function"]["name"], args=json.loads(tc["function"]["arguments"]))
            for tc in msg.get("tool_calls", [])]

def parse_anthropic(resp):
    return [Call(id=b["id"], name=b["name"], args=b["input"])
            for b in resp.get("content", []) if b.get("type") == "tool_use"]

def parse_gemini(resp):
    return [Call(id=fc.get("id",""), name=fc["name"], args=fc["args"])
            for p in resp["candidates"][0]["content"].get("parts", []) if "functionCall" in p]
```

## Use It

`code/main.py` defines one canonical `Tool` dataclass, three translators that emit the provider declaration JSON, and parsers that extract a provider-agnostic `Call` object from each response shape.

## Exercises

1. Verify the three declaration JSONs serialize the same underlying `Tool`.
2. Add a `ListToolsResponse` parser for each provider.
3. Implement `tool_choice` conversion: map a canonical `ToolChoice(mode="force", tool_name="x")` to all three shapes.
4. Find one field in each provider's schema spec the other two do not support.
5. Write a test vector violating the declared schema and record which errors fire.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Tool declaration | Name + description + JSON Schema input payload |
| `tool_choice` | Auto / required / none / specific-name modes |
| Strict mode | OpenAI flag constraining decoding to match schema |
| `tool_use` block | Anthropic's inline content block with id, name, input |
| `functionCall` part | Gemini's parts[] entry with name, args, id |
| Arguments-as-string | OpenAI returns args as a JSON string, not object |
| Refusal | Strict-mode-only decline block instead of a call |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/02-function-calling-deep-dive)
