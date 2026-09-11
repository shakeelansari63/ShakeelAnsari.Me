# Tool Interface, Function Calling & Schemas

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch247): The Tool Interface — Why Agents Need Structured I/O

> A language model produces tokens. A program takes actions. The gap between those two is the tool interface: a contract that lets the model request an action and the host execute it. Every 2026 stack — function calling on OpenAI, Anthropic, and Gemini; MCP's `tools/call`; A2A's task parts — is a different encoding of the same four-step loop.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11
**Time:** ~45 minutes

## Learning Objectives
- Explain why an LLM that can only generate text cannot take actions against the real world
- Draw the four-step tool-call loop (describe → decide → execute → observe) and name who owns each step
- Write a tool description as three parts: name, JSON Schema input, and a deterministic executor function
- Distinguish pure and side-effecting tools and state why the split matters for safety

## The Problem

An LLM emits a probability distribution over the next token. That is the entire output surface. If you ask "what is the weather in Bengaluru right now," it can write a plausible sentence, but it cannot dial into a weather API.

The host program advertises callable tools to the model. The model emits a structured payload naming a tool and its arguments. The host parses that payload, runs the tool, and feeds the result back. The loop continues until the model decides no more calls are needed.

## The Concept

### Step one: describe

The host declares each tool with three fields: **Name** (stable machine-readable identifier), **Description** (one-paragraph brief), and **Input schema** (JSON Schema 2020-12).

```python
@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    executor: Callable[[dict], Any]
    consequential: bool = False
```

### Step two: decide

The model chooses: answer directly, call one or more tools, or refuse. A tool call payload has three stable fields: a call `id`, a tool `name`, and a JSON `arguments` object.

### Step three: execute

The host validates arguments against the schema and runs the executor.

```python
def validate(schema: dict, value: Any) -> list[str]:
    errors: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"expected object, got {type(value).__name__}"]
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"missing required field '{field}'")
        ...
```

### Step four: observe

The host appends the tool result as a `tool` role message and re-invokes the model.

### The trust split

Tools come in two flavors: **Pure** (read-only, no side effects — `get_weather`, `search_docs`) and **Consequential** (mutates state — `send_email`, `delete_file`). Meta's 2026 "Rule of Two" says a single turn may combine at most two of: untrusted input, sensitive data, consequential action.

### Where the loop lives

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |

### Circuit breakers

Production hosts set max turns between 5 and 20. Claude Code defaults to 20; OpenAI Assistants to 10. Never ship without a bound.

## Use It

`code/main.py` runs the four-step loop without an LLM. A fake "decider" function simulates the model by keyword matching; the executor, schema validator, and observe-step harness are real.

```python
REGISTRY: list[Tool] = [
    Tool(name="add", description="Use when the user asks for the sum of two numbers.",
         input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
         "required": ["a", "b"]}, executor=tool_add),
    Tool(name="get_time", ...),
    Tool(name="get_weather", ...),
]
```

A TypeScript equivalent (`code/main.ts`) mirrors the same shape.

## Exercises

1. Add a `get_stock_price(ticker)` tool to the harness and confirm the fake decider routes ticker queries to it.
2. Break the schema validator: pass a call missing a required field, then an extra unknown field. Decide: reject or ignore?
3. Classify each tool as pure or consequential. Add a confirmation gate for consequential tools.
4. Draw the four-step loop with the provider-column table filled for your favorite client.
5. Read OpenAI's function-calling guide and identify the one convenience field that sits outside the four-step loop.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Tool | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | Provider-level API support for emitting structured tool calls |
| Tool call | JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | Executor's output wrapped in a `tool` role message |
| Parallel tool calls | Multiple call objects in one model turn |
| Strict mode | Constrained decoding that forces output to validate against the schema |
| Pure tool | No side effects; safe to re-run |
| Consequential tool | Mutates external state; requires gate or user confirmation |
| Four-step loop | describe → decide → execute → observe |
| Host | The program that holds the registry, calls the model, runs the executor |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/01-the-tool-interface)

---

## Part 2 (ch248): Function Calling Deep Dive — OpenAI, Anthropic, Gemini

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

---

## Part 3 (ch249): Parallel Tool Calls and Streaming with Tools

> Three independent weather lookups serialized is three round trips. Run them in parallel and total time collapses to the slowest single call. Every frontier provider now emits multiple tool calls in a single turn. The payoff is real; the plumbing is subtle.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02
**Time:** ~75 minutes

## Learning Objectives
- Explain why `parallel_tool_calls: true` exists and when to disable it
- Correlate streamed argument chunks to the right tool-call id during parallel fan-out
- Reassemble partial `arguments` strings into complete JSON without parsing early
- Run a three-city weather benchmark demonstrating sequential vs parallel latency

## The Problem

Without parallel calls, three weather lookups cost three LLM round trips plus executor latency. With parallel calls: one LLM round trip, executor time is max of three not sum. Production benchmarks show 60-70% wall-clock reduction.

## The Concept

### Enabling parallel

- **OpenAI**: `parallel_tool_calls: true` (default). `false` for serial.
- **Anthropic**: `disable_parallel_tool_use: false` (default on Claude 3.5+).
- **Gemini**: Always parallel-capable.

Disable when tools have ordering dependencies (`create_file` then `write_file`) or when rate limiters cannot handle fan-out.

### Id correlation

Every call has an `id`. Every result must echo the same id. Without it, results are ambiguous.

### Running calls concurrently

```python
def run_parallel(cities):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cities)) as pool:
        results = list(pool.map(executor_weather, cities))
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results
```

### Streaming tool calls — the accumulator pattern

Arguments arrive in pieces. Three parallel call streams interleave on the wire. You need one accumulator per id.

```python
@dataclass
class CallBuffer:
    id: str; name: str = ""; args_buf: str = ""; done: bool = False

@dataclass
class StreamAccumulator:
    buffers: dict[str, CallBuffer] = field(default_factory=dict)
    def on_event(self, event):
        kind = event["type"]; idx = event.get("id")
        if kind == "call_start":
            self.buffers[idx] = CallBuffer(id=idx, name=event["name"])
        elif kind == "args_delta":
            self.buffers[idx].args_buf += event["chunk"]
        elif kind == "call_stop":
            self.buffers[idx].done = True
            completed.append(self.buffers[idx])
        return completed
```

### The parse-early trap

Partial JSON like `{"city": "Beng` is not valid. Wait for the provider's end-of-call signal (OpenAI's `finish_reason="tool_calls"`, Anthropic's `content_block_stop`) before parsing.

### Benchmark

Three executors with 400, 600, 800 ms latency: sequential = 1800 ms, parallel = 800 ms. The savings grow with tool count.

## Use It

`code/main.py` has two halves: sequential vs parallel weather calls with wall-clock timing, and a fake streaming response reassembly using `StreamAccumulator`.

```python
seq_ms, _ = run_sequential(cities)
par_ms, _ = run_parallel(cities)
print(f"speedup: {seq_ms/par_ms:.2f}x")
```

## Exercises

1. Vary simulated latencies and confirm the parallel-to-sequential ratio approximates `max/sum`.
2. Extend the accumulator to handle a "call was cancelled mid-stream" case.
3. Replace the thread pool with `asyncio.gather` and benchmark.
4. Add an `ordering_dependency` graph to gate parallel fan-out.
5. Identify the real-world tool type where Anthropic recommends disabling parallelism.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Parallel tool calls | Multiple tool calls in one assistant message |
| `disable_parallel_tool_use` | Anthropic's opt-out flag |
| Accumulator | Per-id string buffer for partial arguments chunks |
| Out-of-order completion | Parallel calls finish in unpredictable order; ids are the glue |
| Dependency graph | Tools whose outputs feed into inputs of other tools |
| Parse-early trap | Attempting to parse an incomplete arguments string |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls)

---

## Part 4 (ch250): Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

> "Ask the model nicely to return JSON" fails 5 to 15 percent of the time, even on frontier models. Structured outputs close that gap with constrained decoding: the model is literally prevented from emitting a token that would violate the schema.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02
**Time:** ~75 minutes

## Learning Objectives
- Write a JSON Schema 2020-12 for an extraction target using the right constraints
- Explain why strict mode and constrained decoding give different guarantees from "validate after generation"
- Distinguish the three failure modes: parse error, schema violation, model refusal
- Ship an extraction pipeline with typed repair and typed refusal handling

## The Problem

An agent reading a purchase-order email needs to turn free text into `{customer, line_items, total_usd}`. Three approaches: prompt for JSON (85-95% reliable), validate after generation (reliable but expensive retries), or constrained decoding (guaranteed valid at decode time).

## The Concept

### JSON Schema 2020-12 — the lingua franca

Every provider accepts JSON Schema. Key constructs: `type`, `properties`, `required`, `enum`, `minimum`/`maximum`, `minLength`/`maxLength`/`pattern`, `items`, `additionalProperties: false`.

OpenAI strict mode adds three requirements: every property in `required`, `additionalProperties: false` everywhere, no unresolved `$ref`.

### The validator

```python
INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string", "minLength": 1, "maxLength": 200},
        "line_items": {"type": "array", "items": {
            "type": "object", "properties": {
                "sku": {"type": "string", "pattern": "^[A-Z0-9-]+$"},
                "qty": {"type": "integer", "minimum": 1},
                "unit_usd": {"type": "number", "minimum": 0}},
            "required": ["sku", "qty", "unit_usd"], "additionalProperties": False}},
        "total_usd": {"type": "number", "minimum": 0},
        "currency": {"type": "string", "enum": ["USD", "EUR", "INR"]}},
    "required": ["customer", "line_items", "total_usd", "currency"],
    "additionalProperties": False}
```

### The three failure modes

1. **Parse error** — not valid JSON. Impossible under strict mode.
2. **Schema violation** — parsed but violates schema. Impossible under strict mode.
3. **Refusal** — model declines. Must be handled as typed outcome.

```python
def process_model_output(raw, schema):
    if raw.startswith("__REFUSAL__"):
        return ParsedResult("refusal", raw.removeprefix("__REFUSAL__").strip(), [])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return ParsedResult("parse_error", None, [ValidationError("$", str(e))])
    errs = validate(schema, parsed)
    if errs: return ParsedResult("violation", parsed, errs)
    return ParsedResult("ok", parsed, [])
```

### Constrained decoding techniques

Open-weights approaches: grammar-based decoding (outlines, guidance — build a DFA from the schema and mask logits), logit masking with a streaming JSON parser, speculative decoding with a verifier. Commercial providers use these behind the scenes.

### Retry strategy

Outside strict mode: generate → parse → validate → if fail, inject error and retry, max 3x.

## Use It

`code/main.py` ships a minimal JSON Schema 2020-12 validator and runs it on five test cases: happy path, parse error, schema violation (extra field + bad SKU), schema violation (missing required field), and refusal.

## Exercises

1. Add a test case with negative `total_usd` and confirm the validator rejects it.
2. Extend the validator to support `oneOf` with a discriminator.
3. Write the same Invoice schema as a Pydantic BaseModel and compare `model_json_schema()` output.
4. Measure refusal rates: construct ten non-extractable inputs and run through a real provider with strict mode.
5. Identify the one construct OpenAI strict mode forbids that plain JSON Schema allows.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| JSON Schema 2020-12 | IETF-draft schema dialect every modern provider speaks |
| Strict mode | OpenAI flag enforcing schema via constrained decoding |
| Constrained decoding | Decode-time enforcement masking invalid next-tokens |
| Refusal | Typed outcome when input cannot fit the schema |
| Parse error | Output did not parse as JSON; impossible under strict |
| Schema violation | Parsed but violated types / required / enum / range |
| Grammar enforcement | FSM-based logit masking (outlines / guidance) |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/04-structured-output)

---

## Part 5 (ch251): Tool Schema Design — Naming, Descriptions, Parameter Constraints

> A correct tool fails silently when the model cannot tell when to use it. Naming, descriptions, and parameter shapes drive 10 to 20 percentage-point swings in tool-selection accuracy.

**Type:** Learn
**Languages:** Python (stdlib, tool schema linter)
**Prerequisites:** Phase 13 · 01, Phase 13 · 04
**Time:** ~45 minutes

## Learning Objectives
- Write a tool description using the "Use when X. Do not use for Y." pattern
- Name tools in stable `snake_case` across a large registry
- Choose between atomic tools and monolithic tools
- Run a tool-schema linter against a registry and fix the findings

## The Problem

An agent with 30 tools: wrong tool picked because descriptions are ambiguous, or no tool picked when one fits. Composio's 2025 field guide measured 10-20 percentage-point accuracy swings from renaming and rewriting descriptions alone.

## The Concept

### Naming rules

1. `snake_case` — every provider's tokenizer handles it cleanly.
2. Verb-noun order: `get_weather`, not `weather_get`.
3. No tense markers: `get_weather`, not `got_weather`.
4. Stable: renaming is breaking. Version by adding new names.
5. Namespace prefixes: `notes_list`, `notes_search`, `notes_create`.
6. No arguments in the name: `get_weather_for_city(city)`, not `get_weather_in_tokyo()`.

### Description pattern

```
Use when {condition}. Do not use for {close-but-wrong-cases}.
```

Stay under 1024 characters. Include format hints for parameters.

### Atomic vs monolithic

```python
# Monolithic (bad): forces the model to pick action from strings
do_everything(action: str, target: str, options: dict)

# Atomic (good): each has a tight description and typed schema
notes_list()
notes_create(title, body)
notes_delete(note_id)
```

Benchmarks show 15-30% worse selection on monolithic tools. Rule of thumb: if `action` has more than 3 values, split the tool.

### The linter

```python
def lint_name(name):
    findings = []
    if not SNAKE_CASE.match(name):
        findings.append(Finding("block", name, "name must be snake_case"))
    if any(m in name for m in TENSE_MARKERS):
        findings.append(Finding("warn", name, "name includes tense marker"))
    ...

def lint_description(desc, tool_name):
    findings = []
    if len(desc) < 40: findings.append(Finding("block", tool_name, "description under 40 chars"))
    if "use when" not in desc.lower(): findings.append(Finding("warn", tool_name, "missing 'Use when'"))
    ...
```

### Error messages as teaching signals

BAD: `TypeError: object of type 'NoneType' has no attribute 'lower'`
GOOD: `Invalid input: 'city' is required. Example: {"city": "Bengaluru"}.`

Typed error messages cut retry counts in half on weak models.

### Tool poisoning prevention

Descriptions land in the model's context verbatim. The linter rejects descriptions containing indirect-injection keywords: `<SYSTEM>`, `ignore previous`, URL shorteners.

## Use It

`code/main.py` ships a tool-schema linter that audits a registry against the design rules. Run on `GOOD_REGISTRY` (passes) and `BAD_REGISTRY` (fails on every rule).

## Exercises

1. Rewrite `BAD_REGISTRY` to pass the linter.
2. Design an MCP server for a notes app with atomic tools; lint to zero findings.
3. Pick a popular MCP server and find two actionable improvements in its tool descriptions.
4. Add the linter to your CI, failing on severity `block`.
5. Read Composio's field guide and add one uncovered rule to the linter.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Tool description | Natural-language brief the model reads during selection |
| Atomic tool | One tool one action; name uniquely identifies behavior |
| Monolithic tool | Single tool with `action` string; selection accuracy tanks |
| Enum-closed set | `{type: "string", enum: [...]}` for closed domains |
| Tool poisoning | Hidden instructions in tool descriptions that hijack the agent |
| Tool-selection accuracy | Percentage of queries where the model picks correctly |
| Namespace prefix | `notes_*` — shared prefix grouping related tools |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/05-tool-schema-design)
