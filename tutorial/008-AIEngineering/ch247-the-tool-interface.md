# The Tool Interface — Why Agents Need Structured I/O

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
