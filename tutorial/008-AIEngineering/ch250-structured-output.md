# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

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
