# Structured Outputs & Constrained Decoding

Ask an LLM for JSON. Get JSON most of the time. In production, "most" is the problem. Constrained decoding turns "most" into "always" by editing the logits before sampling.

## The Concept

Three layers exist in 2026:

1. **Prompting.** Ask nicely. Works ~80% on frontier models, less on smaller ones.
2. **Native structured output APIs.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode.
3. **Constrained decoding.** Modify the logits so the model cannot emit invalid tokens. 100% valid by construction.

**How constrained decoding works.** At each generation step, the LLM produces logits over the full vocabulary. A logit processor computes which tokens are valid given the current position in the target grammar and sets invalid token logits to -inf.

Implementations: **Outlines** (FSM-based), **XGrammar** (CFG-based), **vLLM guided decoding**, **Instructor** (Pydantic + retries).

The counterintuitive result: constrained decoding is often faster than unconstrained because it shrinks the search space.

### The Pitfall That Costs You

Field order matters. Put `answer` before `reasoning`, and the model commits to an answer before it thinks. Schema field order is logic, not formatting.

```json
// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

## Build It

### Step 1: Regex-Constrained Generation from Scratch

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask

def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

### Step 2: Outlines for JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str

model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Zero validation errors. The FSM makes invalid output unreachable.

### Step 3: Instructor for Provider-Agnostic Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]

client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Instructor formats the schema into the prompt, parses the output, and retries on validation failure (default 3 times).

### Step 4: Native Vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                   "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

## Pitfalls

- **Recursive schemas.** Outlines flattens recursion. Tree-structured outputs need XGrammar.
- **Huge enums.** Switch to a retriever-predict-top-k approach.
- **Grammar too strict.** Force `date: "YYYY-MM-DD"` and the model cannot output "unknown". Allow null.
- **Premature commitment.** Put reasoning first, answer last.

## Use It

| Situation | Pick |
|-----------|------|
| OpenAI/Anthropic model, simple schema | Native vendor structured output |
| Any provider, can tolerate retries | Instructor |
| Local model, need 100% validity | Outlines (FSM) |
| Local model, recursive schema | XGrammar or llguidance |

## Exercises

1. **Easy.** Prompt a small model without constrained decoding for `Review`. Measure JSON validity rate.
2. **Medium.** Same corpus with Outlines JSON mode. Compare compliance rate, latency, and accuracy.
3. **Hard.** Implement a regex-constrained decoder from scratch for phone numbers. Verify 0 invalid outputs.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Constrained decoding | Mask invalid-token logits at every generation step. |
| Logit processor | Function: `(logits, state) -> masked_logits`. |
| FSM | Compiled grammar representation; O(1) valid-next-token lookup. |
| CFG | Context-free grammar; handles recursion. |
| Schema field order | First field commits; put reasoning before answer. |
| JSON mode | Guarantees JSON syntax; does NOT guarantee schema match. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding)
