# Dialogue State Tracking

"I want a cheap restaurant in the north... actually make it moderate... and add Italian." Three turns, three state updates. DST keeps the slot-value dict in sync so the booking works.

## The Concept

In task-oriented dialogue, the user's goal is a set of slot-value pairs: `{cuisine: italian, area: north, price: moderate}`. Every turn can add, change, or remove a slot. DST is the hinge between what the user said and what the backend executes.

**Two DST formulations:**

1. **Classification.** For each (slot, candidate_value) pair, predict yes/no. Works for closed-vocab slots.
2. **Generation.** Given the dialogue, generate slot values. Works for open-vocab slots. The modern default.

**Metric.** Joint Goal Accuracy (JGA) — fraction of turns where every slot is correct. All-or-nothing.

**Architectures:**
1. Rule-based (regex + keyword). Strong baseline for narrow domains.
2. TripPy / BERT-DST. Copy-based generation.
3. LDST (LLaMA + LoRA). Instruction-tuned LLM.
4. Prompt + structured output (2024-26). LLM with Pydantic schema + constrained decoding.

## Build It

### Step 1: Rule-Based Slot Extractor

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}

def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

### Step 2: State Update Loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Three invariants: never reset a slot the user did not touch, explicit negation must clear, user correction must overwrite.

### Step 3: LLM-Driven DST with Structured Output

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None

def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic guarantees a valid state object.

### Step 4: JGA Evaluation

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

### Step 5: Handling Correction

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}

def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

The modern pattern: always let the LLM regenerate the whole state from history rather than incrementally updating — this naturally handles corrections.

## Pitfalls

- Full-history regeneration cost is O(n²) tokens. Cap history or summarize.
- Schema drift. Version your schema.
- Case sensitivity. Normalize everywhere.
- Implicit inheritance. Passing full history prevents clearing slots the user didn't touch.
- Free-form vs closed-set. Mix both in the schema.

## Use It

| Situation | Approach |
|-----------|----------|
| Narrow domain | Rule-based + regex |
| Broad domain, labeled data | LDST (LLaMA + LoRA) |
| Broad domain, no labels, prod-ready | LLM + Instructor + Pydantic schema |
| Compliance-sensitive | Rule-based primary, LLM fallback with confirmation flow |

## Exercises

1. **Easy.** Build the rule-based state tracker for 3 slots. Test on 10 hand-crafted dialogues.
2. **Medium.** Same dataset with Instructor + Pydantic + small LLM. Compare JGA.
3. **Hard.** Implement both and route: rule-based primary, LLM fallback when rule-based emits <2 slots. Measure combined JGA and cost.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| DST | Maintain the slot-value dict across dialogue turns. |
| Slot | Named parameter the backend needs (cuisine, date). |
| Domain | The task area — restaurant, hotel, taxi. |
| JGA | Fraction of turns where every slot is correct. All-or-nothing. |
| MultiWOZ | Multi-domain WOZ dataset; standard DST evaluation. |
| Ontology-free DST | Generate slot names and values directly, no fixed list. |
| Correction | Turn that overwrites a previously-filled slot. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking)
