# Chatbots, Multilingual, Dialogue & Structured Outputs

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch110): Chatbots — Rule-Based to Neural to LLM Agents

ELIZA replied with pattern matches. DialogFlow mapped intents. GPT answered from weights. Claude runs tools and verifies. Each era solved the previous one's worst failure.

## The Concept

**Rule-based (ELIZA, AIML, DialogFlow).** Hand-authored patterns match user input and produce responses. Intent classifiers route to predefined flows. Works brilliantly in narrow scope. Fails outside it.

**Retrieval-based.** Encode every (utterance, response) pair. At runtime, retrieve the nearest stored response. No generation, no hallucination.

**Neural (seq2seq).** Encoder-decoder trained on conversation logs. Generates responses from scratch. Fluent but prone to generic outputs and factual drift.

**LLM agents.** A language model wrapped in a loop that plans, calls tools, and verifies outcomes. The 2026 architecture.

The four paradigms are not sequential replacements. A 2026 production chatbot routes through all four.

## Build It

### Step 1: Rule-Based Pattern Matching

```python
import re

class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template

PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]

def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

### Step 2: Retrieval-Based (FAQ)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)

def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

### Step 3: LLM Agent Loop

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

### Step 4: Hybrid Routing

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)

def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

## Failure Modes

- **Confident fabrication.** LLM claims it completed an action it did not.
- **Prompt injection.** User inserts text overriding the system prompt. Ranked LLM01 in OWASP Top 10 for LLM Applications 2025.
- **Scope creep.** Agent goes off-task from tangentially related tool returns.
- **Infinite loops.** Agent keeps calling the same tool.
- **Context window exhaustion.** Long conversations push early turns out of context.

## Use It

| Use case | Architecture |
|---------|---------------|
| Booking, payment, authentication | Rule-based state machines + slot filling |
| Customer support FAQs | Retrieval over curated answers |
| Open-ended help chat | LLM agent with RAG + tool calls |

## Exercises

1. **Easy.** Implement rule-based respond with 10 patterns for a coffee-shop ordering bot.
2. **Medium.** Build hybrid FAQ + LLM fallback with 50 canned FAQ entries.
3. **Hard.** Implement the agent loop with three tools. Evaluate on 50 test scenarios including prompt injection attempts.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Intent | Categorical label (book_flight, reset_password). |
| Slot | Parameter the bot needs (date, destination). |
| RAG | Retrieve relevant docs, then ground the LLM's response. |
| Tool call | LLM emits structured call with name + args. |
| Agent loop | Controller running LLM calls interleaved with tool calls. |
| Prompt injection | Malicious input overriding the system prompt. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural)

---

## Part 2 (ch111): Multilingual NLP

One model, 100+ languages, zero training data for most of them. Cross-lingual transfer is the practical miracle of the 2020s.

## The Concept

**Shared vocabulary.** Multilingual models use a SentencePiece or WordPiece tokenizer trained on text from all target languages. The same subword unit represents the same morpheme across related languages.

**Shared representation.** A transformer pretrained on masked language modeling across many languages learns that semantically similar sentences in different languages produce similar hidden states.

**Zero-shot transfer.** Fine-tune on labeled data in one language (usually English). Run on any other language the model supports. Strong for typologically related languages, weaker for distant ones.

**Few-shot fine-tuning.** Add 100-500 labeled examples in the target language. Accuracy jumps to 95-98% of the English baseline.

## The Models

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | First practical multilingual LM. |
| XLM-R | 2019 | 100 languages | Trained on CommonCrawl. Cross-lingual baseline. |
| NLLB-200 | 2022 | 200 languages | Meta's translation model. |
| Aya-23 | 2024 | 23 languages | Cohere's multilingual LLM. |

## The Source-Language Decision (2026 research)

Most teams default to English as the fine-tuning source. Recent research shows this is often wrong. Language similarity predicts transfer quality better than raw corpus size. For Slavic targets, German or Russian may beat English. For Indic targets, Hindi may beat English.

## Build It

### Step 1: Zero-shot Cross-lingual Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))

print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
```

One model, multiple languages, same API.

### Step 2: Multilingual Embedding Space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Translations land close in embedding space. A different English sentence lands further.

### Step 3: Few-Shot Fine-Tuning Strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset

def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)
    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out
    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

For 100-500 target examples, `num_train_epochs=5` and `learning_rate=2e-5` are safe defaults.

## Evaluation

- Per-language accuracy on held-out sets. Not aggregated.
- Benchmark against monolingual baseline.
- Cross-lingual consistency: same meaning in two languages should produce the same prediction.

### The Tokenization Tax

Low-resource languages tokenize into far more tokens per word. That 3-5x eats your context window, training efficiency, and latency. Mitigations: pick a tokenizer with good coverage (XLM-V's 1M vocab), verify tokenization fertility, use byte-level fallback.

## Use It

| Task | Recommended |
|-----|-------------|
| Classification, 100 languages | XLM-R-base (~270M) fine-tuned |
| Multilingual sentence embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M` |

## Exercises

1. **Easy.** Run zero-shot classification on 10 sentences per language across English, French, Hindi, Arabic.
2. **Medium.** Build a cross-lingual retriever over a small mixed-language corpus.
3. **Hard.** Compare English-source and Hindi-source fine-tuning for a Hindi classification task.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Multilingual model | Shared vocabulary and parameters across languages. |
| Cross-lingual transfer | Fine-tune on source, evaluate on target without target labels. |
| Zero-shot | Transfer without fine-tuning on the target language. |
| Few-shot | 100-500 target-language examples used for fine-tuning. |
| XLM-R | 100-language RoBERTa pretrained on CommonCrawl. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/18-multilingual-nlp)

---

## Part 3 (ch113): Structured Outputs & Constrained Decoding

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

---

## Part 4 (ch122): Dialogue State Tracking

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
