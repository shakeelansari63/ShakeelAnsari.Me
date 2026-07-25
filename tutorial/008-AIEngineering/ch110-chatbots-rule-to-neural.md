# Chatbots — Rule-Based to Neural to LLM Agents

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
