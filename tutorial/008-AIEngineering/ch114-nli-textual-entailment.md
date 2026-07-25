# Natural Language Inference — Textual Entailment

"t entails h" means a human reading t would conclude h is true. NLI is the task of predicting entailment / contradiction / neutral. Boring on the surface, load-bearing in production.

## The Problem

You built a summarizer. How do you know the summary does not contain a hallucination? You built a chatbot. How do you know the answer is supported by the retrieved passage? All three problems reduce to NLI:

- **Hallucination check:** premise = source, hypothesis = summary claim. Not entailment = hallucination.
- **Grounded QA:** premise = retrieved passage, hypothesis = answer. Not entailment = fabrication.
- **Zero-shot classification:** premise = document, hypothesis = verbalized label ("This is about sports"). Entailment = predicted label.

## The Concept

**The three labels:**
- **Entailment.** "The cat is on the mat" entails "There is a cat."
- **Contradiction.** "The cat is on the mat" contradicts "There is no cat."
- **Neutral.** No inference either way.

**Not logical entailment.** NLI is what a typical human reader would infer, not strict logic. "John walked his dog" entails "John has a dog" in NLI.

**Datasets:** SNLI (570k pairs), MultiNLI (433k pairs, 10 genres), ANLI (adversarial), DocNLI (document-length).

**The architecture.** A transformer encoder reads `[CLS] premise [SEP] hypothesis [SEP]`. The `[CLS]` representation feeds a 3-way softmax.

## Build It

### Step 1: Run a Pretrained NLI Model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

### Step 2: Zero-shot Classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

No training data required. Customize `hypothesis_template` if needed.

### Step 3: Faithfulness Check for RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

This is the core of RAGAS faithfulness. Decompose the answer into atomic claims and check each.

## Pitfalls

- **Hypothesis-only shortcuts.** Models can predict label from hypothesis alone at ~60% on SNLI.
- **Lexical overlap heuristic.** Subsequence heuristic passes SNLI but fails HANS/ANLI.
- **Document-length degradation.** Single-sentence NLI models drop 20+ F1 on long premises.
- **Zero-shot template sensitivity.** Tune the hypothesis template.
- **Domain mismatch.** MNLI trains on general English. Legal/medical need domain-specific NLI.

## Use It

| Use case | Model |
|---------|-------|
| General-purpose NLI | `microsoft/deberta-v3-large-mnli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |

The 2026 meta-pattern: NLI is the duct tape of text understanding. Whenever you need "does A support B?" — reach for NLI before another LLM call.

## Exercises

1. **Easy.** Run BART-MNLI on 20 hand-crafted triples. Add adversarial subsequence-heuristic traps.
2. **Medium.** Compare zero-shot templates on 100 AG News headlines. Report accuracy swing.
3. **Hard.** Build a RAG faithfulness checker: atomic-claim decomposition + NLI per claim. Evaluate on 50 examples.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| NLI | 3-way classification of premise-hypothesis relationship. |
| RTE | Older name for NLI. Same task. |
| Entailment | Reader would conclude h is true given t. |
| Contradiction | Reader would conclude h is false given t. |
| Neutral | No inference either way. |
| Zero-shot classification | Verbalize labels as hypotheses, pick max entailment. |
| Faithfulness | Is the answer supported by context? NLI over (context, answer). |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment)
