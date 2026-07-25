# Named Entity Recognition

Pull the names out. Sounds easy until you deal with ambiguous boundaries, nested entities, and domain jargon.

NER is the workhorse underneath every structured extraction pipeline: resume parsing, compliance log scanning, medical record anonymization, search query understanding, legal contract extraction.

## The Concept

**BIO tagging** turns entity extraction into a sequence-labeling problem. Label each token with `B-TYPE` (beginning), `I-TYPE` (inside), or `O` (outside).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

The architecture progression: Rule-based → HMM → CRF → BiLSTM-CRF → Transformer-based.

## Build It

### Step 1: BIO Tagging Helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels

def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

### Step 2: Hand-crafted Features

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }

def word_shape(word):
    out = []
    for c in word:
        if c.isupper(): out.append("X")
        elif c.islower(): out.append("x")
        elif c.isdigit(): out.append("d")
        else: out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` returns `xXxxxx`. Capitalization patterns are high-signal for proper nouns.

### Step 3: Rule-based + Dictionary Baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}

def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

### Step 4: CRF with sklearn-crfsuite

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
```

### Step 5: BiLSTM-CRF Sketch

```python
import torch
import torch.nn as nn

class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

## Use It

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

### Hugging Face

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

### LLM-based NER (2026 option)

Zero-shot and few-shot LLM NER is now competitive with fine-tuned models. Start with an LLM zero-shot baseline before collecting training data.

### Where Classical NER Still Wins

Latency under 50ms, thousands of labeled examples needing 98%+ F1, regulatory constraints requiring on-prem non-generative models.

### Where It Falls Apart

Domain shift, nested entities, long entities, sparse entity types.

## Exercises

1. **Easy.** Implement `bio_to_spans` and verify round-trip consistency.
2. **Medium.** Train sklearn-crfsuite CRF on CoNLL-2003. Report per-entity F1.
3. **Hard.** Fine-tune `distilbert-base-cased` on a domain-specific NER dataset.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| NER | Label token spans with types (PERSON, ORG, GPE, DATE...). |
| BIO | `B-X` begins, `I-X` continues, `O` outside. |
| BILOU | Adds `L-X` (last), `U-X` (unit) for cleaner boundaries. |
| CRF | Models transitions between labels, not just emissions. |
| Nested NER | Overlapping entities. BIO cannot express this. |
| Entity-level F1 | Predicted span must match true span exactly. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition)
