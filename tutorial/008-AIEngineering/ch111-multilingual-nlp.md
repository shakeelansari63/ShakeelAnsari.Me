# Multilingual NLP

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
