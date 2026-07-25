# BERT — Masked Language Modeling

> GPT predicts the next word. BERT predicts a missing word. One sentence of difference — and half a decade of everything embedding-shaped.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## The Problem

In 2018 every NLP task — sentiment, NER, QA, entailment — trained its own model from scratch on its own labeled data. There was no pre-trained "understand English" checkpoint you could fine-tune. ELMo (2018) showed you could pre-train contextual embeddings with a bidirectional LSTM; it helped but did not generalize.

BERT (Devlin et al. 2018) asked: what if we took a transformer encoder, trained it on every sentence on the internet, and forced it to predict missing words from context on both sides? Then you fine-tune one head on your downstream task. Parameter efficiency was a revelation.

Within 18 months BERT and its variants (RoBERTa, ALBERT, ELECTRA) dominated every NLP leaderboard. In 2026 encoder-only models are still the right tool for classification, retrieval, and structured extraction — they run 5–10× faster per token than decoders. ModernBERT (Dec 2024) pushed the architecture to 8K context with Flash Attention + RoPE + GeGLU.

## The Concept

### The training signal

Take a sentence: `the quick brown fox jumps over the lazy dog`.

Mask 15% of tokens randomly:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Train the model to predict the original tokens at masked positions. Because the encoder is bidirectional, predicting `[MASK]` at position 1 can use `brown fox jumps` at positions 2+. That is the thing GPT cannot do.

### The BERT mask rules

Of the 15% of tokens selected for prediction:

- 80% are replaced with `[MASK]`.
- 10% are replaced with a random token.
- 10% are left unchanged.

Why not always `[MASK]`? Because `[MASK]` never appears at inference time. The 10% random + 10% unchanged keeps the model honest.

### What changed in 2026: ModernBERT

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

### Use cases that still pick an encoder in 2026

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better embedding quality per token |
| Classification (sentiment, intent, toxicity) | One forward pass; no generation overhead |
| NER / token labeling | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classifier head on top of encoder |
| Reranker for RAG | Cross-encoder scoring, 10x faster than LLM rerankers |

## Build It

### Step 1: masking logic

```python
MASK_ID = 0
IGNORE_INDEX = -100

def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [IGNORE_INDEX] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                rand_id = rng.randrange(vocab_size)
                input_ids[i] = rand_id
    return input_ids, labels
```

### Step 2: distribution check

```python
def distribution_check(n_tokens, vocab_size, mask_prob=0.15, seed=42):
    rng = random.Random(seed)
    tokens = [rng.randrange(3, vocab_size) for _ in range(n_tokens)]
    input_ids, labels = create_mlm_batch(tokens, vocab_size, mask_prob, rng)
    selected = sum(1 for l in labels if l != IGNORE_INDEX)
    masked = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t == MASK_ID)
    randomized = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t != MASK_ID and t != l)
    unchanged = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t == l)
    return selected, masked, randomized, unchanged
```

Training on 100,000 tokens should show ~15% selected, ~80% masked, ~10% random, ~10% unchanged.

### Step 3: compare mask types

Show how the three-way rule keeps the model usable without `[MASK]`. Predict on an unmasked sentence and on a masked sentence.

### Step 4: fine-tune head

Replace the MLM head with a classification head on a toy sentiment dataset. Only the head trains; the encoder is frozen.

## Use It

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models are fine-tuned BERT.** `sentence-transformers` models like `all-MiniLM-L6-v2` are BERTs trained with contrastive loss.

**Cross-encoder rerankers are also fine-tuned BERT.** Pair-classification on `[CLS] query [SEP] doc [SEP]`.

## Ship It

See `outputs/skill-bert-finetuner.md`. The skill scopes a BERT fine-tune for a new classification or extraction task.

## Exercises

1. **Easy.** Run the masking code and print the mask distribution across 10,000 tokens. Confirm ~15% are selected.
2. **Medium.** Implement whole-word masking: if a word is tokenized into subwords, mask all subwords together.
3. **Hard.** Train a tiny (2-layer, d=64) BERT on 10,000 sentences. Fine-tune for SST-2 sentiment. Compare against a decoder-only baseline.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Randomly replace 15% of tokens with `[MASK]`, predict originals |
| Bidirectional | "Looks both ways" | Encoder attention has no causal mask |
| `[CLS]` | "The pooler token" | Prepended special token; its final embedding is the sentence representation |
| `[SEP]` | "Segment separator" | Separates paired sequences |
| Fine-tuning | "Adapt to a task" | Keep the encoder frozen; train a small head on top |
| Cross-encoder | "A reranker" | BERT that takes both query and doc as input |
| ModernBERT | "2024 refresh" | Encoder rebuilt with RoPE, RMSNorm, GeGLU, alternating attention |

## Further Reading

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators](https://arxiv.org/abs/2003.10555)
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/06-bert-masked-language-modeling)
