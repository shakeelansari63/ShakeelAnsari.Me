# Text Generation Before Transformers — N-gram Language Models

If a word is surprising, the model is bad. Perplexity makes surprise a number. Smoothing keeps it finite.

Before transformers, a language model predicted the next word by counting how often it followed the previous n-1 words. That is an n-gram model. It ran every speech recognizer, spell checker, and phrase-based MT system from 1980 through 2015.

## The Concept

**N-gram probability:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Compute from counts: `P(w | context) = count(context, w) / count(context)`.

**The zero-count problem.** Any n-gram not seen in training gets probability zero. A 4-gram model on Brown has 30% of held-out 4-grams unseen.

**Smoothing approaches (in order of sophistication):**

1. **Laplace (add-one).** Adds 1 to every count. Simple, terrible.
2. **Good-Turing.** Reallocate mass from high-frequency events to unseen ones.
3. **Interpolation.** Combine n-gram, (n-1)-gram estimates with weights.
4. **Backoff.** Fall back to shorter context if n-gram count is zero.
5. **Absolute discounting.** Subtract fixed discount D from all counts.
6. **Kneser-Ney.** Absolute discounting + continuation probability (how many contexts a word appears in).

**Perplexity:** `exp(-(1/N) * Σ log P(w_i | context_i))`. Lower is better. A perplexity of 100 means the model is as confused as choosing uniformly among 100 words.

## Build It

### Step 1: Trigram Counts

```python
from collections import Counter, defaultdict

def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts
```

### Step 2: Laplace Smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

### Step 3: Kneser-Ney (Bigram, Interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Three moving parts: `continuation_prob` (the Kneser-Ney innovation), `lambda_prev` (mass freed by discount), and the final probability as discounted main term plus weighted continuation term.

### Step 4: Generating Text with Sampling

```python
import random

def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

### Step 5: Perplexity

```python
import math

def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

For Brown corpus, a well-tuned 4-gram KN model hits ~140 perplexity. A transformer LM hits 15-30. That 10x gap is why the field moved on.

## Exercises

1. **Easy.** Train a trigram LM on 1,000 Shakespeare sentences. Generate 20 sentences.
2. **Medium.** Implement perplexity for your KN model. Compare against Laplace.
3. **Hard.** Build a trigram spell corrector using LM context probability.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| N-gram | Sequence of n consecutive tokens. |
| Smoothing | Reallocating probability mass so unseen events get non-zero prob. |
| Perplexity | `exp(-average log-prob)` on held-out data. Lower is better. |
| Backoff | If trigram count is zero, use bigram. |
| Kneser-Ney | Absolute discounting + continuation probability. |
| Continuation probability | P(w) weighted by number of contexts w appears in. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer)
