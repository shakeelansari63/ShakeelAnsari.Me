# GloVe, FastText, and Subword Embeddings

Word2Vec left two open questions. GloVe factorized the co-occurrence matrix. FastText embedded the pieces. BPE bridged to transformers.

## The Concept

**GloVe (Global Vectors).** Build the word-word co-occurrence matrix `X` where `X[i][j]` is how often word `j` appears in the context of word `i`. Train vectors such that `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Weight the loss so frequent pairs do not dominate.

**FastText.** A word is the sum of its character n-grams plus the word itself. `where` becomes `<wh, whe, her, ere, re>, <where>`. Unseen words compose from known n-grams.

**BPE (Byte-Pair Encoding).** Start with a vocabulary of individual bytes/characters. Count every adjacent pair. Merge the most frequent pair into a new token. Repeat for `k` iterations. Every sentence tokenizes into something.

## Build It

### GloVe: Factorize the Co-occurrence Matrix

```python
import numpy as np
from collections import Counter

def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts

def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff
            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef
    return W + W_tilde
```

The weighting function `f(x) = (x/x_max)^alpha` downweights very frequent pairs. The final embedding is the sum of `W` (center) and `W_tilde` (context) tables.

### FastText: Subword-aware Embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams

def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

For an unseen word, you still get a vector as long as some of its n-grams are known. `whereupon` shares `<wh`, `her`, `ere` with `where`, so they land near each other.

### BPE: Learned Subword Vocabulary

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)
        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges

def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

Real GPT/BERT/T5 tokenizers learn 30k-100k merges. Any text tokenizes into a bounded-length sequence of known IDs, no OOV ever.

## Use It

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

### When to Pick Which

| Situation | Pick |
|-----------|------|
| Pretrained word vectors, no OOV tolerance needed | GloVe 300d |
| Must handle misspellings / morphologically rich languages | FastText |
| Anything going into a transformer | That model's tokenizer. Never swap. |
| Training your own LM from scratch | Train a BPE or SentencePiece tokenizer first |
| Production text classification with linear model | Still TF-IDF |

## Exercises

1. **Easy.** Run `char_ngrams("playing")` and `char_ngrams("played")`. Compute Jaccard overlap.
2. **Medium.** Extend `learn_bpe` to track vocabulary growth. Plot tokens-per-corpus-character vs merges.
3. **Hard.** Train a 1k-merge BPE on Shakespeare. Compare tokenization of common words vs rare proper nouns.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Co-occurrence matrix | `X[i][j]` = how often word `j` appears near word `i`. |
| Subword | Character n-gram (FastText) or learned token (BPE/WordPiece/SentencePiece). |
| BPE | Iterative merging of most-frequent adjacent pairs until target vocab size. |
| OOV | Word the model has never seen. FastText and BPE handle it. |
| Byte-level BPE | GPT-2's scheme. Vocabulary starts with 256 bytes, nothing is ever OOV. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword)
