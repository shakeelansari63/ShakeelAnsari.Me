# Word Embeddings — Word2Vec from Scratch

A word is the company it keeps. Train a shallow net on that idea and geometry falls out.

TF-IDF knows `dog` and `puppy` are different words. It does not know they mean nearly the same thing. Word2Vec gave us a space where `dog` and `puppy` land close together, where `king - man + woman` lands near `queen`.

## The Concept

**Distributional hypothesis** (Firth, 1957): "You shall know a word by the company it keeps."

Word2Vec has two flavors:

- **Skip-gram.** Given a center word, predict surrounding words. Slower but better for rare words.
- **CBOW.** Given surrounding words, predict the center word.

The network has one hidden layer with no nonlinearity. Input is one-hot. Output is softmax. After training, the hidden layer weights are the embeddings.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
```

Softmax over 100k words is expensive. **Negative sampling** turns it into binary classification: predict "did this context word appear near this center word?" Sample a handful of negative words per training pair.

## Build It

### Step 1: Training Pairs from a Corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

### Step 2: Embedding Tables

```python
import numpy as np

def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

`W` is the center-word embedding table (the one you keep). `W'` is the context-word table (often discarded, sometimes averaged with `W`).

### Step 3: Negative Sampling Objective

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Logistic loss on positive pair (want sigmoid near 1) plus logistic loss on negative pairs (want sigmoid near 0).

### Step 4: Train on a Toy Corpus

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

### Step 5: The Analogy Trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out

def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Not because the model knows royalty. Because `(king - man)` captures something like "royal" and adding it to `woman` lands near royal-female.

## Use It

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv.most_similar("cat", topn=3))
```

For real work, download pre-trained vectors: GloVe (Stanford), fastText (Facebook), or Google News Word2Vec.

### Where Word2Vec Still Wins in 2026

Lightweight domain-specific retrieval, analogy-style feature engineering, interpretability via PCA/t-SNE, on-device inference with no GPU.

### Where Word2Vec Fails

The polysemy wall. `bank` has one vector for `river bank` and `financial bank`. Contextual embeddings (BERT, every transformer since) solved this by producing a different vector per occurrence.

## Exercises

1. **Easy.** Run training on 20 sentences about cats and dogs. Verify `nearest(cat)` returns `dog` in top 3.
2. **Medium.** Add subsampling of frequent words. Measure effect on rare-word similarity.
3. **Hard.** Train on 20 Newsgroups. Compute bias axes `he - she` and `doctor - nurse`. Report which occupations have the largest bias gap.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Word embedding | Dense, low-dim (100-300) representation learned from context. |
| Skip-gram | Predict context words from center word. Better for rare words. |
| Negative sampling | Replace softmax with binary classification against k random words. |
| Static embedding | One vector per word regardless of context. Fails on polysemy. |
| Contextual embedding | Different vector per occurrence based on surrounding context. |
| OOV | Word not seen in training. Word2Vec cannot produce a vector for these. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec)
