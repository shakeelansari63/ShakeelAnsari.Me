# Text Representation & Subword Tokenization

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch094): Text Processing — Tokenization, Stemming, Lemmatization

Language is continuous. Models are discrete. Preprocessing is the bridge.

A model cannot read "The cats were running." It reads integers. Every NLP system opens with three questions: where does a word start, what is the root of the word, and how do we treat "run", "running", "ran" as the same thing when it helps but different when it doesn't.

## The Three Operations

**Tokenization** splits a string into tokens. Word-level for classical NLP, subword for transformers, character for languages without whitespace.

**Stemming** chops suffixes with rules. Fast, aggressive, dumb. `running` → `run`. `organization` → `organ`. Hard failure mode on the second example.

**Lemmatization** reduces a word to its dictionary form using grammar knowledge. Slower, accurate, needs a lookup table. `ran` → `run` (needs to know "ran" is past tense). `better` → `good` (needs to know comparative forms).

Rule of thumb: stem when speed matters and you can tolerate noise (search indexing, rough classification). Lemmatize when meaning matters (question answering, semantic search, anything the user will read).

## Build It

### Regex Word Tokenizer

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Three patterns: words with optional apostrophe (`don't`, `it's`), pure numbers, single non-whitespace non-alphanumeric characters (punctuation).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Failure modes: `3pm` splits to `['3', 'pm']`. URLs, emails, hashtags all break. For production, add patterns before the general ones.

### Porter Stemmer (Step 1a)

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Rules compete. Earlier rules win. Order matters more than any single rule.

### Lookup-based Lemmatizer

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

`watched` is not in our table. Real lemmatization covers `ed`, irregular verbs, comparative adjectives. This is why production uses WordNet, spaCy, or a full morphological analyzer.

### Pipe Them Together

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

## Use It

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)

def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"): return "v"
    if tag.startswith("J"): return "a"
    if tag.startswith("R"): return "r"
    return "n"

lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` handles contractions and unicode. `PorterStemmer` runs all five phases. `WordNetLemmatizer` needs the POS tag translated from Penn Treebank to WordNet abbreviations.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy hides the whole pipeline behind `nlp(text)`. Tokenization, POS tagging, and lemmatization all run. Faster than NLTK at scale.

### When to Pick Which

| Situation | Pick |
|-----------|------|
| Teaching, research, swapping components | NLTK |
| Production, multi-language, speed matters | spaCy |
| Transformer pipeline | Use `tokenizers` / `transformers` and skip classical preprocessing |

## Two Failure Modes Nobody Warns About

**Reproducibility drift.** NLTK and spaCy change behavior between versions. Pin library versions. Write a preprocessing regression test.

**Training / inference mismatch.** Train with aggressive preprocessing, deploy on raw user input, watch performance crater. Ship preprocessing as a function inside the model package, not as a notebook cell.

## Exercises

1. **Easy.** Extend `tokenize` to keep URLs as single tokens.
2. **Medium.** Implement Porter step 1b with the double-consonant rule.
3. **Hard.** Build a lemmatizer using WordNet with Porter fallback. Measure accuracy against plain WordNet and plain Porter.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Token | Whatever unit the model consumes. Word, subword, character, or byte. |
| Stem | Result of rule-based suffix stripping. Not always a real word. |
| Lemma | Dictionary form. Requires grammatical context. |
| POS tag | Grammatical category like NOUN, VERB, ADJ. Needed for lemmatization. |
| Morphology | How a word changes form based on tense, number, case. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/01-text-processing)

---

## Part 2 (ch095): Bag of Words, TF-IDF, and Text Representation

Count first, think later. TF-IDF still beats embeddings on well-defined tasks in 2026.

Every NLP pipeline needs to turn a variable-length stream of tokens into a fixed-size vector. The first answer: count the words. Make a vector. That vector has carried more production NLP than any embedding model — spam filters, topic classifiers, log anomaly detection, search ranking.

## The Concept

**Bag of Words (BoW)** throws away order. For each document, count how many times each vocabulary word appears. Vector length is the vocabulary size.

**TF-IDF** reweights BoW. A word in every document is uninformative, so scale it down. A word rare across the corpus but frequent in a single document is signal, so scale it up.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Both produce sparse vectors with interpretable axes. You can read which words push a document toward each class. You cannot do this with a 768-dimensional BERT embedding.

## Build It

### Step 1: Build the Vocabulary

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

### Step 2: Bag of Words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

### Step 3: Term Frequency and Document Frequency

```python
import math

def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]

def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df

def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Two smoothing tricks: `(n+1)/(d+1)` avoids `log(x/0)`. The trailing `+1` ensures a word in every document has IDF 1 (not 0).

### Step 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

Three documents, five vocab words. `the` appears in all three so its IDF is low. `dog` appears in one so its IDF is high.

### Step 5: L2-normalize Rows

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Without normalization, a longer document dominates similarity scores. L2 normalization puts every document on the unit hypersphere.

## Use It

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
```

| Arg | Effect |
|-----|--------|
| `ngram_range=(1, 2)` | Include bigrams. Boosts classification. |
| `min_df=2` | Drop words in fewer than 2 docs. |
| `max_df=0.95` | Drop words in more than 95% of docs. |
| `stop_words="english"` | scikit-learn's builtin stopword list. |
| `sublinear_tf=True` | Use `1 + log(tf)` instead of raw `tf`. |

### When TF-IDF Still Wins

Spam detection, topic labeling, log anomaly flagging. Low-data regimes (hundreds of examples). Anywhere latency matters. Systems that must explain their predictions.

### When TF-IDF Fails

Semantic blindness: "The movie was not good at all" vs "The movie was excellent" have the same BoW overlap `{the, movie, was}`. Out-of-vocabulary words at inference have no representation.

### Hybrid: TF-IDF Weighted Embeddings

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

You get semantic capacity from embeddings and rare-word emphasis from TF-IDF. Outperforms either alone for sentiment, topic, and intent classification below ~50k labeled examples.

## Exercises

1. **Easy.** Implement `cosine_similarity(doc_vec_a, doc_vec_b)` on L2-normalized TF-IDF output.
2. **Medium.** Add n-gram support to `bag_of_words`.
3. **Hard.** Build the TF-IDF-weighted-embedding hybrid using GloVe 100d. Compare against plain TF-IDF and plain mean-pooled embeddings on 20 Newsgroups.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| BoW | Counts of vocabulary words in one document. Throws away order. |
| TF | Count of a word in a document, optionally normalized by length. |
| DF | Count of documents containing the word at least once. |
| IDF | `log(N / df)` smoothed. Downweights words that appear everywhere. |
| Sparse vector | Mostly zeros. Vocabulary is typically 10k-100k words. |
| Cosine similarity | Dot product of L2-normalized vectors. 1 is identical, 0 is orthogonal. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf)

---

## Part 3 (ch096): Word Embeddings — Word2Vec from Scratch

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

---

## Part 4 (ch097): GloVe, FastText, and Subword Embeddings

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

---

## Part 5 (ch112): Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece

Word tokenizers choke on unseen words. Character tokenizers blow up sequence length. Subword tokenizers split the difference. Every modern LLM ships on one.

## The Concept

**BPE (Byte-Pair Encoding).** Start with character-level vocabulary. Count every adjacent pair. Merge the most frequent pair into a new token. Repeat until target vocab size. Used by GPT-2/3/4, Llama, Mistral.

**Byte-level BPE.** Same algorithm over raw bytes (256 base tokens). Guarantees zero [UNK] tokens. GPT-2 uses 50,257 tokens.

**Unigram.** Start with a huge vocabulary. Iteratively prune tokens whose removal least increases corpus log-likelihood. Used by T5, mBART, Gemma.

**WordPiece.** Merge pairs that maximize likelihood rather than raw frequency. Used by BERT, DistilBERT.

**SentencePiece vs tiktoken.** SentencePiece trains vocabularies (BPE or Unigram) directly on raw Unicode text. tiktoken is OpenAI's fast encoder against pre-built vocabularies.

## Build It

### Step 1: BPE from Scratch

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Three facts: `</w>` marks word end, frequency weighting makes high-frequency pairs win early, merge list is ordered.

### Step 2: Encode with the Learned Merges

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

### Step 3: SentencePiece in Practice

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995,
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

No pre-tokenization required. Space encoded as `▁`. `character_coverage` controls preservation of rare characters.

### Step 4: tiktoken for OpenAI-Compatible Vocabs

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Encoding-only. Fast (Rust backend). Exact match with GPT-4/5 tokenization.

## Pitfalls

- **Tokenizer drift.** Training on vocab A, deploying against vocab B. Check `tokenizer.json` hash in CI.
- **Whitespace ambiguity.** "hello" vs " hello" produce different tokens.
- **Multilingual undertraining.** English-heavy corpora produce vocabs that split non-Latin scripts into 5-10x more tokens.
- **Emoji splits.** A single emoji can take 5 tokens.

## Use It

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram) |
| Serving an OpenAI-compatible API | tiktoken (o200k_base) |
| Domain-specific vocab | Train custom BPE on domain corpus |

Vocabulary size heuristic: 32k for <1B params, 50-100k for 1-10B, 200k+ for multilingual/frontier.

## Exercises

1. **Easy.** Train a 500-merge BPE on a tiny corpus. Encode three held-out words.
2. **Medium.** Compare token counts on 100 English Wikipedia sentences between cl100k_base, o200k_base, and a SentencePiece BPE.
3. **Hard.** Train the same corpus with BPE, Unigram, and WordPiece. Measure downstream accuracy on a small sentiment classifier.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| BPE | Greedy merge of most-frequent character pairs until target vocab size. |
| Byte-level BPE | BPE over raw 256 bytes. No unknown tokens ever. |
| Unigram | Prunes from large candidate set using log-likelihood. |
| SentencePiece | Library training BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | OpenAI's Rust-backed BPE encoder for pre-built vocabs. |
| Merge list | Ordered list of `(a, b) → ab` merges. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/19-subword-tokenization)
