# Bag of Words, TF-IDF, and Text Representation

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
