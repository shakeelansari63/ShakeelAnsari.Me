# Information Retrieval and Search

BM25 is precise but brittle. Dense casts a wide net but misses keywords. Hybrid is the 2026 default. Everything else is tuning.

## The Concept

Four layers of retrieval:

1. **Sparse retrieval (BM25).** Fast, precise on exact matches, terrible on semantics. Runs over an inverted index in sub-10ms per query.
2. **Dense retrieval.** Encode query and documents into vectors. Nearest neighbor search. Captures paraphrases and semantic similarity.
3. **Fusion.** Merge ranked lists from sparse and dense. Reciprocal Rank Fusion (RRF) ignores raw scores and only uses rank positions.
4. **Cross-encoder rerank.** Take top-30 from fusion, run a cross-encoder, keep top-5. Slower per pair but far more accurate.

## Build It

### Step 1: BM25 from Scratch

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

`k1=1.5` controls term-frequency saturation. `b=0.75` controls length normalization.

### Step 2: Dense Retrieval with a Bi-Encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings

def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

### Step 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

### Step 4: Hybrid Search + Rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

### Step 5: Evaluation

| Metric | Meaning |
|--------|---------|
| Recall@k | How often is the correct doc in top-k? |
| MRR | Average of 1/rank of first relevant document. |
| nDCG@k | Accounts for relevance gradations. |

## Hard-won Lessons from 2026 Production RAG

- 80% of RAG failures trace to ingestion and chunking, not the model.
- Chunking strategy matters more than chunk size.
- Parent-doc pattern: retrieve small children, return larger parents.
- k_rerank=3 is usually optimal.
- HyDE / query expansion bridges the phrasing gap.
- Three-way retrieval (BM25 + dense + learned-sparse) outperforms two-way.

## Use It

| Scale | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + MiniLM embeddings + RRF |
| 100k-10M docs | FAISS or pgvector + Elasticsearch |
| 10M+ docs | Qdrant / Weaviate / Vespa with hybrid support |

## Exercises

1. **Easy.** Implement `hybrid_search` on a 500-document corpus. Compare recall@5 for BM25-only, dense-only, and hybrid.
2. **Medium.** Add MRR calculation. Report MRR for each retrieval method.
3. **Hard.** Fine-tune a dense encoder on your domain using MultipleNegativesRankingLoss.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| BM25 | Okapi BM25. Scores by term frequency, IDF, and length. |
| Dense retrieval | Encode query + doc into vectors, find nearest neighbors. |
| Bi-encoder | Encodes query and doc independently. Fast. |
| Cross-encoder | Encodes query + doc together. Slow but accurate. |
| RRF | Combine rankings by summing `1/(k + rank)`. |
| Recall@k | Fraction of queries where a relevant doc is in top-k. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/14-information-retrieval-search)
