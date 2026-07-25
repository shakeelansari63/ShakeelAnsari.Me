# Embedding Models — The 2026 Deep Dive

Word2Vec gave you a vector per word. Modern embedding models give you a vector per passage, cross-lingual, with sparse, dense, and multi-vector views, sized to fit your index.

## The Concept

**Dense embeddings.** One vector per passage (384-3072 dim). Cosine similarity ranks passages. Default choice.

**Sparse embeddings.** SPLADE-style. A transformer predicts weight for every vocab token, zeros out most. Captures lexical matching like BM25 with learned term weights.

**Multi-vector (late interaction).** ColBERTv2. One vector per token. MaxSim scoring. Wins on long queries and domain-specific corpora.

**BGE-M3: all three at once.** Single model outputs dense, sparse, and multi-vector representations simultaneously.

**Matryoshka Representation Learning.** First N dimensions form a useful standalone embedding. Truncate 1536 dim to 256 dim and pay ~1% accuracy for 6x storage savings.

### The Three-Tier Pattern

| Use case | Pattern |
|----------|---------|
| Fast first-pass | Dense bi-encoder (BGE-M3, text-3-small) |
| Recall boost | Sparse (SPLADE) + RRF fuse |
| Precision on top-50 | Multi-vector (ColBERTv2) or cross-encoder reranker |

## Build It

### Step 1: Baseline — Dense Embeddings with Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` makes dot product equal cosine similarity.

### Step 2: Matryoshka Truncation

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Re-normalize after truncation. Non-Matryoshka models degrade sharply.

### Step 3: BGE-M3 Multi-Functionality

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
```

Score fusion:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

### Step 4: MTEB Eval on a Custom Task

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Run candidate models on a representative subset. Leaderboard rank alone is not enough.

## Pitfalls

- Same model for query and doc? Check the model card for asymmetric encoding.
- Missing prefix? `bge-*` models need `"Represent this sentence for searching relevant passages: "` prepended to queries.
- Over-trimming Matryoshka? Validate on your eval set.
- Context truncation? Long docs need chunking (lesson 23).
- Ignoring latency tail? MTEB scores hide p99 latency.

## Use It

| Situation | Pick |
|-----------|------|
| English-only, fast, API | `text-embedding-3-large` or `voyage-3-large` |
| Open-weight, English | `BAAI/bge-large-en-v1.5` |
| Open-weight, multilingual | `BAAI/bge-m3` or `Qwen3-Embedding-8B` |
| Long context (32k+) | Voyage-3-large, Cohere embed-v4 |
| Storage-constrained | Matryoshka-truncated + int8 quantization |

## Exercises

1. **Easy.** Encode 100 sentences with `bge-small-en-v1.5` at full dim (384) and Matryoshka 128. Measure MRR drop.
2. **Medium.** Compare BGE-M3 dense, sparse, and colbert on 500 passages from your domain.
3. **Hard.** Run MTEB on three candidate models. Report MTEB score, p99 latency, and cost.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Dense embedding | One fixed-size vector per text. Cosine similarity for ranking. |
| Sparse embedding | One weight per vocab token; mostly zeros. |
| Multi-vector | One vector per token; MaxSim scoring. |
| Matryoshka | First N dims are a valid smaller embedding on their own. |
| MTEB | Massive Text Embedding Benchmark. |
| Asymmetric encoding | Model uses different projections for queries and documents. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive)
