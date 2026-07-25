# Topic Modeling — LDA and BERTopic

LDA: documents are mixtures of topics, topics are distributions over words. BERTopic: documents cluster in embedding space, clusters are topics. Same goal, different decompositions.

## The Concept

**LDA generative story.** Each topic is a distribution over words. Each document is a mixture of topics. Inference reverses this: given observed words, infer the topic distribution per document and the word distribution per topic.

Key LDA output:
- `doc_topic`: matrix `(n_docs, n_topics)`, each row sums to 1.
- `topic_word`: matrix `(n_topics, vocab_size)`, each row sums to 1.

**BERTopic pipeline:**
1. Encode each document with a sentence transformer.
2. Reduce dimensionality with UMAP to ~5 dimensions.
3. Cluster with HDBSCAN (density-based, variable-size clusters + outlier label).
4. For each cluster, compute class-based TF-IDF to extract top words.

## Build It

### Step 1: LDA via scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names

def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

### Step 2: BERTopic

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

The filter on `Topic != -1` drops the outlier bucket. `min_topic_size` controls HDBSCAN's minimum cluster size.

### Step 3: Evaluation

- **Topic coherence (c_v).** NPMI of top-word pairs. Use `gensim.models.CoherenceModel`.
- **Topic diversity.** Fraction of unique words across all topics' top words.
- **Qualitative inspection.** Read the top words. Do they name a real thing?

## When to Pick Which

| Situation | Pick |
|-----------|------|
| Short text (tweets, reviews) | BERTopic |
| Long documents with topic mixtures | LDA |
| No GPU / limited compute | LDA or NMF |
| Need document-level multi-topic distributions | LDA |
| Max semantic coherence | BERTopic |

## Use It

BERTopic is the default for short text. `gensim.models.LdaModel` for production LDA. NMF as a fast alternative. LLM-based labeling for any clustering method.

## Exercises

1. **Easy.** Fit LDA with 5 topics on 20 Newsgroups. Label each topic by hand.
2. **Medium.** Fit BERTopic on the same subset. Compare number of topics, top words, and qualitative coherence.
3. **Hard.** Compute c_v coherence for both LDA and BERTopic at 5, 10, 20, 50 topics.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Topic | Probability distribution over words (LDA) or cluster of similar docs (BERTopic). |
| Mixed membership | LDA assigns each doc a distribution over all topics. |
| UMAP | Manifold learning that preserves local structure. |
| HDBSCAN | Density clustering with "noise" label (-1) for outliers. |
| c_v coherence | Average pointwise mutual information of top topic words. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/15-topic-modeling)
