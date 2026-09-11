# NLP Tasks, Inference & Knowledge Graphs

> Combined lessons (8 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch098): Sentiment Analysis

The canonical NLP task. Most of what you need to know about classical text classification shows up here.

"The food was not great." Positive or negative? Sentiment sounds simple until negation flips meaning, sarcasm inverts it, and "not bad at all" is positive despite two negative-coded words.

## The Concept

Classical sentiment is a two-step recipe:

1. **Represent.** Turn text into a feature vector (BoW, TF-IDF, n-grams).
2. **Classify.** Fit a linear model (Naive Bayes, logistic regression, SVM) on labeled examples.

Naive Bayes assumes every feature is independent given the label. The assumption is wrong but results are strong: with sparse text features, the classifier cares about which side each word leans toward more than how much.

Logistic regression fixes the independence assumption. It learns a weight per feature, including negative weights. `not_good` as a bigram feature gets a negative weight.

## Build It

### Step 1: A Real Mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

### Step 2: Multinomial Naive Bayes from Scratch

```python
import math
from collections import Counter

def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs

def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

### Step 3: Logistic Regression from Scratch

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization is essential for sparse text features. Start at `0.01` and tune.

### Step 4: Handling Negation

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}

def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Now `good` and `NOT_good` are different features. Three lines of preprocessing, measurable accuracy jump.

### Step 5: Evaluation Metrics That Matter

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

Report per-class precision/recall, macro-F1 (not micro-F1), confusion matrix, and per-class error samples. For severely imbalanced data, report AUROC and AUPRC.

## Use It

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Three flags that matter: `stop_words=None` keeps negations, `ngram_range=(1, 2)` adds bigrams so `not_good` becomes a feature, `sublinear_tf=True` dampens repeated words.

### When to Reach for a Transformer

Sarcasm detection, long reviews with mid-document sentiment shifts, aspect-based sentiment, non-English low-resource languages. Otherwise, Naive Bayes or logistic regression on TF-IDF plus bigrams plus negation handling is your 2026 production baseline.

## Exercises

1. **Easy.** Add `apply_negation` as a preprocessing step and measure the F1 delta.
2. **Medium.** Implement class-weighted logistic regression. Measure the effect on a 90-10 class imbalance.
3. **Hard.** Build a sarcasm detector by training a second classifier on the residuals of the sentiment model.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Polarity | Positive or negative. Sometimes extended to neutral or fine-grained. |
| Aspect-based sentiment | Per-aspect polarity attributed to specific entities. |
| Negation scoping | Prefix tokens after "not" with `NOT_` until punctuation. |
| Laplace smoothing | Adding 1 to counts. Prevents zero-probability in Naive Bayes. |
| L2 regularization | Adds `lambda * sum(w^2)` to loss. Essential for sparse text features. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/05-sentiment-analysis)

---

## Part 2 (ch099): Named Entity Recognition

Pull the names out. Sounds easy until you deal with ambiguous boundaries, nested entities, and domain jargon.

NER is the workhorse underneath every structured extraction pipeline: resume parsing, compliance log scanning, medical record anonymization, search query understanding, legal contract extraction.

## The Concept

**BIO tagging** turns entity extraction into a sequence-labeling problem. Label each token with `B-TYPE` (beginning), `I-TYPE` (inside), or `O` (outside).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

The architecture progression: Rule-based → HMM → CRF → BiLSTM-CRF → Transformer-based.

## Build It

### Step 1: BIO Tagging Helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels

def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

### Step 2: Hand-crafted Features

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }

def word_shape(word):
    out = []
    for c in word:
        if c.isupper(): out.append("X")
        elif c.islower(): out.append("x")
        elif c.isdigit(): out.append("d")
        else: out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` returns `xXxxxx`. Capitalization patterns are high-signal for proper nouns.

### Step 3: Rule-based + Dictionary Baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}

def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

### Step 4: CRF with sklearn-crfsuite

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
```

### Step 5: BiLSTM-CRF Sketch

```python
import torch
import torch.nn as nn

class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

## Use It

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

### Hugging Face

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

### LLM-based NER (2026 option)

Zero-shot and few-shot LLM NER is now competitive with fine-tuned models. Start with an LLM zero-shot baseline before collecting training data.

### Where Classical NER Still Wins

Latency under 50ms, thousands of labeled examples needing 98%+ F1, regulatory constraints requiring on-prem non-generative models.

### Where It Falls Apart

Domain shift, nested entities, long entities, sparse entity types.

## Exercises

1. **Easy.** Implement `bio_to_spans` and verify round-trip consistency.
2. **Medium.** Train sklearn-crfsuite CRF on CoNLL-2003. Report per-entity F1.
3. **Hard.** Fine-tune `distilbert-base-cased` on a domain-specific NER dataset.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| NER | Label token spans with types (PERSON, ORG, GPE, DATE...). |
| BIO | `B-X` begins, `I-X` continues, `O` outside. |
| BILOU | Adds `L-X` (last), `U-X` (unit) for cleaner boundaries. |
| CRF | Models transitions between labels, not just emissions. |
| Nested NER | Overlapping entities. BIO cannot express this. |
| Entity-level F1 | Predicted span must match true span exactly. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition)

---

## Part 3 (ch100): POS Tagging and Syntactic Parsing

Grammar was unfashionable for a while. Then every LLM pipeline needed to validate structured extraction, and it came back.

Lesson 01 promised that lemmatization needs a POS tag. Without knowing `running` is a verb, a lemmatizer cannot reduce it to `run`. This lesson introduces the tagsets, the baselines, and the point where you stop implementing from scratch and call spaCy.

## The Concept

**POS tagging** labels each token with a grammatical category. The **Penn Treebank (PTB)** tagset: 36 tags (`NN` singular noun, `NNS` plural noun, `VBD` verb past tense, etc.). The **Universal Dependencies (UD)** tagset is coarser (17 tags) and language-agnostic.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing** produces a tree. Two styles:

- **Constituency parsing.** Noun phrases, verb phrases nest inside each other.
- **Dependency parsing.** Each word has one head word, labeled with a grammatical relation.

Dependency parsing won because it generalizes cleanly across languages.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It

### Step 1: Most-Frequent-Tag Baseline

```python
from collections import Counter, defaultdict

def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag

def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

On the Brown corpus, this hits ~85% accuracy.

### Step 2: Bigram HMM Tagger

Model the joint probability: `P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)`

```python
import math

def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab

def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)

def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        V[0][j] = log_prob(transitions, "<BOS>", tag, tr_denom, alpha) + log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM on Brown hits ~93% accuracy. The jump from 85% to 93% is mostly transition probabilities.

## Use It

Every production NLP library ships POS and dependency parsers.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

### Where This Matters in 2026

Lemmatization (needs POS), structured extraction from LLM outputs, aspect-based sentiment (dependency tells which adjective modifies which noun), query understanding, cross-lingual transfer, low-compute pipelines.

## Exercises

1. **Easy.** Measure most-frequent-tag baseline accuracy on a small tagged corpus (~85%).
2. **Medium.** Train the bigram HMM and report per-tag precision/recall.
3. **Hard.** Use spaCy's dependency parse to extract subject-verb-object triples from 1000 sentences.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| POS tag | Grammatical category. PTB has 36; UD has 17. |
| Penn Treebank | English-specific fine-grained tagset. |
| Universal Dependencies | Multilingual, coarser than PTB. |
| Dependency parse | Each word has one head, each edge a grammatical relation. |
| Viterbi | Dynamic programming for highest-probability tag sequence. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing)

---

## Part 4 (ch108): Topic Modeling — LDA and BERTopic

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

---

## Part 5 (ch114): Natural Language Inference — Textual Entailment

"t entails h" means a human reading t would conclude h is true. NLI is the task of predicting entailment / contradiction / neutral. Boring on the surface, load-bearing in production.

## The Problem

You built a summarizer. How do you know the summary does not contain a hallucination? You built a chatbot. How do you know the answer is supported by the retrieved passage? All three problems reduce to NLI:

- **Hallucination check:** premise = source, hypothesis = summary claim. Not entailment = hallucination.
- **Grounded QA:** premise = retrieved passage, hypothesis = answer. Not entailment = fabrication.
- **Zero-shot classification:** premise = document, hypothesis = verbalized label ("This is about sports"). Entailment = predicted label.

## The Concept

**The three labels:**
- **Entailment.** "The cat is on the mat" entails "There is a cat."
- **Contradiction.** "The cat is on the mat" contradicts "There is no cat."
- **Neutral.** No inference either way.

**Not logical entailment.** NLI is what a typical human reader would infer, not strict logic. "John walked his dog" entails "John has a dog" in NLI.

**Datasets:** SNLI (570k pairs), MultiNLI (433k pairs, 10 genres), ANLI (adversarial), DocNLI (document-length).

**The architecture.** A transformer encoder reads `[CLS] premise [SEP] hypothesis [SEP]`. The `[CLS]` representation feeds a 3-way softmax.

## Build It

### Step 1: Run a Pretrained NLI Model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

### Step 2: Zero-shot Classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

No training data required. Customize `hypothesis_template` if needed.

### Step 3: Faithfulness Check for RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

This is the core of RAGAS faithfulness. Decompose the answer into atomic claims and check each.

## Pitfalls

- **Hypothesis-only shortcuts.** Models can predict label from hypothesis alone at ~60% on SNLI.
- **Lexical overlap heuristic.** Subsequence heuristic passes SNLI but fails HANS/ANLI.
- **Document-length degradation.** Single-sentence NLI models drop 20+ F1 on long premises.
- **Zero-shot template sensitivity.** Tune the hypothesis template.
- **Domain mismatch.** MNLI trains on general English. Legal/medical need domain-specific NLI.

## Use It

| Use case | Model |
|---------|-------|
| General-purpose NLI | `microsoft/deberta-v3-large-mnli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |

The 2026 meta-pattern: NLI is the duct tape of text understanding. Whenever you need "does A support B?" — reach for NLI before another LLM call.

## Exercises

1. **Easy.** Run BART-MNLI on 20 hand-crafted triples. Add adversarial subsequence-heuristic traps.
2. **Medium.** Compare zero-shot templates on 100 AG News headlines. Report accuracy swing.
3. **Hard.** Build a RAG faithfulness checker: atomic-claim decomposition + NLI per claim. Evaluate on 50 examples.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| NLI | 3-way classification of premise-hypothesis relationship. |
| RTE | Older name for NLI. Same task. |
| Entailment | Reader would conclude h is true given t. |
| Contradiction | Reader would conclude h is false given t. |
| Neutral | No inference either way. |
| Zero-shot classification | Verbalize labels as hypotheses, pick max entailment. |
| Faithfulness | Is the answer supported by context? NLI over (context, answer). |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment)

---

## Part 6 (ch117): Coreference Resolution

"She called him. He did not answer. The doctor was at lunch." Three references to two people and nobody is named. Coreference resolution figures out who is who.

## The Concept

Coreference resolution links every expression that refers to the same real-world entity into one cluster. It is the glue between surface-level NLP (NER, parsing) and downstream semantics (IE, QA, summarization, KG).

**Mention types:**
- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Appositive.** "Tim Cook, Apple's CEO,"

**Architectures (in order of sophistication):**

1. **Rule-based (Hobbs, 1978).** Syntactic-tree-based pronoun resolution using grammar rules.
2. **Mention-pair classifier.** Predict whether each pair of mentions corefer.
3. **Mention-ranking.** For each mention, rank candidate antecedents.
4. **Span-based end-to-end (Lee et al., 2017).** Enumerate all candidate spans, predict mention scores and antecedent probabilities. The modern default.
5. **Generative (2024+).** Prompt an LLM to list pronouns and antecedents.

**Evaluation.** Five standard metrics (MUC, B³, CEAF, BLANC, LEA). Report the average of the first three as CoNLL F1. State-of-the-art on CoNLL-2012: ~83 F1.

## Build It

### Step 1: Pretrained Neural Coreference

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Cluster 1: [Apple, The company, they], Cluster 2: [new products]

### Step 2: Using LLMs for Coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Two failure modes: LLMs over-merge ("him" and "her" referring to distinct people), and silently drop mentions in long documents.

## Pitfalls

- **Singleton explosion.** Some systems report every mention as its own cluster.
- **Pronouns in long context.** Performance drops ~15 F1 on documents over 2000 tokens.
- **Gender assumptions.** Hard-coded gender rules break on non-binary referents.
- **LLM drift on long docs.** Use sliding-window + merge.

## Use It

| Situation | Pick |
|-----------|------|
| English, single document | `en_coreference_web_trf` or AllenNLP neural coref |
| Multilingual | SpanBERT / XLM-R on OntoNotes or Multilingual CoNLL |
| Quick LLM baseline | GPT-4o / Claude with structured-output coref prompt |

Integration pattern: run NER first, run coref, merge coref clusters into NER entities. Downstream tasks see one entity per cluster.

## Exercises

1. **Easy.** Run the rule-based resolver on 5 hand-crafted paragraphs. Measure mention-link accuracy.
2. **Medium.** Use a pretrained neural coref model on a news article. Compare clusters against your manual annotation.
3. **Hard.** Build a coref-enhanced NER pipeline. Measure entity-coverage improvement vs NER-only on 100 articles.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Mention | A span of text that refers to an entity. |
| Antecedent | The earlier mention a later one corefers with. |
| Cluster | Set of mentions that all refer to the same real-world entity. |
| Anaphora | Later mention refers to earlier ("he" → "John"). |
| Cataphora | Earlier mention refers to later ("When he arrived, John..."). |
| Bridging | Implicit reference ("The wheels" → a previously mentioned car). |
| CoNLL F1 | Average of MUC, B³, CEAF-φ4 F1 scores. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/24-coreference-resolution)

---

## Part 7 (ch118): Entity Linking & Disambiguation

NER found "Paris." Entity linking decides: Paris, France? Paris Hilton? Paris, Texas? Without linking, your knowledge graph stays ambiguous.

## The Concept

Entity linking (EL) resolves each mention to a unique entry in a knowledge base. Two subtasks:

1. **Candidate generation.** Given "Jordan," which KB entries are plausible?
2. **Disambiguation.** Given the context, which candidate is the right one?

**Disambiguation approaches:**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Fast, no training.
2. **Embedding-based (BLINK).** Encode mention + context and each candidate's description. Pick max cosine. The 2020-2024 default.
3. **Generative (GENRE, LLM-based).** Decode the entity's canonical name token-by-token with constrained decoding.

## Build It

### Step 1: Build an Alias Index

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data: ~18M (alias, entity) pairs.

### Step 2: Embedding-Based (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

At index time, embed every KB entity once. At query time, embed mention + context, dot-product against candidates, pick max.

### Step 3: Generative Entity Linking

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Combined with a whitelist (Outlines `choice`), this is the simplest EL pipeline to ship in 2026.

## Pitfalls

- **NIL handling.** Some mentions are not in the KB. Must predict NIL.
- **Mention boundary errors.** Upstream NER misses partial spans.
- **Popularity bias.** Trained systems over-predict frequent entities.
- **KB staleness.** New companies/events are not in last year's Wikipedia dump.

## Use It

| Situation | Pick |
|-----------|------|
| General-purpose English + Wikipedia | BLINK or REL |
| LLM-friendly, few mentions/day | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| Domain-specific KB | Custom BERT with KB-aware retrieval + fine-tune |

Production pattern: NER → coref → EL on each mention → collapse clusters to one canonical entity per cluster.

## Exercises

1. **Easy.** Implement the prior+context disambiguator on 10 ambiguous mentions.
2. **Medium.** Encode 50 ambiguous mentions with a sentence transformer. Compare embedding-based to Jaccard overlap.
3. **Hard.** Build a 1k-entity domain KB. Implement NER + EL end-to-end.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Entity linking | Map a mention to a unique KB entry. |
| Candidate generation | Return a shortlist of plausible KB entries. |
| Disambiguation | Score candidates using context, pick the winner. |
| Alias index | Map from surface form → candidate entities. |
| NIL | Explicit prediction that no KB entry matches. |
| AIDA-CoNLL | 1,393 Reuters articles with gold entity links. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/25-entity-linking)

---

## Part 8 (ch119): Relation Extraction & Knowledge Graph Construction

NER found the entities. Entity linking anchored them. Relation extraction finds the edges between them. A knowledge graph is the sum of nodes, edges, and their provenance.

## The Concept

**Triple form.** `(subject_entity, relation_type, object_entity)`. Relations come from a closed ontology or an open set.

**Three extraction approaches:**

1. **Rule/pattern-based.** Hearst patterns: "X such as Y" → `(Y, isA, X)`. Precise, brittle.
2. **Supervised classifier.** Given two entity mentions, predict the relation. Trained on TACRED, ACE.
3. **Generative LLM.** Prompt the model to emit triples. Works out of the box. Needs provenance.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Anchor every entity span with exact positions. Extract triples linked to anchors. Verify each triple element against source text. Supplement with a coverage pass.

## Build It

### Step 1: Pattern-Based Extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

### Step 2: Supervised Relation Classification

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL is a seq2seq relation extractor: text in, triples out, already in Wikidata property ids.

### Step 3: LLM-Prompted Extraction with Anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text.
"""
```

Verify every returned span against the source. Reject anything where `text[start:end] != triple_entity`.

### Step 4: Canonicalize onto a Closed Ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",
    "was born in":   "P19",
    "founded":        "P112",
    "works at":       "P108",
}

def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None
```

Canonicalization is often 60-80% of the engineering work.

## Pitfalls

- Coreference before RE. "He founded Apple" needs coreference resolution first.
- Entity canonicalization. "Apple Inc" and "Apple" must resolve to the same node.
- Hallucinated triples. Enforce span verification.
- Relation canonicalization drift. Collapse to canonical ids.
- Temporal errors. Many relations are time-bounded. Use qualifiers.

## Use It

| Situation | Pick |
|-----------|------|
| Fast production, general domain | REBEL with Wikidata canonicalization |
| Domain-specific | SciREX-style domain fine-tune |
| LLM-prompted, audited output | AEVS pipeline |

Integration pattern: NER → coref → entity linking → RE → ontology mapping → graph load.

## Exercises

1. **Easy.** Run the pattern extractor on 5 news-article sentences. Hand-check precision.
2. **Medium.** Use REBEL on the same sentences. Compare triples precision/recall.
3. **Hard.** Build the AEVS pipeline. Measure hallucination rate before vs after verify step.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Triple | `(s, r, o)` tuple. Atomic unit of a KG. |
| Open IE | Open-vocabulary relation phrases; high recall, low precision. |
| Closed ontology | Bounded set of relation types (Wikidata, UMLS). |
| Canonicalization | Map surface names/relations to canonical ids. |
| AEVS | Anchor-Extraction-Verification-Supplement pipeline (2026). |
| Provenance | Every triple carries a doc id + char-span to its source. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg)
