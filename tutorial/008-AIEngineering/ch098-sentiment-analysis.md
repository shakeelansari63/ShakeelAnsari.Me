# Sentiment Analysis

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
