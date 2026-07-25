# Naive Bayes

> The "naive" assumption is wrong, and it works anyway. That's the beauty of it.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Learning Objectives

- Implement Multinomial Naive Bayes from scratch with Laplace smoothing for text classification
- Explain why the naive independence assumption is mathematically wrong but produces correct class rankings in practice
- Compare Multinomial, Bernoulli, and Gaussian Naive Bayes variants and select the right one for a given feature type
- Evaluate Naive Bayes against logistic regression on high-dimensional sparse data and explain the bias-variance tradeoff at work

## The Problem

You need to classify text. Emails into spam or not-spam. Customer reviews into positive or negative. Support tickets into categories. You have thousands of features (one per word) and limited training data.

Most classifiers choke here. Logistic regression needs enough samples to estimate thousands of weights reliably. Decision trees split on one word at a time and overfit wildly. KNN in 10,000 dimensions is meaningless.

Naive Bayes handles this. It makes a mathematically wrong assumption (that every feature is independent of every other feature given the class), and it still outperforms "smarter" models on text classification, especially with small training sets.

## The Concept

### Bayes' Theorem (Quick Review)

Bayes' theorem flips conditional probabilities:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

We want `P(class | features)` -- the probability that a document belongs to a class given the words in it.

### The Naive Independence Assumption

The naive assumption: every feature is conditionally independent given the class.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Instead of one impossible joint distribution, you estimate n simple per-feature distributions. This assumption is obviously wrong, but the classifier does not need correct probability estimates. It needs correct rankings.

### Why It Still Works

1. **Ranking over calibration.** Classification only needs the top-ranked class to be correct.
2. **High bias, low variance.** The independence assumption constrains the model heavily, preventing overfitting.
3. **Feature redundancy cancels out.** Correlated features provide redundant evidence, and NB double-counts it for the correct class.

### Three Variants

| Variant | Feature Type | Best For |
|---------|-------------|----------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words |
| Gaussian | Continuous values | Tabular data with normal-ish features |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors |

### Laplace Smoothing

What happens when a word appears in the test data but never appeared in the training data? Without smoothing, `P(word | class) = 0/N = 0`, and one zero destroys the entire prediction.

Laplace smoothing adds a small count `alpha` (usually 1) to every feature count:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

### Log-Space Computation

Multiplying hundreds of probabilities causes floating-point underflow. Work in log space:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

### Naive Bayes vs Logistic Regression

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models P(X|Y)) | Discriminative (models P(Y|X)) |
| Small data | Better (strong prior helps) | Worse |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Speed | Single pass, very fast | Iterative optimization |

## Build It

### MultinomialNB

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]
        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())
        return self
```

After fitting, prediction is just matrix multiplication plus a bias. This is why Naive Bayes is so fast.

### GaussianNB

For continuous features, estimate mean and variance per class per feature:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))
        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]
        return self
```

## Use It

With sklearn:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
```

For text classification:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])
text_clf.fit(train_texts, train_labels)
```

### TF-IDF with Naive Bayes

```python
from sklearn.feature_extraction.text import TfidfVectorizer

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

### BernoulliNB for Short Text

```python
from sklearn.naive_bayes import BernoulliNB

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

## Ship It

This lesson produces `outputs/skill-naive-bayes-chooser.md`.

## Exercises

1. **Smoothing experiment.** Train MultinomialNB on text data with alpha values of 0.01, 0.1, 1.0, 10.0, and 100.0. Plot accuracy vs alpha. Where does performance peak?
2. **Feature independence test.** Take a real text dataset. Pick two correlated words and compute P(word1|class) * P(word2|class) vs P(word1 AND word2|class). How wrong is the independence assumption?
3. **Bernoulli implementation.** Extend the code with a BernoulliNB class. Compare against MultinomialNB. When does Bernoulli win?
4. **NB vs Logistic Regression.** Train both on text data. Start with 100 training samples and increase to 10,000. At what point does Logistic Regression overtake Naive Bayes?
5. **Spam filter.** Build a complete spam classifier: tokenize, build vocabulary, bag-of-words, MultinomialNB, evaluate with precision and recall.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | A classifier that applies Bayes' theorem with the assumption that features are conditionally independent given the class |
| Conditional independence | "Features don't affect each other" | P(A, B | C) = P(A | C) * P(B | C) |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities |
| Prior | "What you believed before seeing data" | P(class) -- the probability of each class before observing any features |
| Likelihood | "How well the data fits" | P(features | class) -- the probability of observing these features if the class is known |
| Posterior | "What you believe after seeing data" | P(class | features) -- the updated probability after observing the features |
| Generative model | "Models how data is generated" | A model that learns P(X|Y) and P(Y), then uses Bayes' theorem |
| Discriminative model | "Models the decision boundary" | A model that directly learns P(Y|X) |
| Log probability | "Avoid underflow" | Working with log P to prevent product of many small numbers from becoming zero |

## Further Reading

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/02-ml-fundamentals/14-naive-bayes)
