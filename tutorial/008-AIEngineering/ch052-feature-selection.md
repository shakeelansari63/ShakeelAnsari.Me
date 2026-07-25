# Feature Selection

> More features is not better. The right features is better.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-09, 08 (feature engineering)
**Time:** ~75 minutes

## Learning Objectives

- Implement filter methods (variance threshold, mutual information, chi-squared) and wrapper methods (RFE, forward selection) from scratch
- Explain why mutual information captures nonlinear feature-target relationships that correlation misses
- Compare L1 regularization (embedded selection) with RFE (wrapper selection) and evaluate their computational tradeoffs
- Build a feature selection pipeline that combines multiple methods and demonstrate improved generalization on held-out data

## The Problem

You have 500 features. Your model trains slowly, overfits constantly, and nobody can explain what it learned. This is the curse of dimensionality -- as features grow, the feature space volume explodes, data becomes sparse, and noise drowns out signal.

Feature selection is the antidote. Strip away the noise. Remove the redundancy. Keep the features that carry actual information about the target.

## The Concept

### Three Categories of Feature Selection

| Method | Type | Speed | Nonlinear | Feature Interactions |
|--------|------|-------|-----------|---------------------|
| Variance threshold | Filter | Very fast | No | No |
| Mutual information | Filter | Fast | Yes | No |
| Correlation filter | Filter | Fast | No | No |
| RFE | Wrapper | Slow | Depends on model | Yes |
| L1 / Lasso | Embedded | Fast | No (linear) | No |
| Tree importance | Embedded | Medium | Yes | Yes |
| Permutation importance | Model-agnostic | Slow | Yes | Yes |

### Variance Threshold

The simplest filter. If a feature barely varies across samples, it carries almost no information.

```
variance(x) = mean((x - mean(x))^2)
```

Drop every feature with variance below a threshold (e.g., 0.01).

### Mutual Information

Mutual information measures how much knowing the value of feature X reduces uncertainty about target Y.

```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

Key advantage over correlation: mutual information captures nonlinear relationships. A feature might have zero correlation with the target but high mutual information because the relationship is quadratic or periodic.

### Recursive Feature Elimination (RFE)

RFE is a wrapper method that uses a model's own feature importance to iteratively prune:

1. Train the model with all features
2. Rank features by importance
3. Remove the least important feature(s)
4. Repeat until the desired number of features remains

RFE considers feature interactions because the model sees all remaining features together.

### L1 (Lasso) Regularization

L1 regularization adds the absolute value of weights to the loss function:

```
loss = prediction_error + alpha * sum(|w_i|)
```

Higher alpha means more weights go to exactly zero. This is embedded feature selection -- the model learns during training which features to ignore.

### Tree-Based Feature Importance

Decision trees and their ensembles naturally rank features by impurity reduction. Features that produce larger impurity reductions are more important.

Caution: tree-based importance is biased toward features with many unique values.

### Decision Flowchart

| Feature Count | Recommended Approach |
|---------------|---------------------|
| < 50 | Variance threshold + mutual information |
| 50-500 | Variance threshold, then L1 or tree importance |
| > 500 | Variance threshold, then mutual info filter, then RFE on survivors |

## Build It

### Step 1: Variance threshold

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### Step 2: Mutual information (discrete)

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned

def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)
    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples
    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))
        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi
    return mi_scores
```

### Step 3: Recursive Feature Elimination

```python
def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total
    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)
        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)
    for idx in remaining:
        rankings[idx] = 1
    selected_mask = rankings == 1
    return selected_mask, rankings
```

### Step 4: L1 feature selection

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)

def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)
        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b
    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### Step 5: Tree-based importance

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0
    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)
    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            continue
        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return best_threshold, best_gain
```

## Use It

With scikit-learn:

```python
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, RFE, SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

## Ship It

This lesson produces `outputs/skill-feature-selector.md`.

## Exercises

1. **Forward selection**: implement the opposite of RFE. Start with zero features. At each step, add the feature that improves performance the most.
2. **Stability selection**: run L1 feature selection 50 times, each on a random 80% subsample. Count how often each feature is selected.
3. **Multicollinearity detection**: compute the correlation matrix. Remove one feature from each highly-correlated pair (keeping the one with higher mutual information).
4. **Feature selection pipeline**: chain variance threshold, mutual information filter, and RFE into a single pipeline.
5. **Permutation importance from scratch**: for each feature, shuffle its values 10 times, measure the average drop in F1 score.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Filter method | "Score features independently" | A feature selection approach that ranks features using a statistical measure without training a model |
| Wrapper method | "Use the model to pick features" | A feature selection approach that evaluates feature subsets by training a model |
| Embedded method | "The model selects features during training" | Feature selection that happens as part of model fitting, such as L1 regularization |
| Mutual information | "How much one variable tells you about another" | A measure of the reduction in uncertainty about Y given knowledge of X |
| Recursive Feature Elimination | "Train, rank, prune, repeat" | An iterative wrapper method that removes the least important feature(s) until a target count is reached |
| L1 / Lasso regularization | "Penalty that kills features" | Adding the sum of absolute weight values to the loss function, driving unimportant weights to zero |
| Variance threshold | "Remove constant features" | Dropping features whose variance falls below a specified threshold |
| Feature importance | "Which features matter most" | A score indicating how much each feature contributes to model predictions |
| Permutation importance | "Shuffle and measure the damage" | Evaluating feature importance by randomly shuffling each feature's values and measuring the performance drop |
| Curse of dimensionality | "Too many features, not enough data" | The phenomenon where adding features increases the volume of feature space exponentially |

## Further Reading

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html)
- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Stability Selection (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932)
- [Beware Default Random Forest Importances (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/02-ml-fundamentals/18-feature-selection)
