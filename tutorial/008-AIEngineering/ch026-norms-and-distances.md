# Norms and Distances

> The geometry of data is defined by how we measure distance.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04  
**Time:** ~120 minutes  

## Learning Objectives

- Implement L1, L2, L-infinity norms from scratch and explain their geometric behavior
- Prove and verify the triangle inequality, Cauchy-Schwarz, and Hölder's inequalities numerically
- Implement cosine similarity, Euclidean distance, Manhattan distance, and Hamming distance
- Use distance computations to find nearest neighbors in synthetic data

## The Concept

### What Is a Norm?

A norm is a function that assigns a non-negative length or magnitude to a vector. It measures how "large" a vector is.

A function `f: R^n -> R` is a norm if it satisfies three properties:

1. **Positive definiteness:** `f(x) >= 0` and `f(x) = 0` if and only if `x = 0`
2. **Absolute homogeneity:** `f(alpha * x) = |alpha| * f(x)` for any scalar alpha
3. **Triangle inequality:** `f(x + y) <= f(x) + f(y)`

The triangle inequality is the most important. It says that going directly from point A to point C is never longer than going from A to B and then B to C. This property is what makes a norm a norm, and it is the reason norms define a consistent geometry.

### The Vector p-Norms

The L-p norm generalizes almost all common norms into a single formula:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

For different values of p, you get fundamentally different geometries:

**L1 norm (p=1):** `||x||_1 = |x_1| + |x_2| + ... + |x_n|`

The sum of absolute values. Also called Manhattan distance, taxicab norm, or grid distance. It measures distance along axes, like a city grid. Unit circle: a diamond shape.

**L2 norm (p=2):** `||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)`

The Euclidean norm. This is what most people mean by "distance." It is the straight-line distance. Unit circle: a circle.

**L-infinity norm (p -> inf):** `||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)`

The maximum absolute value among all components. Also called the Chebyshev distance or sup norm. It measures how far apart the most different component is. Unit circle: a square.

**L0 "norm" (p -> 0):** `||x||_0 = number of non-zero components`

Not a true norm (violates absolute homogeneity), but heavily used in sparsity and compression. The L0 "norm" counts non-zero entries, used in the Lasso regularization (approximated by L1, since L0 optimization is NP-hard).

### The Unit Circle Geometry

The unit circle `{x: ||x||_p = 1}` visualizes the geometry each norm defines:

```
p=1:   A diamond centered at origin, corners at (1,0), (0,1), (-1,0), (0,-1)
p=1.5: A rounded diamond, slightly bulging toward a circle
p=2:   A perfect circle, radius 1
p=4:   Nearly a rounded square, more bulging than L2
p=inf: A perfect square with sides parallel to axes, corners at (+-1, +-1)
```

As p increases, the unit circle morphs from a spiky diamond (p=1) through a circle (p=2) to a square (p=inf). This tells you how each norm treats vectors with mixed vs concentrated magnitudes.

Example: compare `x = (1, 0)` and `y = (0.707, 0.707)`.

```
||x||_2 = 1.000, ||y||_2 = 1.000
||x||_1 = 1.000, ||y||_1 = 1.414
||x||_inf = 1.000, ||y||_inf = 0.707
```

Under L2, they are the same length. Under L1, x is shorter (its mass is concentrated on one axis). Under L-infinity, y is shorter (its spread keeps each component lower).

### How Norms Show Up in Machine Learning

**L2 regularization (weight decay):** Adds `lambda * ||W||_2^2` to the loss. Encourages weights to be small. The squared L2 norm is differentiable everywhere (unlike L1, which has a kink at zero).

**L1 regularization (Lasso):** Adds `lambda * ||W||_1` to the loss. Encourages sparsity in weights. The kink at zero means weights are pushed exactly to zero, not just made small.

**L2 distance as loss:** Mean squared error `(1/n) * sum((y_pred - y_true)^2)` is the squared L2 norm of the error vector.

**L1 distance as loss:** Mean absolute error `(1/n) * sum(|y_pred - y_true|)` is the L1 norm of the error vector, averaged.

**Gradient clipping by norm:** If `||gradient||_2 > threshold`, scale the gradient to `threshold / ||gradient||_2`. Ensures the update step is never larger than `threshold` in L2 distance.

### Distances Derived from Norms

Every norm induces a distance metric:

```
d(x, y) = ||x - y||_p
```

**Euclidean distance (L2):** `d(x, y) = sqrt(sum((x_i - y_i)^2))`

Straight-line distance. The default choice for continuous data. Sensitive to outliers (squaring amplifies large differences).

**Manhattan distance (L1):** `d(x, y) = sum(|x_i - y_i|)`

Distance along axis-aligned paths. Less sensitive to outliers than L2. Used in high-dimensional problems where L2 distances concentrate (the "curse of dimensionality").

**Chebyshev distance (L-inf):** `d(x, y) = max(|x_i - y_i|)`

Worst-case component difference. Used in game board distance metrics.

**Hamming distance:** Number of positions where two binary vectors differ.

```
d(1011, 1001) = 1 (only the third position differs)
```

Used in error correction, feature hashing, and binary embedding spaces.

**Cosine distance:** `1 - cos_similarity(x, y)`, where `cos_similarity = dot(x, y) / (||x|| * ||y||)`

Measures directional similarity, ignoring magnitude. Not a true metric (violates the triangle inequality). Standard for text embeddings where direction matters more than magnitude.

### The Inequality Trio

Three inequalities are the foundation of norm-based reasoning:

**Cauchy-Schwarz inequality:**

```
|dot(x, y)| <= ||x||_2 * ||y||_2
```

The absolute value of the dot product never exceeds the product of the L2 norms. Equality holds when x and y are linearly dependent (parallel). This inequality is the basis of cosine similarity (-1 to 1), the angle between vectors, and the correlation coefficient.

**Triangle inequality:**

```
||x + y||_p <= ||x||_p + ||y||_p
```

The direct path is never longer than the sum of two legs. This is the defining property of a norm. Everything that follows in geometry depends on it.

**Hölder's inequality:**

```
sum(|x_i * y_i|) <= ||x||_p * ||y||_q    where 1/p + 1/q = 1
```

Generalizes Cauchy-Schwarz (which is the special case p=q=2). Connects different norms. For p=1, q=inf: `sum(|x_i * y_i|) <= ||x||_1 * ||y||_inf`.

### Norm Equivalence

In finite dimensions, all norms are equivalent. This is a theorem: for any two norms ||.||_a and ||.||_b, there exist positive constants c and C such that:

```
c * ||x||_a <= ||x||_b <= C * ||x||_a
```

This means that convergence in one norm implies convergence in all norms. But the constants matter:

```
||x||_2 <= ||x||_1 <= sqrt(n) * ||x||_2
||x||_inf <= ||x||_2 <= ||x||_inf
||x||_inf <= ||x||_1 <= n * ||x||_inf
```

In high dimensions, the ratio between norms can be large. `||x||_1` can be up to `sqrt(n)` times larger than `||x||_2`. This matters for optimization: L1 regularization has very different effects than L2.

### Norms of Matrices

The matrix norm induced by a vector norm is:

```
||A||_p = max_{||x||_p = 1} ||Ax||_p
```

**Spectral norm (p=2):** The largest singular value of A. Measures how much the matrix can stretch a vector in the L2 sense. This is the most important matrix norm for ML.

**Frobenius norm:** `sqrt(sum(A_ij^2))`. Like the L2 norm of the matrix treated as a vector. Not induced by a vector norm, but used extensively in regularization and matrix completion.

The spectral norm appears in:
- Lipschitz constants of neural network layers
- Stability of GAN training (spectral normalization)
- Condition number of linear systems `Ax = b`

### Why Distance Metrics Matter

The choice of distance metric determines:
1. **Who is a neighbor.** Different metrics give different nearest neighbors.
2. **What is clustered together.** K-means with L1 distance gives different clusters than with L2.
3. **How gradients flow.** L2 loss (MSE) is smooth; L1 loss (MAE) has a non-differentiable point at zero.
4. **What sparsity structure emerges.** L1 induces sparsity; L2 does not.

The rule of thumb:
- Use L2 for smooth, continuous data where magnitude matters
- Use L1 for high-dimensional data or when robustness to outliers is needed
- Use cosine for data where direction matters more than magnitude (text, embeddings, normalized features)
- Use Hamming for binary or discrete features

## Build It

### Step 1: Vector norms from scratch

```python
import math

def norm_l1(x):
    return sum(abs(v) for v in x)

def norm_l2(x):
    return math.sqrt(sum(v ** 2 for v in x))

def norm_linf(x):
    return max(abs(v) for v in x)

def norm_p(x, p):
    return sum(abs(v) ** p for v in x) ** (1 / p)
```

### Step 2: Distance metrics

```python
def euclidean_distance(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))

def chebyshev_distance(x, y):
    return max(abs(a - b) for a, b in zip(x, y))

def hamming_distance(x, y):
    if len(x) != len(y):
        raise ValueError("Vectors must be same length")
    return sum(a != b for a, b in zip(x, y))

def cosine_similarity(x, y):
    dot = sum(a * b for a, b in zip(x, y))
    nx = math.sqrt(sum(a ** 2 for a in x))
    ny = math.sqrt(sum(b ** 2 for b in y))
    if nx == 0 or ny == 0:
        return 0.0
    return dot / (nx * ny)
```

### Step 3: Verify inequalities

```python
import random

def verify_cauchy_schwarz(x, y):
    dot = sum(a * b for a, b in zip(x, y))
    nx = math.sqrt(sum(a ** 2 for a in x))
    ny = math.sqrt(sum(b ** 2 for b in y))
    return abs(dot) <= nx * ny + 1e-10

def verify_triangle_inequality(x, y, p=2):
    sum_norm = norm_p([a + b for a, b in zip(x, y)], p)
    x_norm = norm_p(x, p)
    y_norm = norm_p(y, p)
    return sum_norm <= x_norm + y_norm + 1e-10
```

### Step 4: Nearest neighbor search

```python
def nearest_neighbor(query, points, metric='euclidean'):
    best_idx = -1
    best_dist = float('inf')
    for i, p in enumerate(points):
        if metric == 'euclidean':
            d = euclidean_distance(query, p)
        elif metric == 'manhattan':
            d = manhattan_distance(query, p)
        elif metric == 'cosine':
            d = 1 - cosine_similarity(query, p)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist
```

## Use It

The all implementations from `code/distances.py` include complete vectorized versions:

```python
import math
import random

def dot_product(x, y):
    return sum(a * b for a, b in zip(x, y))

def norm_l1(x):
    return sum(abs(v) for v in x)

def norm_l2(x):
    return math.sqrt(sum(v ** 2 for v in x))

def norm_linf(x):
    return max(abs(v) for v in x)

def norm_p(x, p):
    return sum(abs(v) ** p for v in x) ** (1.0 / p)

def euclidean_distance(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))

def chebyshev_distance(x, y):
    return max(abs(a - b) for a, b in zip(x, y))

def hamming_distance(x, y):
    if len(x) != len(y):
        raise ValueError("Vectors must be same length")
    return sum(a != b for a, b in zip(x, y))

def cosine_similarity(x, y):
    dot = sum(a * b for a, b in zip(x, y))
    nx = math.sqrt(sum(a ** 2 for a in x))
    ny = math.sqrt(sum(b ** 2 for b in y))
    if nx == 0 or ny == 0:
        return 0.0
    return dot / (nx * ny)

def minkowski_distance(x, y, p):
    return sum(abs(a - b) ** p for a, b in zip(x, y)) ** (1.0 / p)

def normalized_correlation(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    dx = [a - mx for a in x]
    dy = [b - my for b in y]
    return cosine_similarity(dx, dy)

def nearest_neighbor(query, points, metric='euclidean'):
    best_idx = -1
    best_dist = float('inf')
    for i, p in enumerate(points):
        if metric == 'euclidean':
            d = euclidean_distance(query, p)
        elif metric == 'manhattan':
            d = manhattan_distance(query, p)
        elif metric == 'chebyshev':
            d = chebyshev_distance(query, p)
        elif metric == 'cosine':
            d = 1 - cosine_similarity(query, p)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist

def verify_cauchy_schwarz(x, y):
    dot = sum(a * b for a, b in zip(x, y))
    nx = math.sqrt(sum(a ** 2 for a in x))
    ny = math.sqrt(sum(b ** 2 for b in y))
    return abs(dot) <= nx * ny + 1e-10

def verify_triangle_inequality(x, y, p=2):
    z = [a + b for a, b in zip(x, y)]
    return norm_p(z, p) <= norm_p(x, p) + norm_p(y, p) + 1e-10

def verify_holder(x, y, p, q):
    sum_abs = sum(abs(a * b) for a, b in zip(x, y))
    combo = norm_p(x, p) * norm_p(y, q)
    return sum_abs <= combo + 1e-10
```

## Ship It

This lesson produces `code/distances.py` with all norm, distance, and inequality verification functions. These are used directly in Phase 3 for k-NN classification, Phase 4 for attention mechanisms (dot product as similarity), and throughout the ML curriculum whenever distance or similarity is needed.

## Exercises

1. **Inequality verifier.** Generate 100 random 5-dimensional vectors. Count how many satisfy Cauchy-Schwarz, the triangle inequality for L1 and L-infinity, and Hölder's inequality with p=3, q=1.5.

2. **Nearest neighbor comparison.** Generate synthetic data with 3 clusters. For each point, find the nearest neighbor using L1, L2, and cosine distance. Count how often each metric gives a different neighbor.

3. **Norm ratio in high dimensions.** Generate random vectors of dimension n = 10, 100, 1000, 10000. Compute `||x||_1 / ||x||_2` and `||x||_inf / ||x||_2` and verify they match the theoretical bounds: `1 <= ||x||_1 / ||x||_2 <= sqrt(n)` and `1/sqrt(n) <= ||x||_inf / ||x||_2 <= 1`.

4. **Cosine vs Euclidean for normalized vectors.** Show that for unit vectors (`||x||_2 = ||y||_2 = 1`), the relationship `||x - y||_2^2 = 2 - 2*cos_similarity(x, y)` holds. This means ordering by Euclidean distance is the same as ordering by cosine distance when vectors are normalized.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| L1 norm | "Absolute value sum" | Sum of absolute values of vector components. Unit circle is a diamond. Robust to outliers. |
| L2 norm | "Euclidean length" | Square root of sum of squares. Unit circle is a circle. The standard distance metric. |
| L-infinity norm | "Maximum component" | The largest absolute value among components. Unit circle is a square. |
| Frobenius norm | "Matrix L2" | Square root of sum of squared matrix entries. Like L2 norm on the flattened matrix. |
| Spectral norm | "Largest singular value" | Maximum stretching factor of a matrix in the L2 sense. Controls Lipschitz constants. |
| Cauchy-Schwarz | "Dot product bound" | |dot(x,y)| <= ||x||_2 * ||y||_2. Equality when vectors are parallel. |
| Triangle inequality | "Direct path is shortest" | ||x+y|| <= ||x|| + ||y||. The defining property of a norm. |
| Hölder's inequality | "Generalized dot bound" | sum(|x_i*y_i|) <= ||x||_p * ||y||_q where 1/p+1/q=1. |
| Cosine similarity | "Direction match" | dot(x,y) / (||x||_2 * ||y||_2). Ranges from -1 (opposite) to 1 (same direction). |
| Minkowski distance | "Generalized p-distance" | (sum(|x_i - y_i|^p))^(1/p). Unifies L1, L2, L-inf into one formula. |
| Hamming distance | "Bit difference count" | Number of positions where two binary vectors differ. Used for binary codes. |
| Norm equivalence | "All norms are similar" | In finite dimensions, c*||x||_a <= ||x||_b <= C*||x||_a. Convergence in one implies convergence in all. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/14-norms-and-distances)
