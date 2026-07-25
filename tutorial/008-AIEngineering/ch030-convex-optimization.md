# Convex Optimization

> If you can make your problem convex, you have solved it. If you cannot, you are doing approximate inference.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-08, 17  
**Time:** ~120 minutes  

## Learning Objectives

- Verify convexity of functions and sets using definitions and epigraphs
- Derive and implement gradient descent, Newton's method, and coordinate descent
- Derive the KKT conditions for constrained optimization
- Apply Lagrangian duality to derive the dual form of SVM and lasso

## The Concept

### Why Convexity Matters

A convex optimization problem has two properties:

1. The objective function `f(x)` is convex
2. The feasible set (all constraint-satisfying points) is convex

When both hold, the problem is tractable:
- Every local minimum is a global minimum
- There is a complete duality theory
- Polynomial-time algorithms exist
- You can certify optimality

When either property fails (non-convex objective or non-convex constraints), you get:
- Multiple local minima
- No guarantee your optimizer found the best one
- NP-hardness in the worst case

Deep learning is entirely non-convex. Every loss landscape is riddled with saddle points and local minima. The fact that SGD works at all is a deep empirical mystery.

### Convex Sets

A set `C` is convex if for any `x, y` in `C` and any `theta in [0, 1]`:

```
theta * x + (1 - theta) * y ∈ C
```

The line segment between any two points in the set lies entirely in the set.

Examples of convex sets:
- Affine subspaces: `{x: Ax = b}`
- Halfspaces: `{x: a^T x <= b}`
- Polyhedra: intersection of finitely many halfspaces
- Norm balls: `{x: ||x|| <= r}`
- Positive semidefinite cone: `{X: X >= 0}` (symmetric positive semidefinite matrices)
- The empty set and singleton sets

Examples of non-convex sets:
- The set of integers
- A sphere surface (not filled ball)
- `{x: ||x|| = 1}` (the boundary, not the interior)
- The set of rank-1 matrices

### Convex Functions

A function `f` is convex if its domain is convex and for any `x, y` in the domain and `theta in [0, 1]`:

```
f(theta * x + (1 - theta) * y) <= theta * f(x) + (1 - theta) * f(y)
```

The function at any interpolated point is at most the interpolation of the function values. The line segment lies above the function.

**First-order condition (for differentiable f):**

```
f(y) >= f(x) + gradient_f(x)^T (y - x)
```

The function lies above its tangent plane at every point. This is the defining property used in optimization algorithms.

**Second-order condition (for twice differentiable f):**

```
Hessian_f(x) >= 0  (positive semidefinite)
```

The curvature is non-negative everywhere. This is how you check convexity in practice.

Examples of convex functions:
- Linear: `a^T x + b`
- Quadratic: `x^T Q x + b^T x` when `Q >= 0`
- Exponential: `exp(ax)` for any `a`
- Log-sum-exp: `log(sum(exp(x_i)))`
- Norms: `||x||` for any norm
- Max of convex functions

Examples of non-convex functions:
- `sin(x)`
- `x^3`
- `x^4 - x^2` (the double-well potential)
- Neural network loss landscapes

### The Epigraph

The epigraph of a function `f` is the set of points above its graph:

```
epi(f) = {(x, t): f(x) <= t}
```

A function is convex if and only if its epigraph is a convex set. This connects the set definition of convexity to the function definition.

The epigraph view is useful for:
- Proving convexity: show the epigraph is a convex set
- Converting optimization problems: minimizing `f` over `x` is equivalent to finding the lowest point of the epigraph
- Duality: the supporting hyperplane of the epigraph gives dual variables

### Gradient Descent

The simplest and most widely used optimization algorithm:

```
x_{k+1} = x_k - alpha_k * gradient_f(x_k)
```

**Step size choices:**
- Fixed: `alpha_k = alpha`. Simple, but may diverge or converge slowly.
- Diminishing: `alpha_k = 1/k`. Guaranteed convergence for convex functions if sum(alpha_k) = inf and sum(alpha_k^2) < inf.
- Exact line search: `alpha_k = argmin f(x_k - alpha * gradient)`. Theoretically nice, expensive in practice.
- Backtracking (Armijo): Start with `alpha = 1`, halve until `f(x_k - alpha * gradient) <= f(x_k) - c * alpha * ||gradient||^2`. Simple and effective.

**Convergence rate for convex functions:**
- Strongly convex: linear convergence (error shrinks geometrically)
- General convex: sublinear convergence (error shrinks as O(1/k))
- Non-convex: no guarantee of finding global minimum

The convergence rate depends on the condition number of the Hessian (ratio of largest to smallest eigenvalue). Poor conditioning creates narrow valleys where gradient descent zigzags.

### Gradient Descent with Momentum

Standard gradient descent can oscillate in narrow valleys. Momentum smooths the path:

```
v_{k+1} = beta * v_k + gradient_f(x_k)
x_{k+1} = x_k - alpha * v_{k+1}
```

`beta` is typically 0.9. The velocity `v` accumulates gradients over time, smoothing oscillations and accelerating convergence. This is the basis of SGD with momentum (SGDM), Adam, and most deep learning optimizers.

**Nesterov accelerated gradient:** Look ahead before computing the gradient.

```
v_{k+1} = beta * v_k + gradient_f(x_k - alpha * beta * v_k)
x_{k+1} = x_k - alpha * v_{k+1}
```

Nesterov achieves the optimal convergence rate O(1/k^2) for smooth convex optimization, versus O(1/k) for standard gradient descent.

### Newton's Method

Newton's method uses second-order information (the Hessian) for faster convergence:

```
x_{k+1} = x_k - Hessian_f(x_k)^(-1) * gradient_f(x_k)
```

**Properties:**
- Quadratic convergence near the optimum (error squaring at each step)
- Affine invariant (scaling variables does not change the iteration)
- Requires O(n^3) per step to factor the Hessian
- Works well when the Hessian is available and n is moderate (n < 10,000)

**When Newton excels:**
- The objective is smooth and strongly convex
- High precision is required (machine epsilon)
- n is small enough to compute Hessian (n < 10,000)

**When Newton fails:**
- n is large (O(n^3) per step is prohibitive)
- Hessian is not available (automatic differentiation can help)
- Hessian is not positive definite (Newton goes uphill)
- The starting point is far from the optimum (Newton can diverge)

**Quasi-Newton methods (BFGS, L-BFGS):** Approximate the Hessian using gradient differences. L-BFGS stores only the last m gradient differences (typically m=10-100), giving O(n) memory and O(n^2) per step. This is the default for deterministic optimization in ML.

### Coordinate Descent

Update one variable at a time while holding all others fixed:

```
For k = 1, 2, ...:
  Pick coordinate i
  x_i = argmin f(x_1, ..., x_i, ..., x_n)
```

**Properties:**
- O(n) per full cycle (update each coordinate once)
- Often faster than gradient descent when coordinates are cheap to update
- Can handle non-differentiable objectives (e.g., L1 regularization)
- May converge slowly if variables are strongly correlated

Coordinate descent is the workhorse for Lasso regression (L1-regularized linear models). The update for each coefficient has a closed form involving soft-thresholding.

### Constrained Optimization and KKT

The general constrained optimization problem:

```
minimize f(x)
subject to g_i(x) <= 0, i = 1, ..., m
         h_j(x) = 0,  j = 1, ..., p
```

The Lagrangian:

```
L(x, lambda, nu) = f(x) + sum(lambda_i * g_i(x)) + sum(nu_j * h_j(x))
```

where `lambda_i >= 0` are the KKT multipliers for inequality constraints and `nu_j` are multipliers for equality constraints.

The KKT conditions (necessary for optimality under constraint qualifications):

1. **Stationarity:** `0 in ∂L(x*, lambda*, nu*)` (gradient of Lagrangian = 0)
2. **Primal feasibility:** `g_i(x*) <= 0`, `h_j(x*) = 0`
3. **Dual feasibility:** `lambda_i* >= 0`
4. **Complementary slackness:** `lambda_i* * g_i(x*) = 0`

The last condition is the most informative: either the constraint is active (g_i = 0) or its multiplier is zero. If a constraint is not binding, it does not affect the solution.

### Lagrangian Duality

The Lagrangian dual function:

```
g(lambda, nu) = inf_x L(x, lambda, nu)
```

The dual problem:

```
maximize g(lambda, nu)
subject to lambda >= 0
```

**Weak duality:** `g(lambda, nu) <= f(x*)` for any feasible `(lambda, nu)` and `x*`. The dual gives a lower bound on the primal optimum.

**Strong duality:** `g(lambda*, nu*) = f(x*)`. The dual optimum equals the primal optimum. Holds for convex problems under Slater's condition (there exists a strictly feasible point).

Strong duality is the foundation of:
- SVM dual formulation (kernel trick)
- Support vector regression
- Lasso dual derivation
- Optimal transport (Wasserstein distance)
- Robust optimization

### The Dual of SVM

Primal SVM (hard margin):

```
minimize (1/2) * ||w||^2
subject to y_i * (w^T x_i + b) >= 1, for all i
```

Lagrangian:

```
L = (1/2)*||w||^2 - sum(alpha_i * (y_i * (w^T x_i + b) - 1))
```

where `alpha_i >= 0`. Setting gradients to zero gives KKT stationarity:

```
w = sum(alpha_i * y_i * x_i)
0 = sum(alpha_i * y_i)
```

Substituting back gives the dual:

```
maximize sum(alpha_i) - (1/2) * sum(sum(alpha_i * alpha_j * y_i * y_j * x_i^T x_j))
subject to alpha_i >= 0, sum(alpha_i * y_i) = 0
```

The primal solution is `w = sum(alpha_i * y_i * x_i)`. Only support vectors (points on the margin boundary) have `alpha_i > 0`.

The dual is a quadratic program and is independent of the dimension of `x`. This enables the kernel trick: replace `x_i^T x_j` with `K(x_i, x_j)` to get non-linear SVMs.

### The Dual of Lasso

Primal Lasso:

```
minimize (1/2) * ||y - X * beta||^2 + lambda * ||beta||_1
```

The L1 norm makes this non-differentiable. The dual is a box-constrained quadratic program:

```
maximize (1/2) * ||y - X * beta||^2 + (1/2) * ||y||^2 - (lambda^2/2) * ||theta||^2
subject to ||X^T theta||_inf <= lambda
```

Where `theta` are the dual variables. The dual is smooth (quadratic) with simple box constraints. This is why dual methods (like coordinate descent) work well for Lasso.

### Subgradients

For non-differentiable convex functions (like `|x|` or `||beta||_1`), the gradient does not exist everywhere. The subgradient generalizes the gradient:

A vector `g` is a subgradient of `f` at `x` if for all `y`:

```
f(y) >= f(x) + g^T (y - x)
```

The set of all subgradients at `x` is the subdifferential `∂f(x)`.

Examples:
- `f(x) = |x|`: `∂f(0) = [-1, 1]`
- `f(x) = max(0, x)` (ReLU): `∂f(0) = [0, 1]`
- `f(x) = ||x||_1`: `∂f(0) = {g: ||g||_inf <= 1}`

**Subgradient method:** `x_{k+1} = x_k - alpha_k * g_k` where `g_k` is any subgradient. Converges at rate O(1/sqrt(k)) for convex functions. This is what SGD does for non-differentiable objectives.

### Proximal Operators

For composite objectives `f(x) + g(x)` where `f` is smooth and `g` is non-smooth but simple, proximal methods are the standard approach.

The proximal operator of `g`:

```
prox_g(x) = argmin_u (g(u) + (1/2) * ||u - x||^2)
```

For many important `g`, the proximal operator has a closed form:
- `g(x) = lambda * ||x||_1`: prox is soft-thresholding: `S_lambda(x) = sign(x) * max(|x| - lambda, 0)`
- `g(x) = I_C(x)` (indicator of convex set C): prox is projection onto C

**ISTA (Iterative Shrinkage-Thresholding Algorithm):** For Lasso regression:

```
x_{k+1} = S_{lambda * alpha}(x_k - alpha * gradient_f(x_k))
```

where `S` is the soft-thresholding operator. This alternates between a gradient step on the smooth part and a proximal step on the L1 part.

**FISTA:** The accelerated version of ISTA with convergence rate O(1/k^2) instead of O(1/k). This is the default method for L1-regularized problems.

## Build It

### Step 1: Gradient descent

```python
def gradient_descent(f, grad_f, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    x = x0
    for _ in range(max_iter):
        g = grad_f(x)
        x_new = [x[i] - alpha * g[i] for i in range(len(x))]
        if math.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(len(x)))) < tol:
            break
        x = x_new
    return x

def backtracking_line_search(f, grad_f, x, direction, alpha=1.0, c=0.5, rho=0.5):
    while f([x[i] + alpha * direction[i] for i in range(len(x))]) > \
          f(x) + c * alpha * sum(grad_f(x)[i] * direction[i] for i in range(len(x))):
        alpha *= rho
    return alpha
```

### Step 2: Newton's method

```python
def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    x = x0
    for _ in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        delta = solve_linear_system(H, [-gi for gi in g])
        x_new = [x[i] + delta[i] for i in range(len(x))]
        if math.sqrt(sum(d ** 2 for d in delta)) < tol:
            break
        x = x_new
    return x
```

### Step 3: Coordinate descent for Lasso

```python
def soft_threshold(z, gamma):
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0

def coordinate_descent_lasso(X, y, lam, max_iter=10000, tol=1e-6):
    n, p = len(X), len(X[0])
    beta = [0.0] * p
    for _ in range(max_iter):
        beta_old = beta[:]
        for j in range(p):
            r_j = sum((y[i] - sum(X[i][k] * beta[k] for k in range(p))) for i in range(n))
            r_j += sum(X[i][j] * beta[j] for i in range(n))
            rho_j = sum(X[i][j] * r_j for i in range(n))
            z_j = sum(X[i][j] ** 2 for i in range(n))
            if z_j == 0:
                continue
            beta[j] = soft_threshold(rho_j / z_j, lam / z_j)
        if math.sqrt(sum((beta[i] - beta_old[i]) ** 2 for i in range(p))) < tol:
            break
    return beta
```

## Use It

The all implementations from `code/convex.py` include complete functions:

```python
import math

def is_convex_set(points):
    for x in points:
        for y in points:
            for t in [i / 10.0 for i in range(11)]:
                z = [t * x[i] + (1 - t) * y[i] for i in range(len(x))]
                if not any(all(abs(z[j] - q[j]) < 1e-10 for j in range(len(z))) for q in points):
                    return False
    return True

def is_convex_function(f, domain, n_checks=100):
    import random
    for _ in range(n_checks):
        x = [random.uniform(domain[0], domain[1]) for _ in range(2)]
        y = [random.uniform(domain[0], domain[1]) for _ in range(2)]
        t = random.random()
        lhs = f([t * x[i] + (1 - t) * y[i] for i in range(2)])
        rhs = t * f(x) + (1 - t) * f(y)
        if lhs > rhs + 1e-10:
            return False
    return True

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = list(x0)
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = [x[j] - learning_rate * grad[j] for j in range(len(x))]
        diff = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(len(x))))
        if diff < tol:
            break
        x = x_new
    return x

def gradient_descent_momentum(f, grad_f, x0, learning_rate=0.01, beta=0.9, max_iter=1000, tol=1e-6):
    x = list(x0)
    v = [0.0] * len(x)
    for i in range(max_iter):
        grad = grad_f(x)
        v = [beta * v[j] + learning_rate * grad[j] for j in range(len(x))]
        x_new = [x[j] - v[j] for j in range(len(x))]
        diff = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(len(x))))
        if diff < tol:
            break
        x = x_new
    return x

def nesterov_gradient_descent(f, grad_f, x0, learning_rate=0.01, beta=0.9, max_iter=1000, tol=1e-6):
    x = list(x0)
    v = [0.0] * len(x)
    for i in range(max_iter):
        lookahead = [x[j] - beta * v[j] for j in range(len(x))]
        grad = grad_f(lookahead)
        v = [beta * v[j] + learning_rate * grad[j] for j in range(len(x))]
        x_new = [x[j] - v[j] for j in range(len(x))]
        diff = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(len(x))))
        if diff < tol:
            break
        x = x_new
    return x

def backtracking_line_search(f, grad_f, x, direction, alpha=1.0, c=0.5, rho=0.5):
    n = len(x)
    f_x = f(x)
    grad_dot_dir = sum(grad_f(x)[i] * direction[i] for i in range(n))
    while f([x[i] + alpha * direction[i] for i in range(n)]) > f_x + c * alpha * grad_dot_dir:
        alpha *= rho
    return alpha

def newton_method(f, grad_f, hess_f, solve_system, x0, max_iter=100, tol=1e-6):
    x = list(x0)
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        delta = solve_system(hess, [-g for g in grad])
        x_new = [x[j] + delta[j] for j in range(len(x))]
        diff = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(len(x))))
        if diff < tol:
            break
        x = x_new
    return x

def coordinate_descent(f, grad_f_i, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    x = list(x0)
    for i in range(max_iter):
        x_old = list(x)
        for j in range(n):
            x[j] = grad_f_i(f, x, j)
        diff = math.sqrt(sum((x[j] - x_old[j]) ** 2 for j in range(n)))
        if diff < tol:
            break
    return x

def soft_threshold(z, gamma):
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0

def proximal_operator_l1(x, lam):
    return [soft_threshold(v, lam) for v in x]

def ista(A, b, lam, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    n = len(A[0])
    AtA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    Atb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    x = list(x0)
    for i in range(max_iter):
        grad = [sum(AtA[i][j] * x[j] for j in range(n)) - Atb[i] for i in range(n)]
        x_new = proximal_operator_l1([x[j] - alpha * grad[j] for j in range(n)], lam * alpha)
        diff = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(n)))
        if diff < tol:
            break
        x = x_new
    return x

def fista(A, b, lam, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    n = len(A[0])
    m = len(A)
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    Atb = [sum(A[k][i] * b[k] for k in range(m)) for i in range(n)]
    x = list(x0)
    y = list(x0)
    t = 1.0
    for i in range(max_iter):
        grad = [sum(AtA[i][j] * y[j] for j in range(n)) - Atb[i] for i in range(n)]
        x_new = proximal_operator_l1([y[j] - alpha * grad[j] for j in range(n)], lam * alpha)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = [x_new[j] + ((t - 1) / t_new) * (x_new[j] - x[j]) for j in range(n)]
        diff = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(n)))
        if diff < tol:
            break
        x = x_new
        t = t_new
    return x
```

## Ship It

This lesson produces `code/convex.py` with gradient descent, Newton, coordinate descent, ISTA, and FISTA implementations. These reappear in Phase 2 for linear regression fitting, Phase 3 for SVMs, and Phase 4 for training neural networks.

## Exercises

1. **Convexity checker.** Write a function that numerically checks whether a function of one variable is convex over an interval. Test it on `x^2` (convex), `|x|` (convex), `sin(x)` (not convex), and `x^3` (not convex).

2. **Gradient descent comparison.** Minimize `f(x, y) = 100*(y - x^2)^2 + (1 - x)^2` (Rosenbrock function) using: (a) gradient descent with fixed step size, (b) gradient descent with backtracking line search, (c) Newton's method. Compare the number of iterations and path taken.

3. **Coordinate descent for Lasso.** Generate synthetic data with n=100, p=20 where only 5 features are relevant. Fit a Lasso model using coordinate descent. Compare the estimated coefficients with the true coefficients. How does the regularization parameter lambda affect sparsity?

4. **ISTA vs FISTA.** For the same Lasso problem, run ISTA and FISTA. Plot the objective value vs iteration number for both. How many iterations does each need to reach the same accuracy?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Convex set | "No dents" | Line segment between any two points stays in the set. |
| Convex function | "Bowl-shaped" | Function lies below its chords. Every local minimum is global. |
| Epigraph | "Above the graph" | Set of points (x, t) where f(x) <= t. Function is convex iff epigraph is convex. |
| Gradient descent | "Follow the slope" | x = x - alpha * gradient. The simplest optimization algorithm. O(1/k) rate. |
| Newton's method | "Second-order optimizer" | x = x - H^(-1)*gradient. Quadratic convergence, but O(n^3) per step. |
| Momentum | "Smooth the gradient path" | Accumulate gradient over time to smooth oscillations. Beta = 0.9 is standard. |
| KKT conditions | "Optimality conditions" | Stationarity, primal/dual feasibility, complementary slackness. Necessary for constrained optima. |
| Lagrangian | "Augmented objective" | L(x, lambda) = f(x) + sum(lambda_i * g_i(x)). Encodes constraints into objective. |
| Dual problem | "The other problem" | Maximize inf_x L(x, lambda). Gives lower bound on primal. Strong duality = equal. |
| Subgradient | "Gradient for non-smooth functions" | Generalization of gradient for functions like |x|. ∂f(0) = [-1, 1]. |
| Proximal operator | "Regularized projection" | prox_g(x) = argmin_u g(u) + (1/2)||u - x||^2. Closed form for L1 (soft thresholding). |
| ISTA | "Iterative Shrinkage" | Proximal gradient for L1 problems. Alternates gradient step + soft thresholding. |
| FISTA | "Fast ISTA" | Accelerated ISTA with momentum. O(1/k^2) convergence instead of O(1/k). |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/18-convex-optimization)
