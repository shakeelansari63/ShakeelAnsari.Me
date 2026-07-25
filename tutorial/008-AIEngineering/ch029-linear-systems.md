# Linear Systems

> Most of machine learning is solving linear systems you cannot directly solve, and pretending nonlinear systems are linear.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04, 06  
**Time:** ~120 minutes  

## Learning Objectives

- Implement Gaussian elimination, LU decomposition, and Cholesky decomposition from scratch
- Solve linear systems using forward and backward substitution
- Compute and interpret matrix condition numbers
- Distinguish the tradeoffs between direct and iterative solvers

## The Concept

### The Problem

A linear system is a set of equations:

```
a_11 * x_1 + a_12 * x_2 + ... + a_1n * x_n = b_1
a_21 * x_1 + a_22 * x_2 + ... + a_2n * x_n = b_2
...
a_n1 * x_1 + a_n2 * x_2 + ... + a_nn * x_n = b_n
```

In matrix form: `Ax = b`, where `A` is an n x n matrix, `x` is an n x 1 unknown vector, and `b` is an n x 1 right-hand side.

Solving `Ax = b` means finding `x` such that the matrix equation holds. This is the most fundamental computation in numerical linear algebra. It appears in:

- Linear regression: solving `(X^T X) beta = X^T y`
- Newton's method: solving `H * delta = -gradient`
- Differential equations: solving `A * u = f` for discretized PDEs
- Graph algorithms: solving Laplacian systems for spectral clustering
- Reinforcement learning: solving `(I - gamma * P) * V = R` for value functions

### When Does a Solution Exist?

A solution exists and is unique if and only if `A` is invertible (non-singular). Equivalent conditions:

- `det(A) != 0`
- `A` has full rank (rank = n)
- All columns/rows of `A` are linearly independent
- `A` has no zero eigenvalues
- The linear transformation represented by `A` is bijective

If `A` is singular, either no solution exists or infinitely many solutions exist (if `b` is in the column space of `A`).

For rectangular systems (more equations than unknowns, or vice versa), we solve in the least-squares sense: minimize `||Ax - b||_2^2`. The normal equations `(A^T A) x = A^T b` convert this to a square system.

### Gaussian Elimination

The workhorse algorithm for solving `Ax = b`. It transforms `A` into an upper triangular matrix (row echelon form) using three row operations:

1. **Swap** two rows
2. **Multiply** a row by a non-zero scalar
3. **Add** a multiple of one row to another row

These operations do not change the solution `x`. The algorithm:

```
1. Forward elimination:
   For column k = 1 to n-1:
     Find pivot: largest absolute value in column k from rows k to n
     If pivot is zero, the matrix is singular
     Swap pivot row to row k
     For row i = k+1 to n:
       factor = A[i][k] / A[k][k]
       For col j = k to n:
         A[i][j] -= factor * A[k][j]
       b[i] -= factor * b[k]

2. Back substitution:
   For row i = n down to 1:
     x[i] = (b[i] - sum from j=i+1 to n of A[i][j] * x[j]) / A[i][i]
```

The computational cost is `O(n^3)` for the forward elimination, `O(n^2)` for the back substitution. For n=1000, that is about 1 billion operations.

**Partial pivoting** (swapping rows to put the largest available element on the diagonal) is essential for numerical stability. Without pivoting, dividing by a small pivot amplifies rounding errors catastrophically.

**Full pivoting** (swapping both rows and columns) is more stable but rarely used. Partial pivoting is sufficient in practice.

### LU Decomposition

Gaussian elimination computes the LU decomposition as a byproduct: factor `A` into a lower triangular matrix `L` and an upper triangular matrix `U` such that `A = P * L * U`, where `P` is a permutation matrix recording row swaps.

Once you have `LU = P*A`, solving `Ax = b` is two steps:

```
1. Solve L * y = P * b    (forward substitution, O(n^2))
2. Solve U * x = y        (back substitution, O(n^2))
```

The cost of the LU decomposition is `O(n^3)`, same as Gaussian elimination. But once computed, you can solve for many different `b` vectors at only `O(n^2)` per solve. This is the key advantage of LU: factor once, solve many times.

**When to use LU:**
- Multiple right-hand sides `b`
- `A` is not symmetric or positive definite
- `A` is a general square matrix (no special structure)

### Cholesky Decomposition

For symmetric positive definite (SPD) matrices, Cholesky is twice as fast as LU:

```
A = L * L^T
```

where `L` is lower triangular with positive diagonal entries. No pivoting is needed (SPD matrices are well-behaved).

The algorithm:

```
For j = 1 to n:
  L[j][j] = sqrt(A[j][j] - sum from k=1 to j-1 of L[j][k]^2)
  For i = j+1 to n:
    L[i][j] = (A[i][j] - sum from k=1 to j-1 of L[i][k] * L[j][k]) / L[j][j]
```

The cost is `(1/3)n^3` operations, half of LU's `(2/3)n^3`. It is also more numerically stable because SPD matrices have positive eigenvalues.

**When to use Cholesky:**
- `A` is symmetric positive definite (check: all eigenvalues > 0, or all leading principal minors > 0)
- Linear regression: `(X^T X)` is SPD
- Gaussian process regression
- Kalman filters

### Condition Number

The condition number `kappa(A)` measures how sensitive the solution `x` is to changes in `b` or `A`.

```
kappa(A) = ||A|| * ||A^(-1)||
```

For the L2 norm: `kappa(A) = sigma_max / sigma_min`, the ratio of largest to smallest singular value.

Interpretation:
- `kappa ≈ 1`: well-conditioned. Small changes in `b` cause proportionally small changes in `x`.
- `kappa >> 1`: ill-conditioned. Small changes in `b` (or small rounding errors) cause large changes in `x`.
- `kappa > 10^8`: nearly singular. The computed solution may be meaningless due to floating point errors.

Loss of precision: for `kappa(A) ≈ 10^k`, you lose approximately k digits of accuracy. In float64 (15-16 digits), `kappa = 10^8` means you trust about 7-8 digits.

**Rule of thumb:** if `log10(kappa(A))` is close to the number of mantissa digits in your float format (7 for float32, 15 for float64), your solution has no reliable digits.

### Direct vs Iterative Solvers

| Method | Cost | Memory | Best for |
|--------|------|--------|----------|
| Gaussian elimination | O(n^3) | O(n^2) | n < 10,000, dense matrices |
| LU decomposition | O(n^3) + O(n^2) per solve | O(n^2) | Multiple RHS, dense |
| Cholesky | (1/3)n^3 | O(n^2) | SPD dense |
| Conjugate Gradient | O(n * k) per iteration | O(n) | SPD sparse |
| GMRES | O(n * k^2) per iteration | O(n * k) | Non-symmetric sparse |

where `k` is the number of iterations (related to the condition number).

**When to use direct methods:**
- Matrix is small to moderate (n < 10,000)
- Matrix is dense (few zeros)
- You need a direct, reliable solution
- You are solving for many RHS vectors

**When to use iterative methods:**
- Matrix is large and sparse (n > 100,000)
- You only need an approximate solution
- The matrix is too large to factor (memory constraints)
- The matrix-vector product is cheap to compute

### Sparse Direct Methods

For sparse matrices (most entries are zero), fill-in during factorization destroys the sparsity. A matrix with 1 million nonzeros can generate 1 billion nonzeros in its LU factors.

Sparse direct methods:
- **Reordering:** Permute rows and columns to minimize fill-in (AMD, METIS, nested dissection)
- **Symbolic factorization:** Determine the nonzero pattern of the factors without computing numbers
- **Numerical factorization:** Compute L and U entries only where they are non-zero

These methods are highly engineered (UMFPACK, SuperLU, CHOLMOD, MUMPS) and are the standard for moderate-sized sparse systems.

For very large sparse systems (millions of unknowns), you typically switch to iterative methods that use matrix-vector products without forming the factor matrices.

### Iterative Methods: Conjugate Gradient

For SPD matrices, the Conjugate Gradient method solves `Ax = b` by minimizing the quadratic form `(1/2)x^T A x - b^T x`.

```
x_0 = initial guess
r_0 = b - A * x_0
p_0 = r_0

For k = 0, 1, ..., until convergence:
  alpha_k = (r_k^T r_k) / (p_k^T A p_k)
  x_{k+1} = x_k + alpha_k * p_k
  r_{k+1} = r_k - alpha_k * A * p_k
  beta_{k+1} = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
  p_{k+1} = r_{k+1} + beta_{k+1} * p_k
```

Key properties:
- Converges in at most n iterations in exact arithmetic (theoretically)
- In practice, converges in `O(sqrt(kappa))` iterations with good preconditioning
- Each iteration requires one matrix-vector product and a few vector operations
- The residuals are orthogonal (conjugate directions)

**Preconditioning:** Transform the system to improve the condition number.

```
M^(-1) * A * x = M^(-1) * b
```

where `M` is the preconditioner. Good preconditioners are cheap to apply and make `M^(-1) * A` have a lower condition number than `A`.

Common preconditioners:
- Jacobi (diagonal): `M = diag(A)` — cheap but weak
- Incomplete Cholesky: approximate Cholesky that drops small entries
- SSOR: symmetric successive over-relaxation
- Multigrid: for PDE-based problems

### GMRES (Generalized Minimal Residual)

For non-symmetric matrices, the Conjugate Gradient method does not work (the residuals cannot be made orthogonal with a short recurrence). GMRES is the standard iterative method for non-symmetric systems.

GMRES builds an orthogonal basis for the Krylov subspace `K_k = {r_0, A*r_0, A^2*r_0, ..., A^(k-1)*r_0}` and finds the best approximation in that subspace.

The cost per iteration grows linearly (`O(n * k)`) because of the growing orthogonal basis. For long runs, restart GMRES (e.g., GMRES(30) restarts after 30 iterations). This limits memory but can slow convergence.

### Matrix Inversion

**Never explicitly invert a matrix unless you absolutely need the inverse itself.**

Solving `Ax = b` via `x = A^(-1) * b` is:
- 3x more expensive (O(n^3) to invert, plus O(n^2) to multiply)
- Less numerically stable
- Computes n^2 numbers you do not need

Always solve the linear system directly instead of inverting.

When you need the inverse (e.g., for computing covariance matrices in statistics), use `solve(A, I)` — solve for each column of the identity matrix — rather than computing the inverse formula.

## Build It

### Step 1: Gaussian elimination with partial pivoting

```python
def gaussian_elimination(A, b):
    n = len(A)
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (aug[i][n] - total) / aug[i][i]
    return x
```

### Step 2: LU decomposition

```python
def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for k in range(n):
        U[k][k] = A[k][k] - sum(L[k][s] * U[s][k] for s in range(k))
        for i in range(k + 1, n):
            L[i][k] = (A[i][k] - sum(L[i][s] * U[s][k] for s in range(k))) / U[k][k]
        for j in range(k + 1, n):
            U[k][j] = (A[k][j] - sum(L[k][s] * U[s][j] for s in range(k)))
    return L, U
```

### Step 3: Forward and backward substitution

```python
def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def back_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x
```

## Use It

The all implementations from `code/linear_systems.py` include complete functions:

```python
import math

def gaussian_elimination(A, b):
    n = len(A)
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += aug[i][j] * x[j]
        x[i] = (aug[i][n] - total) / aug[i][i]
    return x

def solve_pentadiagonal(A, b):
    n = len(A)
    alpha = [0.0] * n
    beta = [0.0] * n
    gamma = [0.0] * (n - 1)
    for i in range(n):
        alpha[i] = A[i][i]
        if i < n - 1:
            gamma[i] = A[i][i + 1]
        if i > 0:
            beta[i] = A[i][i - 1]
    c_prime = [0.0] * (n - 1)
    d_prime = [0.0] * n
    c_prime[0] = gamma[0] / alpha[0]
    d_prime[0] = b[0] / alpha[0]
    for i in range(1, n):
        denom = alpha[i] - beta[i] * c_prime[i - 1]
        if abs(denom) < 1e-12:
            raise ValueError("Matrix is singular")
        if i < n - 1:
            c_prime[i] = gamma[i] / denom
        d_prime[i] = (b[i] - beta[i] * d_prime[i - 1]) / denom
    x = [0.0] * n
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x

def residual(A, x, b):
    n = len(A)
    r = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]
        r[i] = b[i] - s
    return r

def residual_norm(A, x, b):
    r = residual(A, x, b)
    return math.sqrt(sum(v ** 2 for v in r))

def matrix_vector_mult(A, v):
    n = len(A)
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]

def conjugate_gradient(A, b, max_iter=1000, tol=1e-10):
    n = len(b)
    x = [0.0] * n
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    p = r[:]
    rs_old = sum(ri ** 2 for ri in r)
    for _ in range(max_iter):
        Ap = matrix_vector_mult(A, p)
        alpha = rs_old / sum(p[i] * Ap[i] for i in range(n))
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rs_new = sum(ri ** 2 for ri in r)
        if math.sqrt(rs_new) < tol:
            break
        for i in range(n):
            p[i] = r[i] + (rs_new / rs_old) * p[i]
        rs_old = rs_new
    return x

def condition_number_estimate(A):
    n = len(A)
    x = [1.0] * n
    A_norm = max(sum(abs(A[i][j]) for j in range(n)) for i in range(n))
    for _ in range(20):
        Ax = matrix_vector_mult(A, x)
        x_norm = math.sqrt(sum(v ** 2 for v in Ax))
        x = [v / x_norm for v in Ax]
    A_inv_norm = math.sqrt(sum(v ** 2 for v in x))
    return A_norm * A_inv_norm
```

## Ship It

This lesson produces `code/linear_systems.py` with Gaussian elimination, LU decomposition, forward/backward substitution, tridiagonal/pentadiagonal solvers, and iterative methods. These reappear in Phase 2 for linear regression, Phase 3 for optimization, and Phase 4 for spectral methods.

## Exercises

1. **Gaussian elimination stability.** Solve the system `A*x = b` where `A[i][j] = 1/(i+j+1)` (Hilbert matrix, known to be extremely ill-conditioned). For n = 5, 10, 15, compute the solution using Gaussian elimination with and without partial pivoting. Measure the residual norm `||Ax - b||`. At what n does the solution become useless?

2. **LU for multiple RHS.** Generate a random 100x100 matrix A and 10 random right-hand sides b_1, ..., b_10. Compare the time to solve all 10 systems using: (a) Gaussian elimination from scratch for each, (b) LU decomposition once + forward/back substitution for each.

3. **Cholesky verification.** Generate a random 50x50 SPD matrix (use A = M^T * M for random M). Solve a linear system using Cholesky decomposition and verify `L*L^T - A` is close to zero.

4. **Conjugate Gradient vs Direct.** For a 1000x1000 SPD tridiagonal matrix (diagonal = 4, off-diagonal = -1), solve using Cholesky and Conjugate Gradient. Compare the solution time and residual. How does the CG iteration count relate to the condition number?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gaussian elimination | "Row reduction" | Transforming A into upper triangular form using row operations. The standard direct solver. O(n^3). |
| Pivot | "The dividing element" | The diagonal element used for elimination. A small pivot amplifies rounding errors. Partial pivoting selects the largest available element. |
| LU decomposition | "A = PLU" | Factor A into lower triangular L, upper triangular U, and permutation P. Solve Ax=b in O(n^2) after O(n^3) factorization. |
| Cholesky decomposition | "A = LL^T" | For symmetric positive definite matrices only. Twice as fast as LU. No pivoting needed. |
| Forward substitution | "Solve Ly = b" | Solving a lower triangular system from top to bottom. O(n^2). |
| Back substitution | "Solve Ux = y" | Solving an upper triangular system from bottom to top. O(n^2). |
| Condition number | "Sensitivity measure" | Ratio of largest to smallest singular value. kappa ≈ 10^k means you lose k digits of precision in the solution. |
| Ill-conditioned | "Sensitive to errors" | Small changes in input cause large changes in solution. Condition number is large. |
| Conjugate Gradient | "Iterative SPD solver" | Iterative method for SPD matrices. Converges in O(sqrt(kappa)) iterations. Each iteration is O(n^2). |
| GMRES | "Iterative general solver" | Iterative method for non-symmetric matrices. Cost per iteration grows. Usually restarted. |
| Preconditioner | "Conditioning improvement" | A matrix M such that M^(-1)A has lower condition number than A. The key to fast iterative solvers. |
| Sparse matrix | "Mostly zeros" | Matrix where most entries are zero. Stored in special formats (CSR, CSC, COO). Requires specialized solvers. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/17-linear-systems)
