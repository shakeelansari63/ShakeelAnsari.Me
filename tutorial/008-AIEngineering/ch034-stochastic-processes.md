# Stochastic Processes

> Randomness is not the absence of structure. It is a different kind of structure.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04, 07, 15  
**Time:** ~120 minutes  

## Learning Objectives

- Simulate random walks, Gaussian processes, and Markov chains from scratch
- Compute transition matrices and stationary distributions for Markov chains
- Distinguish between stationary, weakly stationary, and non-stationary time series
- Explain the connection between stochastic processes and ML training dynamics

## The Concept

### What Is a Stochastic Process?

A stochastic process is a collection of random variables indexed by time: `{X_t: t in T}`.

- **Discrete time:** T = {0, 1, 2, ...}. Daily stock prices, number of customers per hour, token-by-token language model outputs.
- **Continuous time:** T = [0, inf). Radioactive decay, particle movement, continuous-time diffusion models.

Each `X_t` is a random variable. A specific sequence of observed values `x_0, x_1, x_2, ...` is a **realization** or **sample path** of the process.

### Random Walk

The simplest stochastic process:

```
X_0 = 0
X_t = X_{t-1} + epsilon_t, where epsilon_t ~ N(0, sigma^2) or epsilon_t = +/-1 with equal probability
```

Properties:
- `E[X_t] = 0` (mean is constant)
- `Var(X_t) = t * sigma^2` (variance grows linearly with time)
- Non-stationary: the distribution changes over time (variance increases)

The random walk is the foundation of:
- Brownian motion (continuous-time limit of random walk)
- Stochastic gradient descent (SGD adds noise to gradient updates)
- Stock price models (efficient market hypothesis)
- Diffusion models (forward process adds noise)

### Gaussian Process

A Gaussian Process (GP) is a collection of random variables where any finite subset has a joint Gaussian distribution. A GP is fully specified by:

- **Mean function:** `m(t) = E[X_t]` (what the process tends to do)
- **Covariance function (kernel):** `k(t, s) = Cov(X_t, X_s)` (how points relate across time)

Popular kernels:
- **RBF (squared exponential):** `k(t, s) = sigma^2 * exp(-(t-s)^2 / (2*l^2))`. Produces infinitely smooth functions.
- **Matérn:** `k(t, s) = sigma^2 * (1 + sqrt(3)*|t-s|/l) * exp(-sqrt(3)*|t-s|/l)`. Less smooth, more realistic for physical processes.
- **Periodic:** `k(t, s) = sigma^2 * exp(-2*sin^2(pi*|t-s|/p) / l^2)`. For periodic data.

GPs are used for:
- Bayesian optimization (hyperparameter tuning)
- Regression with uncertainty estimates
- Time series forecasting
- Neural network interpretation (infinite-width NNs converge to GPs)

### Stationarity

A process is **strictly stationary** if the joint distribution of `(X_{t_1}, ..., X_{t_k})` is the same as `(X_{t_1 + h}, ..., X_{t_k + h})` for any shift h. The process has no trend, no seasonality, no time-dependent structure.

A process is **weakly stationary (or covariance stationary)** if:

1. `E[X_t] = mu` (constant mean)
2. `Var(X_t) = sigma^2` (constant variance)
3. `Cov(X_t, X_{t+h}) = gamma(h)` (covariance depends only on lag h, not on absolute time)

Most practical stationarity checks test weak stationarity.

**Why stationarity matters for ML:**
- Non-stationary data causes train/test mismatch (model learns patterns that do not hold in the future)
- Most time series models require stationarity (ARIMA, spectral methods)
- Differencing (subtracting X_{t-1} from X_t) is the standard way to remove trends and make data stationary

Examples:
- White noise: stationary (mean 0, constant variance, no autocorrelation)
- Random walk: non-stationary (variance grows with time)
- Seasonal pattern with fixed amplitude: non-stationary (mean depends on time of year)
- Monthly sales with trend: non-stationary

### Markov Chains

A Markov chain is a stochastic process with the Markov property:

```
P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t)
```

The future depends only on the present, not on the past given the present.

**Transition matrix:** `P[i][j] = P(X_{t+1} = j | X_t = i)`. Rows sum to 1 (each row is a probability distribution over next states).

**State distribution at time t:** `pi_t[i] = P(X_t = i)`. The vector of probabilities over all states.

Evolution: `pi_{t+1} = pi_t * P`, or `pi_t = pi_0 * P^t`.

**Stationary distribution:** A distribution `pi` such that `pi = pi * P`. Once reached, the chain stays there.

Finding it: solve `pi * P = pi` with `sum(pi) = 1`. This is the left eigenvector of P with eigenvalue 1.

Conditions for a unique stationary distribution:
- **Irreducible:** Every state can be reached from every other state (the graph is strongly connected).
- **Aperiodic:** The chain is not trapped in cycles (return times have GCD = 1).

When both hold, the chain is **ergodic** and converges to the stationary distribution from any starting point.

The stationary distribution of a reversible Markov chain satisfies detailed balance:

```
pi_i * P[i][j] = pi_j * P[j][i]
```

**Applications in ML:**
- MCMC sampling: construct a Markov chain whose stationary distribution is the target distribution
- PageRank: stationary distribution of a random web surfer
- Language models: next-token prediction defines a Markov chain over token sequences (though with very many states)
- Reinforcement learning: Markov decision processes extend Markov chains with actions and rewards

### Autocorrelation

Autocorrelation measures how a time series is correlated with itself at different lags:

```
rho(h) = Cov(X_t, X_{t+h}) / Var(X_t)
```

- `rho(0) = 1` (correlation with itself at lag 0)
- `rho(1)`: correlation with the previous value (how persistent the process is)
- `rho(k)`: correlation with value k steps ago

For white noise: `rho(h) = 0` for all h != 0.

For a random walk: `rho(h) ≈ 1 - h/(2T)` (decays slowly).

For AR(1): `rho(h) = phi^h` (exponential decay).

The autocorrelation function (ACF) is a key diagnostic for time series analysis. It tells you:
- Is the data stationary? (Slow ACF decay suggests non-stationarity)
- What model order to use? (Spikes at specific lags suggest AR or MA terms)
- Is there seasonality? (Periodic ACF peaks)

### The Bias of SGD as a Stochastic Process

SGD is not just a faster way to do gradient descent. It is a stochastic process with different dynamics.

The update:

```
theta_{t+1} = theta_t - alpha * (gradient_f(theta_t) + noise_t)
```

where `noise_t` is the minibatch gradient minus the full-batch gradient.

This noise:
- Has covariance proportional to the empirical Fisher information matrix
- Is approximately Gaussian for large batch sizes (CLT)
- Can help escape sharp minima (the same way temperature helps in sampling)
- Creates a stationary distribution around the minimum (SGD does not converge to a point, but to a distribution)

This is why SGD generalizes better than full-batch GD. The noise acts as an implicit regularizer, biasing the solution toward flatter minima.

**Temperature and noise schedule:**
- High temperature (large learning rate, small batch): more exploration, coarser convergence
- Low temperature (small learning rate, large batch): less exploration, finer convergence
- Learning rate schedules = cooling schedules in simulated annealing

### Brownian Motion and Diffusion

Brownian motion (also called the Wiener process) is the continuous-time limit of a random walk.

Properties:
- `W_0 = 0`
- `W_t` is continuous (no jumps)
- `W_t - W_s ~ N(0, t - s)` (independent Gaussian increments)
- Non-differentiable everywhere (paths are fractal)

**Diffusion processes:**

```
dX_t = mu(X_t, t) * dt + sigma(X_t, t) * dW_t
```

- Drift `mu`: deterministic trend
- Diffusion `sigma`: random fluctuations scaled by Brownian motion

**Score-based diffusion models:** The forward process (data -> noise) is a diffusion:

```
dx = -beta_t * x * dt + sqrt(beta_t) * dw
```

The reverse process (noise -> data) is learned by a neural network:

```
dx = (-beta_t * x - sigma^2 * s_theta(x, t)) * dt + sigma * dw
```

where `s_theta` approximates the score function `gradient_x log p_t(x)`. This is the mathematical foundation of DALL-E, Stable Diffusion, and Sora.

### Ergodic Theory

An ergodic process has the property that time averages equal ensemble averages:

```
lim_{T -> inf} (1/T) * sum from t=1 to T of f(X_t) = E[f(X)]

for any function f (with some regularity conditions).
```

This means you can estimate properties of the process from a single long trajectory. Without ergodicity, you would need many independent realizations.

**Why this matters for ML:**
- Training loss averages over time should converge to expected loss
- Validation metrics are time averages along the training trajectory
- If the training process is non-ergodic (common in deep learning), different runs converge to different solutions with different generalization

### Poisson Process

Models the occurrence of rare events over time:

- Events occur independently
- The probability of an event in a small interval `dt` is `lambda * dt`
- The number of events in an interval of length T is Poisson(lambda * T)
- The time between events is Exponential(lambda)

**Applications in ML:**
- Modeling click streams, arrival times, and event sequences
- Temporal point processes for user behavior modeling
- Hawkes processes (self-exciting point processes) for social media dynamics

### Stochastic Recurrence and Training Dynamics

Many ML training algorithms are stochastic recurrences:

**Linear recurrence:** AR(1) model: `X_t = phi * X_{t-1} + epsilon_t`.

- If `|phi| < 1`: stationary (mean-reverting)
- If `|phi| = 1`: random walk (unit root)
- If `|phi| > 1`: explosive (diverges)

**Momentum SGD:**

```
v_{t+1} = beta * v_t + gradient_t
theta_{t+1} = theta_t - alpha * v_{t+1}
```

This is a second-order stochastic recurrence (two coupled equations). The behavior depends on the eigenvalues of the combined system.

**Training as a stochastic process:** The loss during training is a stochastic process. Understanding its dynamics (convergence rate, variance, autocorrelation) helps design better optimizers and learning rate schedules.

## Build It

### Step 1: Random walk simulation

```python
import math
import random

def random_walk(n_steps, p=0.5):
    steps = [1 if random.random() < p else -1 for _ in range(n_steps)]
    position = [0]
    for s in steps:
        position.append(position[-1] + s)
    return position

def gaussian_random_walk(n_steps, sigma=1.0):
    position = [0.0]
    for _ in range(n_steps):
        position.append(position[-1] + random.gauss(0, sigma))
    return position
```

### Step 2: Markov chain

```python
def markov_chain_step(P, current_state):
    r = random.random()
    cumulative = 0.0
    for next_state, prob in enumerate(P[current_state]):
        cumulative += prob
        if r < cumulative:
            return next_state
    return len(P) - 1

def simulate_markov_chain(P, initial_state, n_steps):
    states = [initial_state]
    for _ in range(n_steps):
        states.append(markov_chain_step(P, states[-1]))
    return states

def stationary_distribution(P, max_iter=10000, tol=1e-10):
    n = len(P)
    pi = [1.0 / n for _ in range(n)]
    for _ in range(max_iter):
        pi_new = [sum(pi[i] * P[i][j] for i in range(n)) for j in range(n)]
        diff = sum(abs(pi_new[i] - pi[i]) for i in range(n))
        pi = pi_new
        if diff < tol:
            break
    return pi
```

### Step 3: Gaussian process simulation

```python
import math
import random

def rbf_kernel(x1, x2, length=1.0, variance=1.0):
    return variance * math.exp(-((x1 - x2) ** 2) / (2 * length ** 2))

def simulate_gp(mean_func, kernel, x_points):
    n = len(x_points)
    K = [[kernel(x_points[i], x_points[j]) for j in range(n)] for i in range(n)]
    L = cholesky(K)
    z = [random.gauss(0, 1) for _ in range(n)]
    f = [mean_func(x_points[i]) + sum(L[i][j] * z[j] for j in range(n)) for i in range(n)]
    return f
```

### Step 4: Autocorrelation

```python
def autocorrelation(x, lag):
    n = len(x)
    mean_x = sum(x) / n
    numerator = sum((x[i] - mean_x) * (x[i + lag] - mean_x) for i in range(n - lag))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0

def autocorrelation_function(x, max_lag):
    return [autocorrelation(x, lag) for lag in range(max_lag + 1)]
```

## Use It

The all implementations from `code/stochastic.py` include complete functions:

```python
import math
import random

def random_walk_symmetric(n_steps=1000):
    walk = [0]
    for _ in range(n_steps):
        step = 1 if random.random() < 0.5 else -1
        walk.append(walk[-1] + step)
    return walk

def random_walk_gaussian(n_steps=1000, sigma=1.0):
    walk = [0.0]
    for _ in range(n_steps):
        walk.append(walk[-1] + random.gauss(0, sigma))
    return walk

def markov_chain_transition(P, state):
    r = random.random()
    cumulative = 0.0
    for next_state, prob in enumerate(P[state]):
        cumulative += prob
        if r < cumulative:
            return next_state
    return state

def simulate_markov_chain(P, initial_state, n_steps):
    states = [initial_state]
    for _ in range(n_steps):
        states.append(markov_chain_transition(P, states[-1]))
    return states

def stationary_distribution(P, max_iter=10000, tol=1e-10):
    n = len(P)
    pi = [1.0 / n for _ in range(n)]
    for it in range(max_iter):
        new_pi = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += pi[i] * P[i][j]
            new_pi[j] = s
        diff = sum(abs(new_pi[j] - pi[j]) for j in range(n))
        pi = new_pi
        if diff < tol:
            break
    return pi

def transition_matrix_to_power(P, power):
    n = len(P)
    result = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    base = P
    p = power
    while p > 0:
        if p % 2 == 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        p //= 2
    return result

def matrix_multiply(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k] != 0:
                for j in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    return C

def is_weakly_stationary(series, max_lag=10, alpha=0.05):
    n = len(series)
    mean1 = sum(series[:n//2]) / (n//2)
    mean2 = sum(series[n//2:]) / (n - n//2)
    var1 = sum((x - mean1)**2 for x in series[:n//2]) / (n//2)
    var2 = sum((x - mean2)**2 for x in series[n//2:]) / (n - n//2)
    if abs(mean1 - mean2) > 0.1 * math.sqrt(var1 + var2):
        return False
    if abs(var1 - var2) > 0.1 * (var1 + var2):
        return False
    return True

def autocorrelation(series, lag):
    n = len(series)
    m = sum(series) / n
    num = sum((series[t] - m) * (series[t - lag] - m) for t in range(lag, n))
    den = sum((series[t] - m) ** 2 for t in range(n))
    return num / den if den != 0 else 0.0

def autocorrelation_function(series, max_lag):
    return [autocorrelation(series, lag) for lag in range(max_lag + 1)]

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    return variance * math.exp(-((x1 - x2) ** 2) / (2 * length_scale ** 2))

def kernel_matrix(kernel, x_points):
    n = len(x_points)
    return [[kernel(x_points[i], x_points[j]) for j in range(n)] for i in range(n)]

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def simulate_gp(mean_func, kernel, x_points):
    n = len(x_points)
    K = [[kernel(x_points[i], x_points[j]) for j in range(n)] for i in range(n)]
    L = cholesky(K)
    z = [random.gauss(0, 1) for _ in range(n)]
    f = [mean_func(x_points[i]) + sum(L[i][j] * z[j] for j in range(n)) for i in range(n)]
    return f

def ar1_process(n, phi, sigma=1.0, x0=0.0):
    x = [x0]
    for _ in range(1, n):
        x.append(phi * x[-1] + random.gauss(0, sigma))
    return x

def moving_average(series, window):
    return [sum(series[i:i+window]) / window for i in range(len(series) - window + 1)]

def differencing(series):
    return [series[i] - series[i - 1] for i in range(1, len(series))]

def fraction_exceeding_threshold(series, threshold):
    return sum(1 for v in series if abs(v) > threshold) / len(series)
```

## Ship It

This lesson produces `code/stochastic.py` with random walks, Markov chains, Gaussian processes, and autocorrelation analysis. These reappear in Phase 3 for time series modeling, Phase 4 for diffusion models, and Phase 5 for training dynamics analysis.

## Exercises

1. **Random walk variance growth.** Simulate 1000 random walks of length 100. Compute the empirical variance at each time step. Verify that `Var(X_t) = t * sigma^2`. How well do the empirical results match the theoretical formula?

2. **Markov chain stationary distribution.** Create a 3-state Markov chain with transition matrix [[0.9, 0.1, 0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]. Verify it has a unique stationary distribution by running the chain for many steps and comparing the empirical state frequencies with the stationary distribution computed by solving `pi * P = pi`.

3. **Autocorrelation of AR(1).** Generate AR(1) processes with phi = 0.1, 0.5, 0.9, and 0.99. Plot the ACF for each. Verify that the ACF decays as `phi^lag`.

4. **Gaussian process prior simulation.** Use the RBF kernel with different length scales (0.1, 1.0, 10.0) and generate sample paths from the GP prior. How does the length scale affect the smoothness and variability of the samples?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Stochastic process | "Random process over time" | A collection of random variables X_t indexed by time. Defines the evolution of a system with randomness. |
| Random walk | "Drunkard's walk" | X_t = X_{t-1} + epsilon. Variance grows with time. Non-stationary. Foundation of Brownian motion and SGD. |
| Markov chain | "Memoryless process" | Future depends only on present, not past. Defined by transition matrix P. Converges to stationary distribution if ergodic. |
| Stationary distribution | "Long-run probabilities" | pi such that pi*P = pi. The equilibrium distribution the chain converges to. |
| Gaussian process | "Distribution over functions" | Any finite set of points has joint Gaussian distribution. Defined by mean and covariance (kernel) functions. |
| Kernel (covariance) function | "Similarity measure" | k(x, x') = Cov(f(x), f(x')). Defines how function values relate at different inputs. |
| Stationarity | "Time-invariant statistics" | Mean, variance, and autocovariance do not change over time. Required for many time series models. |
| Autocorrelation | "Self-correlation at different lags" | Correlation of a series with itself at lag h. Measures temporal persistence. |
| Ergodicity | "Time avg = ensemble avg" | A single long trajectory can estimate the process properties. Without it, you need many independent runs. |
| Diffusion process | "Continuous random process" | dX = mu*dt + sigma*dW. Drift + Brownian noise. Foundation of score-based generative models. |
| AR(1) | "First-order autoregressive" | X_t = phi*X_{t-1} + epsilon. Stationary if |phi| < 1. |
| Detailed balance | "Reversibility condition" | pi_i * P[i][j] = pi_j * P[j][i]. Ensures the chain satisfies time reversibility. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/22-stochastic-processes)
