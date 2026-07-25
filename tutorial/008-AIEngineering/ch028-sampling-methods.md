# Sampling Methods

> If you cannot compute the integral, sample from it.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04, 07, 15  
**Time:** ~120 minutes  

## Learning Objectives

- Implement inverse transform sampling, rejection sampling, and importance sampling
- Compare sampling efficiency (acceptance rate, effective sample size) across methods
- Compute Monte Carlo estimates and confidence intervals for integrals
- Use sampling to estimate expectations when direct computation is intractable

## The Concept

### Why Sample?

Many quantities in ML are expectations over high-dimensional distributions that cannot be computed analytically:

```
E[f(x)] = integral of f(x) * p(x) dx
```

When p(x) is a neural network's posterior, a latent variable model's prior, or a Bayesian model, this integral is intractable. Sampling provides a way to approximate it.

The core idea: instead of computing the integral analytically, draw samples `x_1, x_2, ..., x_N` from `p(x)` and approximate:

```
E[f(x)] ≈ (1/N) * sum from i=1 to N of f(x_i)
```

This is the Monte Carlo estimate. It is unbiased (the estimate equals the true expectation on average) and its variance decreases as `1/N`. To halve the error, quadruple the number of samples.

### The Law of Large Numbers

The Monte Carlo estimate works because of the Law of Large Numbers: as the number of samples increases, the sample average converges to the true expectation.

`(1/N) * sum(f(x_i)) -> E[f(x)]` as `N -> inf`

This is why you need many samples. With too few, the estimate can be far off. The convergence rate is `O(1/sqrt(N))`, which is slow. Getting one more decimal of accuracy requires 100x more samples.

### The Central Limit Theorem and Monte Carlo Error

The CLT tells us the distribution of the Monte Carlo estimate:

```
sqrt(N) * (estimate - true_value) -> N(0, Var[f(x)])
```

This allows us to compute confidence intervals:

```
True value ≈ estimate +/- 1.96 * sqrt(Var[f(x)] / N)
```

The standard error of the estimate is `sigma_f / sqrt(N)`, where `sigma_f` is the standard deviation of `f(x)`. To compute a valid confidence interval, you need to know (or estimate) `sigma_f`.

### Inverse Transform Sampling

The simplest method: if you can compute the inverse of the CDF, you can sample from any distribution.

```
1. Sample u from Uniform(0, 1)
2. Return x = CDF^(-1)(u)
```

Why it works: if `U` is Uniform(0,1) and `F` is a CDF, then `F^(-1)(U)` has CDF `F`.

This requires:
1. The CDF `F(x)` can be computed (integral of PDF)
2. The inverse CDF `F^(-1)(u)` can be computed analytically or numerically

Examples:
- Exponential: `CDF(x) = 1 - exp(-lambda*x)`, `CDF^(-1)(u) = -log(1-u) / lambda`
- Cauchy: `CDF^(-1)(u) = tan(pi*(u-0.5))`
- Normal: No closed form inverse CDF, but `scipy.stats.norm.ppf` implements it numerically

**Limitation:** For most distributions in ML, the CDF and its inverse are intractable (high-dimensional, complex posterior distributions).

### Rejection Sampling

When you cannot sample directly from `p(x)` (the target distribution), but you can evaluate `p(x)` (up to a normalizing constant), and there is a proposal distribution `q(x)` from which you can sample.

```
1. Sample x from q(x)
2. Sample u from Uniform(0, M * q(x))
3. If u < p(x), accept x. Otherwise, reject and go to step 1.
```

Here `M` is a constant such that `p(x) <= M * q(x)` for all `x`. The envelope `M*q(x)` must dominate the target `p(x)` everywhere.

The acceptance rate is `1/M`. If `M` is large, most samples are rejected and the method is inefficient. The key is finding a good proposal `q` with a tight envelope.

**Why it works:** The algorithm accepts points uniformly under the envelope. The accepted points have distribution proportional to the target density. The rejected points fall in the gap between the target and the envelope.

**The kiss of death:** In high dimensions, the acceptance rate drops exponentially. For a 100-dimensional target, rejection sampling is essentially unusable. This is why MCMC exists.

### Importance Sampling

Rejection sampling gives you exact samples from `p(x)`, but many may be rejected. Importance sampling does not sample from `p(x)` directly. Instead, it corrects samples from `q(x)` by reweighting them.

```
E[f(x)] = integral of f(x) * p(x) dx
= integral of f(x) * (p(x) / q(x)) * q(x) dx
= integral of f(x) * w(x) * q(x) dx
≈ (1/N) * sum from i=1 to N of f(x_i) * w(x_i)
```

where `w(x) = p(x) / q(x)` is the importance weight, and `x_i` are samples from `q(x)`.

The estimate is unbiased if `q(x) > 0` wherever `p(x) > 0` (the support condition). The variance depends on how well `q` matches `p`, weighted by `f`.

**Self-normalized importance sampling:** If `p(x)` is only known up to a normalizing constant (very common in Bayesian ML), use:

```
E[f(x)] ≈ sum(w_i * f(x_i)) / sum(w_i)
```

where `w_i = p_unnormalized(x_i) / q(x_i)`. This introduces some bias but converges to the correct value as N increases.

**Effective Sample Size (ESS):** Measures how many independent samples from `p(x)` the weighted importance sample is equivalent to.

```
ESS = (sum(w_i))^2 / sum(w_i^2)
```

If all weights are equal, `ESS = N`. If one weight dominates, `ESS ≈ 1`. Low ESS means high variance in the estimate.

**The variance problem:** Importance sampling can have infinite variance if `p/q` has heavy tails. If the proposal has thinner tails than the target, a single sample with a huge weight can dominate the estimate. Always verify that the importance weights have finite variance.

### Markov Chain Monte Carlo (MCMC)

When every other method fails (which is most of the time in high dimensions), use MCMC. MCMC constructs a Markov chain whose stationary distribution is the target `p(x)`. After a "burn-in" period, the chain's samples come from approximately `p(x)`.

**Metropolis-Hastings:**

```
1. Start at x_0
2. For t = 1, ..., T:
   a. Propose x' from q(x' | x_t)
   b. Compute acceptance ratio:
      alpha = min(1, p(x') * q(x_t | x') / (p(x_t) * q(x' | x_t)))
   c. With probability alpha, accept: x_{t+1} = x'
      Otherwise, reject: x_{t+1} = x_t
```

The proposal distribution `q` can be anything (a Gaussian centered at current point is a common choice). The chain is guaranteed to converge to `p(x)` as `t -> inf`, regardless of `q`.

**Properties:**
- Samples are correlated (neighbors in the chain are nearby in space). This reduces effective sample size below N.
- There is burn-in: initial samples are biased because the chain has not converged. Discard them.
- The acceptance rate should be ~23% for optimal mixing in high dimensions (theoretical result).
- **Thinning:** keeping only every k-th sample reduces correlation at the cost of discarding information.

**Gibbs sampling:** A special case of Metropolis-Hastings where you sample each variable conditioned on all others. The acceptance ratio is always 1, so every proposal is accepted. This is much more efficient but requires being able to sample from conditional distributions.

### Hamiltonian Monte Carlo (HMC)

Standard Metropolis-Hastings with a Gaussian proposal is essentially a random walk. It explores slowly in high dimensions because the number of steps to reach an independent sample grows with dimension.

HMC uses gradient information to propose distant points with high acceptance probability. It augments the parameter space with momentum variables and simulates Hamiltonian dynamics:

```
1. Sample momentum p ~ N(0, M)
2. Simulate (position, momentum) = (x, p) for L steps using leapfrog integration
3. Accept/reject the final state using Metropolis-Hastings
```

The leapfrog steps:

```
p(t + epsilon/2) = p(t) - epsilon/2 * grad_U(x(t))
x(t + epsilon) = x(t) + epsilon * p(t + epsilon/2) / M
p(t + epsilon) = p(t + epsilon/2) - epsilon/2 * grad_U(x(t + epsilon))
```

where `U(x) = -log p(x)` is the potential energy.

**Why HMC wins:** In high dimensions, the proposal moves along energy contours instead of randomly walking. The effective sample size per gradient evaluation is much higher than random-walk Metropolis. HMC (and its variant, the No-U-Turn Sampler) is the default inference method in probabilistic programming languages like Stan and Pyro.

### Bootstrapping

A non-parametric resampling method: given data `x_1, ..., x_n`, repeatedly sample WITH replacement from the data, compute the statistic of interest on each resample, and use the distribution of these bootstrap statistics to infer uncertainty.

```
1. For b = 1, ..., B:
   a. Draw n samples with replacement from the data (bootstrap sample)
   b. Compute the statistic on this bootstrap sample
2. Use the B bootstrap statistics to estimate:
   - Standard error (standard deviation of bootstrap statistics)
   - Confidence intervals (percentiles of bootstrap statistics)
```

Bootstrap confidence intervals (percentile method):

```
95% CI: [2.5th percentile, 97.5th percentile] of bootstrap statistics
```

The bootstrap requires no parametric assumptions. It works for any statistic (mean, median, correlation, even neural network weights). The number of bootstrap replicates B = 1000-10000 is typical.

**Limitation:** The bootstrap fails for statistics where the bootstrap distribution does not consistently estimate the true sampling distribution (e.g., the maximum of a distribution, extrema).

### Sampling vs Optimization in ML

There is a deep duality between sampling and optimization:

| Sampling | Optimization |
|----------|-------------|
| MCMC | SGD |
| Target: p(x) | Objective: L(x) |
| High probability regions | Low loss regions |
| Temperature = 1 | Temperature -> 0 |
| Gibbs sampling | Coordinate descent |
| Langevin dynamics | Gradient descent with noise |
| Evidence (marginal likelihood) | Maximum likelihood |

The "temperature" parameter connects them: sampling from `p(x) = exp(-L(x)/T)` at T=1 gives Bayesian inference. As T -> 0, sampling concentrates on the mode, becoming optimization.

This duality is used in simulated annealing (start hot for exploration, cool down for convergence) and in Bayesian deep learning (SGD with noise approximates Bayesian inference).

## Build It

### Step 1: Inverse transform sampling

```python
import math
import random

def sample_exponential(lambda_param, n=1):
    samples = []
    for _ in range(n):
        u = random.random()
        x = -math.log(1 - u) / lambda_param
        samples.append(x)
    return samples
```

### Step 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf,
                     M, n_samples=1000):
    samples = []
    while len(samples) < n_samples:
        x = proposal_sample()
        u = random.random() * M * proposal_pdf(x)
        if u < target_pdf(x):
            samples.append(x)
    return samples
```

### Step 3: Importance sampling

```python
def importance_sampling(f, target_pdf, proposal_sample, proposal_pdf, n=10000):
    samples = [proposal_sample() for _ in range(n)]
    weights = [target_pdf(x) / proposal_pdf(x) for x in samples]
    estimate = sum(w * f(x) for x, w in zip(samples, weights)) / sum(weights)
    return estimate, samples, weights
```

## Use It

The all implementations from `code/sampling.py` include complete functions:

```python
import math
import random

def sample_uniform(n=1):
    return [random.random() for _ in range(n)]

def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_binomial(n_trials, p, n_samples=1):
    return [sum(sample_bernoulli(p, n_trials)) for _ in range(n_samples)]

def sample_exponential(lambda_param, n=1):
    return [-math.log(1 - random.random()) / lambda_param for _ in range(n)]

def sample_normal_box_muller(n=1):
    samples = []
    while len(samples) < n:
        u1 = random.random()
        u2 = random.random()
        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        samples.append(z1)
        if len(samples) < n:
            samples.append(z2)
    return samples[:n]

def inverse_cdf_exponential(u, lambda_param):
    return -math.log(1 - u) / lambda_param

def rejection_sample(target_pdf, proposal_sample, proposal_pdf,
                     M, n_samples=1000):
    accepted = []
    total_attempts = 0
    while len(accepted) < n_samples:
        total_attempts += 1
        x = proposal_sample()
        u = random.random() * M * proposal_pdf(x)
        if u < target_pdf(x):
            accepted.append(x)
    acceptance_rate = n_samples / total_attempts
    return accepted, acceptance_rate

def importance_sampling(f, target_pdf, proposal_sample, proposal_pdf, n=10000):
    samples = [proposal_sample() for _ in range(n)]
    log_weights = [math.log(target_pdf(x)) - math.log(proposal_pdf(x)) for x in samples]
    max_weight = max(log_weights)
    weights = [math.exp(w - max_weight) for w in log_weights]
    total_weight = sum(weights)
    estimate = sum(w * f(x) for x, w in zip(samples, weights)) / total_weight
    return estimate, samples, weights, total_weight

def effective_sample_size(weights):
    w_sum = sum(weights)
    return w_sum ** 2 / sum(w ** 2 for w in weights)

def bootstrap(data, statistic, n_bootstrap=1000):
    n = len(data)
    estimates = []
    for _ in range(n_bootstrap):
        resample = [random.choice(data) for _ in range(n)]
        estimates.append(statistic(resample))
    return estimates

def bootstrap_ci(data, statistic, alpha=0.05, n_bootstrap=1000):
    estimates = bootstrap(data, statistic, n_bootstrap)
    estimates.sort()
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    return estimates[lower_idx], estimates[upper_idx]

def monte_carlo_integral(f, lower, upper, n=10000):
    width = upper - lower
    samples = [lower + random.random() * width for _ in range(n)]
    estimate = width * sum(f(x) for x in samples) / n
    variance = width ** 2 * (sum(f(x) ** 2 for x in samples) / n -
                             (sum(f(x) for x in samples) / n) ** 2) / n
    return estimate, math.sqrt(variance)

def metropolis_hastings(log_target, proposal_sample, x0, n_iter=1000):
    chain = [x0]
    n_accepted = 0
    for _ in range(n_iter - 1):
        current = chain[-1]
        proposal = proposal_sample(current)
        log_alpha = log_target(proposal) - log_target(current)
        if log_alpha >= 0 or random.random() < math.exp(log_alpha):
            chain.append(proposal)
            n_accepted += 1
        else:
            chain.append(current)
    return chain, n_accepted / n_iter

def metropolis(log_target, step_size, x0, n_iter=1000):
    def proposal(x):
        return x + random.gauss(0, step_size)
    return metropolis_hastings(log_target, proposal, x0, n_iter)
```

## Ship It

This lesson produces `code/sampling.py` with all sampling methods, Monte Carlo estimation, and MCMC. These reappear in Phase 3 for Bayesian linear regression, Phase 4 for variational autoencoders, and Phase 5 for Bayesian deep learning.

## Exercises

1. **Compare sampling methods.** Estimate E[x^2] for a standard normal distribution using (a) inverse transform, (b) Box-Muller, (c) rejection sampling from a Cauchy proposal, (d) importance sampling from a Cauchy proposal. Compare accuracy vs number of target-distribution samples.

2. **Rejection sampling efficiency.** Implement rejection sampling for a Beta(2,5) target using a Uniform(0,1) proposal. Compute the acceptance rate for different M values. What is the theoretical minimum M that makes the envelope dominate everywhere?

3. **MCMC vs importance sampling.** For a bimodal target distribution (mixture of two Gaussians), estimate the mean using importance sampling (Gaussian proposal centered between modes) and Metropolis-Hastings. Which gives a better estimate with 1000 samples?

4. **Bootstrap confidence intervals.** Generate 100 data points from a Gamma(shape=3, rate=1) distribution. Compute the 95% bootstrap confidence interval for the mean. Compare with the theoretical confidence interval using the CLT.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Monte Carlo | "Random sampling" | Approximating integrals by averaging over random samples. Convergence rate O(1/sqrt(N)). |
| Inverse transform | "CDF inversion" | Sample from a distribution by inverting its CDF. Simple but requires known inverse CDF. |
| Rejection sampling | "Accept/reject" | Sample from a proposal, accept with probability proportional to target/proposal ratio. Exact but can be very inefficient in high dimensions. |
| Importance sampling | "Reweighted samples" | Compute expectations under p using samples from q, reweighted by p/q ratio. Unbiased, but can have high variance if p/q is mismatched. |
| MCMC | "Markov chain Monte Carlo" | Build a Markov chain with target distribution as stationary distribution. Most general sampling method, works in high dimensions. |
| Metropolis-Hastings | "Accept/reject MCMC" | Propose new state, accept with probability min(1, ratio). Guaranteed to converge to target. |
| Burn-in | "Warmup period" | Initial samples of MCMC chain before it converges to the target distribution. Always discard these. |
| Effective sample size | "How many independent samples" | Equivalent number of independent samples from the target distribution. Always less than N for MCMC and importance sampling. |
| Hamiltonian Monte Carlo | "Gradient-guided MCMC" | MCMC that uses gradient of log-target to propose distant states with high acceptance. Much more efficient in high dimensions than random-walk MCMC. |
| Bootstrap | "Resample the data" | Non-parametric uncertainty estimation by sampling with replacement from observed data. Works for any statistic. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/16-sampling-methods)
