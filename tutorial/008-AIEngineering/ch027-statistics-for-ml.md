# Statistics for Machine Learning

> Probability is the logic of uncertainty. Statistics is the science of learning from data.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04  
**Time:** ~120 minutes  

## Learning Objectives

- Compute mean, variance, standard deviation, covariance, and correlation from scratch
- Implement Bayes' theorem and interpret it as updating beliefs with evidence
- Apply maximum likelihood estimation (MLE) to estimate distribution parameters
- Describe the bias-variance tradeoff in intuitive and formal terms

## The Concept

### Random Variables: The Basic Setup

A random variable is a function that maps outcomes of a random process to numbers. `X: Omega -> R`.

- **Discrete random variable:** Takes values from a countable set (e.g., number of heads in 10 coin flips: 0, 1, 2, ..., 10)
- **Continuous random variable:** Takes values from an interval or continuum (e.g., height of a randomly selected person)

A random variable is described by its **probability distribution**, which tells you what values it can take and how likely each value is.

### Probability Distributions

**Probability Mass Function (PMF)** for discrete variables:

```
P(X = x) = probability that X equals x
```

Properties: `0 <= P(X = x) <= 1` for all x, and `sum over all x of P(X = x) = 1`.

**Probability Density Function (PDF)** for continuous variables:

```
f(x) = density at point x
P(a <= X <= b) = integral from a to b of f(x) dx
```

Properties: `f(x) >= 0`, and the total area under the curve is 1.

The PDF does not give probabilities directly (a continuous distribution assigns zero probability to any specific point). Only integrals over intervals give probabilities.

### Summary Statistics

**Mean (Expected Value):** The center of mass of the distribution.

```
Discrete: E[X] = sum(x * P(X = x))
Continuous: E[X] = integral(x * f(x) dx)
```

The mean minimizes the sum of squared deviations. It is the value that minimizes `E[(X - c)^2]` over all c.

**Variance:** Average squared deviation from the mean. Measures spread.

```
Var(X) = E[(X - mu)^2] = E[X^2] - E[X]^2
```

There are two formulas. The first is the definition. The second is computationally convenient but numerically unstable (see Lesson 13).

**Standard Deviation:** `sigma = sqrt(Var(X))`. Variance in the original units (dollars squared -> dollars).

**Covariance:** Measures how two variables vary together.

```
Cov(X, Y) = E[(X - mu_X)(Y - mu_Y)] = E[XY] - E[X]E[Y]
```

- Covariance positive: when X is above its mean, Y tends to be above its mean
- Covariance negative: when X is above its mean, Y tends to be below its mean
- Covariance zero: no linear relationship (but could have nonlinear relationship)

**Correlation (Pearson's r):** Normalized covariance, always between -1 and 1.

```
Corr(X, Y) = Cov(X, Y) / (sigma_X * sigma_Y)
```

- `r = 1`: perfect positive linear relationship
- `r = -1`: perfect negative linear relationship
- `r = 0`: no linear relationship

Correlation is scale-invariant. Multiply X by 1000 and correlation does not change. Covariance would change by a factor of 1000.

### The Normal (Gaussian) Distribution

The most important distribution in statistics and ML.

```
PDF: f(x | mu, sigma^2) = (1 / sqrt(2 * pi * sigma^2)) * exp(-(x - mu)^2 / (2 * sigma^2))
```

Parameters:
- `mu` (mean): center of the bell curve
- `sigma^2` (variance): width of the bell curve

The normal distribution emerges from the Central Limit Theorem: the sum of many independent random variables (with finite variance) is approximately normal, regardless of their individual distributions.

68-95-99.7 rule:
| Interval | Contains |
|-----------|----------|
| mu +/- 1 sigma | ~68% of data |
| mu +/- 2 sigma | ~95% of data |
| mu +/- 3 sigma | ~99.7% of data |

### Bayes' Theorem

The foundation of Bayesian reasoning and Bayesian ML:

```
P(A | B) = P(B | A) * P(A) / P(B)
```

- `P(A)`: prior probability (what you believe before seeing evidence)
- `P(B | A)`: likelihood (probability of seeing evidence if A is true)
- `P(A | B)`: posterior probability (updated belief after seeing evidence)
- `P(B)`: evidence (normalizing constant, total probability of B)

Bayes' theorem tells you how to update beliefs in light of new data. It is the mathematical formalization of learning.

Example: Medical test.

```
Disease prevalence: 1% (prior)
Test sensitivity: 99% (true positive rate)
Test false positive: 5%
If you test positive, what is the probability you have the disease?
```

```
P(D | T+) = P(T+ | D) * P(D) / P(T+)
= 0.99 * 0.01 / (0.99 * 0.01 + 0.05 * 0.99)
= 0.0099 / 0.0594
= 0.1667
```

A 99% accurate test gives only 16.7% posterior probability because the disease is rare. This is why Bayes' theorem is essential: it corrects for base rates.

### Maximum Likelihood Estimation (MLE)

MLE finds the parameter values that make the observed data most probable.

Given data `x_1, x_2, ..., x_n` assumed to come from a distribution with parameters `theta`, the likelihood is:

```
L(theta) = product from i=1 to n of f(x_i | theta)
```

MLE finds `theta` that maximizes `L(theta)`. Equivalently (and more practically), maximizes the log-likelihood:

```
LL(theta) = sum from i=1 to n of log(f(x_i | theta))
```

The log transforms the product into a sum, making optimization easier and more numerically stable.

**MLE for Gaussian mean:**

```
mu_hat = (1/n) * sum(x_i)  (the sample mean)
```

**MLE for Gaussian variance (biased):**

```
sigma_hat^2 = (1/n) * sum((x_i - mu_hat)^2)
```

This is the biased variance estimator (divides by n, not n-1). Unbiased uses n-1. For large n, the difference is negligible.

**MLE for Bernoulli:**

```
p_hat = (number of successes) / (total trials)
```

**MLE for Poisson:**

```
lambda_hat = (1/n) * sum(x_i)  (the sample mean)
```

### Sampling Distributions

A statistic (like the sample mean) is itself a random variable. Its distribution is called the sampling distribution.

**Standard error:** Standard deviation of the sampling distribution of a statistic.

```
SE(x_bar) = sigma / sqrt(n)
```

The standard error tells you how precisely you have estimated the mean. Double the sample size -> standard error drops by sqrt(2).

**Confidence interval:** Range within which the true parameter is likely to fall.

95% CI for the mean: `x_bar +/- 1.96 * SE(x_bar)`

Interpretation: if you repeated the experiment many times, 95% of the confidence intervals would contain the true mean.

### Hypothesis Testing

The null hypothesis significance testing framework:

1. State the null hypothesis H0 (typically "no effect" or "no difference")
2. Compute a test statistic from the data
3. Calculate the p-value: probability of observing data as extreme as what you saw, assuming H0 is true
4. If p-value < threshold (typically 0.05), reject H0

**p-value misinterpretation warning:** The p-value is NOT the probability that H0 is true. It is the probability of the data given H0, not the probability of H0 given the data. Bayesian statistics fixes this, but frequentist methods dominate in practice.

### Bias-Variance Tradeoff

The fundamental tension in supervised learning.

For an estimator `theta_hat` of a true parameter `theta`:

```
MSE(theta_hat) = E[(theta_hat - theta)^2]
= Var(theta_hat) + (Bias(theta_hat))^2
```

- **Bias:** `E[theta_hat] - theta`. How far off the estimate is on average.
- **Variance:** `Var(theta_hat)`. How much the estimate varies across different data samples.
- **MSE:** Mean squared error, the sum of bias squared and variance.

In machine learning:
- High bias = underfitting (model is too simple)
- High variance = overfitting (model is too complex)
- The optimal model balances bias and variance to minimize test error

Simple models (linear regression) have high bias but low variance. Complex models (deep neural networks with sufficient data) have low bias but high variance. Regularization increases bias to reduce variance, lowering overall test error.

## Build It

### Step 1: Summary statistics from scratch

```python
import math

def mean(values):
    return sum(values) / len(values)

def variance(values, ddof=0):
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - ddof)

def std(values, ddof=0):
    return math.sqrt(variance(values, ddof))

def covariance(x, y, ddof=0):
    mx, my = mean(x), mean(y)
    N = len(x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (N - ddof)

def correlation(x, y):
    return covariance(x, y) / (std(x) * std(y))
```

### Step 2: Gaussian PDF

```python
def gaussian_pdf(x, mu=0, sigma=1):
    return (1 / math.sqrt(2 * math.pi * sigma ** 2)) * \
           math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
```

### Step 3: MLE for Gaussian

```python
def mle_gaussian(data):
    return mean(data), variance(data, ddof=0)

def log_likelihood_gaussian(data, mu, sigma):
    n = len(data)
    return (-n / 2) * math.log(2 * math.pi * sigma ** 2) - \
           (1 / (2 * sigma ** 2)) * sum((x - mu) ** 2 for x in data)
```

## Use It

The all implementations from `code/statistics.py` include complete functions:

```python
import math
import random

def mean(values):
    return sum(values) / len(values)

def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]

def mode(values):
    return max(set(values), key=values.count)

def variance(values, ddof=0):
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - ddof)

def std(values, ddof=0):
    return math.sqrt(variance(values, ddof))

def covariance(x, y, ddof=0):
    mx, my = mean(x), mean(y)
    N = len(x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (N - ddof)

def correlation(x, y):
    return covariance(x, y) / (std(x) * std(y))

def zscore(value, mu, sigma):
    return (value - mu) / sigma

def normalize(values):
    mu = mean(values)
    s = std(values)
    return [(v - mu) / s for v in values]

def gaussian_pdf(x, mu=0, sigma=1):
    return (1 / math.sqrt(2 * math.pi * sigma ** 2)) * \
           math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def gaussian_log_pdf(x, mu=0, sigma=1):
    return -0.5 * math.log(2 * math.pi * sigma ** 2) - \
           (x - mu) ** 2 / (2 * sigma ** 2)

def mle_gaussian(data):
    return mean(data), variance(data, ddof=0)

def log_likelihood_gaussian(data, mu, sigma):
    n = len(data)
    return (-n / 2) * math.log(2 * math.pi * sigma ** 2) - \
           (1 / (2 * sigma ** 2)) * sum((x - mu) ** 2 for x in data)

def mle_bernoulli(trials):
    successes = sum(trials)
    total = len(trials)
    return successes / total

def likelihood_bernoulli(trials, p):
    successes = sum(trials)
    failures = len(trials) - successes
    return (p ** successes) * ((1 - p) ** failures)

def log_likelihood_bernoulli(trials, p):
    successes = sum(trials)
    failures = len(trials) - successes
    return successes * math.log(p) + failures * math.log(1 - p)

def bayes_posterior(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

def standard_error(sigma, n):
    return sigma / math.sqrt(n)

def confidence_interval_mean(mu, se, z=1.96):
    return (mu - z * se, mu + z * se)

def empirical_rule(mu, sigma):
    return {
        '1sigma': (mu - sigma, mu + sigma),
        '2sigma': (mu - 2 * sigma, mu + 2 * sigma),
        '3sigma': (mu - 3 * sigma, mu + 3 * sigma),
    }

def bias(estimates, true_value):
    return mean(estimates) - true_value

def variance_of_estimator(estimates):
    return variance(estimates, ddof=0)

def mse(estimates, true_value):
    return sum((e - true_value) ** 2 for e in estimates) / len(estimates)
```

## Ship It

This lesson produces `code/statistics.py` with all summary statistics, distribution functions, MLE estimators, and Bayesian update utilities. These are used in Phase 3 for evaluating model performance, Phase 4 for attention variance analysis, and throughout the curriculum for data analysis.

## Exercises

1. **MLE comparison.** Generate 1000 samples from a Gaussian with mu=5, sigma=2. Estimate mu and sigma using MLE. Repeat 1000 times to get the sampling distribution of the MLE estimates. Does the MLE for sigma systematically underestimate the true sigma (bias)?

2. **Bayesian updating.** You have a coin with unknown bias p. Start with a uniform prior (Beta(1,1)). Flip the coin 10 times and get 7 heads. Use Bayes' theorem to compute the posterior distribution of p. How does the posterior change if you use a Beta(2,5) prior instead?

3. **Confidence interval coverage.** Generate 1000 datasets of size n=30 from N(0,1). For each dataset, compute the 95% confidence interval for the mean. What fraction of intervals contains the true mean (0)? How does this change if the data comes from a uniform instead of a normal distribution?

4. **Bias-variance decomposition.** For an estimator of the variance: compute the true variance from data generated with known sigma. Then compute the average bias of the MLE variance estimator (divides by n) and the unbiased estimator (divides by n-1) over 10000 trials. Which has lower MSE?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Expected value | "Average" | Weighted average of all possible values, weighted by their probabilities. The center of mass of the distribution. |
| Variance | "Spread" | Average squared deviation from the mean. Measures how spread out the distribution is. |
| Standard deviation | "Spread in original units" | Square root of variance. Same units as the original data. |
| Covariance | "How two variables co-vary" | Average product of deviations from means. Positive: move together. Negative: move oppositely. |
| Correlation | "Normalized covariance" | Covariance divided by product of standard deviations. Always between -1 and 1. |
| Normal distribution | "Bell curve" | The limiting distribution of sums of independent random variables. Defined by mean and variance. |
| MLE | "Maximum likelihood" | Find the parameter values that make the observed data most probable. |
| Bayes' theorem | "Update beliefs with data" | P(A|B) = P(B|A)*P(A)/P(B). The mathematical foundation of learning from evidence. |
| Posterior | "Updated belief" | Probability distribution over parameters after seeing data. |
| Prior | "Initial belief" | Probability distribution over parameters before seeing data. |
| Likelihood | "Probability of data" | Probability of observing the data given specific parameter values. |
| p-value | "Significance" | Probability of seeing data as extreme as observed, assuming null is true. NOT the probability the null is true. |
| Bias | "Systematic error" | Difference between average estimate and true value. High bias = underfitting. |
| Variance (of estimator) | "Estimate instability" | How much the estimate varies across different datasets. High variance = overfitting. |
| Bias-variance tradeoff | "Underfitting vs overfitting" | The decomposition of MSE into bias^2 + variance. Optimal model balances both. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/15-statistics-for-ml)
