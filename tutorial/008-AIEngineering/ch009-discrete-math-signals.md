# Complex Numbers, Fourier, Graphs & Stochastic Processes

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch031): Complex Numbers

> Imaginary numbers are not imaginary. They are a 90-degree rotation.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04  
**Time:** ~120 minutes  

## Learning Objectives

- Represent complex numbers in Cartesian and polar forms and convert between them
- Implement complex arithmetic from scratch using Python tuples
- Visualize complex multiplication as rotation and scaling on the complex plane
- Apply Euler's formula to represent signals and rotations

## The Concept

### What Is a Complex Number?

A complex number has a real part and an imaginary part:

```
z = a + b*i
where i^2 = -1, a = Re(z), b = Im(z)
```

The `i` (or `j` in engineering) is not a mystery. It is the solution to `x^2 + 1 = 0`. There is no real number whose square is -1, so we define a new number `i` that satisfies this.

A complex number is a point in the 2D complex plane: the x-axis is the real part, the y-axis is the imaginary part.

Every complex number has a conjugate:

```
z = a + b*i
z_bar = a - b*i (complex conjugate)
```

Conjugation flips the sign of the imaginary part. It is equivalent to reflecting across the real axis.

### Cartesian Form vs Polar Form

**Cartesian (rectangular) form:** `z = a + b*i`

Cartesian form is natural for addition and subtraction.

**Polar form:** `z = r * (cos(theta) + i*sin(theta)) = r * e^(i*theta)`

- `r = |z| = sqrt(a^2 + b^2)` — the magnitude (modulus), the distance from origin
- `theta = arg(z) = atan2(b, a)` — the angle (argument), the direction from origin

Polar form is natural for multiplication, division, and exponentiation.

**Conversion:**
```
Cartesian to polar:
  r = sqrt(a^2 + b^2)
  theta = atan2(b, a)   (returns angle in correct quadrant)

Polar to Cartesian:
  a = r * cos(theta)
  b = r * sin(theta)
```

### Complex Arithmetic

**Addition:** Add real and imaginary parts separately.

```
(a + b*i) + (c + d*i) = (a + c) + (b + d)*i
```

**Subtraction:** Subtract real and imaginary parts.

```
(a + b*i) - (c + d*i) = (a - c) + (b - d)*i
```

**Multiplication:** FOIL with the rule i^2 = -1.

```
(a + b*i)*(c + d*i) = a*c + a*d*i + b*c*i + b*d*i^2
= (a*c - b*d) + (a*d + b*c)*i
```

In polar form: multiply magnitudes, add angles.

```
r_1*e^(i*theta_1) * r_2*e^(i*theta_2) = r_1*r_2 * e^(i*(theta_1 + theta_2))
```

**Division:** Multiply numerator and denominator by the conjugate.

```
(a + b*i) / (c + d*i) = ((a + b*i)*(c - d*i)) / ((c + d*i)*(c - d*i))
= ((a*c + b*d) + (b*c - a*d)*i) / (c^2 + d^2)
```

In polar form: divide magnitudes, subtract angles.

```
r_1*e^(i*theta_1) / r_2*e^(i*theta_2) = (r_1/r_2) * e^(i*(theta_1 - theta_2))
```

### The Complex Plane

The complex plane is the 2D Cartesian plane where:
- The x-axis is the real axis (Re)
- The y-axis is the imaginary axis (Im)

Each complex number is a point in this plane. This representation makes geometric intuition possible:

**Addition = vector addition.** Place the vectors head-to-tail.

**Multiplication by i = 90-degree rotation.** Multiply any complex number by i and it rotates 90 degrees counterclockwise.

```
z = a + b*i
i*z = -b + a*i
```

In polar form: `i = e^(i*pi/2)`, so multiplying by `i` adds pi/2 to the angle — a 90-degree rotation.

**Multiplication by any complex number = rotation + scaling.**

```
r * e^(i*theta) multiplies by:
- Scale factor: r
- Rotation: theta radians counterclockwise
```

**Complex conjugation = reflection across the real axis.**

```
z = a + b*i
z_bar = a - b*i
```

### Euler's Formula

The most beautiful equation in mathematics:

```
e^(i*theta) = cos(theta) + i*sin(theta)
```

This connects exponentiation (growth) to trigonometry (rotation).

At theta = pi:

```
e^(i*pi) = -1
or equivalently: e^(i*pi) + 1 = 0
```

This is Euler's identity, linking the five most important constants in mathematics: e, i, pi, 1, and 0.

Euler's formula explains why complex numbers are useful for describing oscillations and waves. Any sinusoidal signal can be represented as the real part of a complex exponential:

```
cos(theta) = Re(e^(i*theta))
```

Or directly as a sum of complex exponentials:

```
cos(theta) = (e^(i*theta) + e^(-i*theta)) / 2
sin(theta) = (e^(i*theta) - e^(-i*theta)) / (2*i)
```

### De Moivre's Formula

A direct consequence of Euler's formula:

```
(cos(theta) + i*sin(theta))^n = cos(n*theta) + i*sin(n*theta)
```

This gives a simple formula for computing powers of complex numbers:

```
z^n = r^n * e^(i*n*theta) = r^n * (cos(n*theta) + i*sin(n*theta))
```

And for finding n-th roots:

```
z^(1/n) = r^(1/n) * e^(i*(theta + 2*pi*k)/n) for k = 0, 1, ..., n-1
```

Every non-zero complex number has exactly n distinct n-th roots, equally spaced around a circle of radius r^(1/n).

### The Unit Circle and Roots of Unity

The solutions to `z^n = 1` are the n-th roots of unity:

```
z_k = e^(i*2*pi*k/n) = cos(2*pi*k/n) + i*sin(2*pi*k/n)
for k = 0, 1, ..., n-1
```

These are n points equally spaced on the unit circle, starting at z=1.

Properties:
- Sum of all n roots of unity = 0 (for n > 1)
- Product of all n roots = (-1)^(n+1)
- They form a cyclic group under multiplication
- They are the basis of the Discrete Fourier Transform

### Why Complex Numbers Matter for ML

**Signal processing and audio:** Audio data is inherently oscillatory. Complex numbers (magnitude and phase) compactly represent sinusoidal components.

**Fourier transforms:** The Fourier transform is defined using complex exponentials. Every convolution, filtering, and spectral analysis operation uses complex arithmetic internally.

**Control theory and robotics:** Transfer functions, stability analysis, and PID controllers all use complex numbers (poles and zeros in the complex plane).

**Quantum computing:** Quantum states are complex vectors. Operations are unitary matrices (complex matrices with orthonormal columns). The entire formalism is built on complex linear algebra.

**Representations of rotations:** For 2D rotations, complex numbers are more efficient than 2x2 rotation matrices. Complex multiplication handles rotation and scaling in one operation.

**Network analysis:** Phase information in recurrent neural networks, oscillatory dynamics in reservoir computing, and complex-valued neural networks for specific domains.

### Complex-Valued Neural Networks

Standard NNs use real-valued weights and activations. Complex-valued NNs extend this to complex numbers:

- Weight matrices are complex
- Activation functions must be carefully designed (complex differentiability is restrictive)
- Backpropagation uses Wirtinger calculus (derivatives with respect to complex variables)
- Applications: MRI reconstruction, communications, radar, audio processing

The complex domain preserves phase information that real networks discard. This matters when the data is naturally complex (Fourier features, quadrature signals, quantum states).

## Build It

### Step 1: Complex number representation

```python
import math

def complex_make(a, b):
    return (a, b)

def complex_real(z):
    return z[0]

def complex_imag(z):
    return z[1]
```

### Step 2: Complex arithmetic

```python
def complex_add(z1, z2):
    return (z1[0] + z2[0], z1[1] + z2[1])

def complex_sub(z1, z2):
    return (z1[0] - z2[0], z1[1] - z2[1])

def complex_mul(z1, z2):
    a, b = z1
    c, d = z2
    return (a*c - b*d, a*d + b*c)

def complex_div(z1, z2):
    a, b = z1
    c, d = z2
    denom = c**2 + d**2
    return ((a*c + b*d) / denom, (b*c - a*d) / denom)

def complex_conj(z):
    return (z[0], -z[1])
```

### Step 3: Polar conversion

```python
def complex_abs(z):
    return math.sqrt(z[0]**2 + z[1]**2)

def complex_arg(z):
    return math.atan2(z[1], z[0])

def complex_polar(z):
    return complex_abs(z), complex_arg(z)

def complex_from_polar(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))
```

### Step 4: Complex exponentiation

```python
def complex_pow(z, n):
    r, theta = complex_polar(z)
    return complex_from_polar(r**n, n * theta)

def complex_exp(z):
    a, b = z
    return (math.exp(a) * math.cos(b), math.exp(a) * math.sin(b))
```

## Use It

The all implementations from `code/complex_numbers.py` include complete functions:

```python
import math

def complex_make(a, b):
    return (a, b)

def complex_real(z):
    return z[0]

def complex_imag(z):
    return z[1]

def complex_conj(z):
    return (z[0], -z[1])

def complex_abs(z):
    return math.sqrt(z[0]**2 + z[1]**2)

def complex_arg(z):
    return math.atan2(z[1], z[0])

def complex_add(z1, z2):
    return (z1[0] + z2[0], z1[1] + z2[1])

def complex_sub(z1, z2):
    return (z1[0] - z2[0], z1[1] - z2[1])

def complex_mul(z1, z2):
    a, b = z1
    c, d = z2
    return (a*c - b*d, a*d + b*c)

def complex_div(z1, z2):
    a, b = z1
    c, d = z2
    denom = c**2 + d**2
    return ((a*c + b*d) / denom, (b*c - a*d) / denom)

def complex_polar(z):
    return complex_abs(z), complex_arg(z)

def complex_from_polar(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))

def complex_pow(z, n):
    r, theta = complex_polar(z)
    return complex_from_polar(r**n, n * theta)

def complex_sqrt(z):
    r, theta = complex_polar(z)
    return complex_from_polar(math.sqrt(r), theta / 2)

def complex_exp(z):
    a, b = z
    return (math.exp(a) * math.cos(b), math.exp(a) * math.sin(b))

def complex_log(z):
    r, theta = complex_polar(z)
    return (math.log(r), theta)

def complex_sin(z):
    a, b = z
    return (math.sin(a) * math.cosh(b), math.cos(a) * math.sinh(b))

def complex_cos(z):
    a, b = z
    return (math.cos(a) * math.cosh(b), -math.sin(a) * math.sinh(b))

def euler_formula(theta):
    return (math.cos(theta), math.sin(theta))

def demoivre(z, n):
    r, theta = complex_polar(z)
    return complex_from_polar(r**n, n * theta)

def roots_of_unity(n):
    roots = []
    for k in range(n):
        angle = 2 * math.pi * k / n
        root = (math.cos(angle), math.sin(angle))
        roots.append(root)
    return roots

def complex_matrix_multiply(A, B):
    m = len(A)
    n = len(B[0])
    p = len(B)
    C = [[(0, 0) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = (0, 0)
            for k in range(p):
                s = complex_add(s, complex_mul(A[i][k], B[k][j]))
            C[i][j] = s
    return C
```

## Ship It

This lesson produces `code/complex_numbers.py` with complex arithmetic, polar conversion, Euler's formula, and roots of unity. These appear heavily in Lesson 20 (Fourier Transform) and Phase 4 for signal processing and sequence models.

## Exercises

1. **Visualizing multiplication.** Take z = 0.8 + 0.6*i. Compute z, z^2, z^3, ..., z^10. Plot each point in the complex plane. Describe the pattern relative to the unit circle.

2. **Roots of unity.** Compute and plot the 5th roots of unity. Verify that their sum is zero and that each satisfies z^5 = 1.

3. **Euler's formula verification.** For theta = 0, pi/4, pi/2, pi, compute e^(i*theta) using complex_exp and verify the result equals cos(theta) + i*sin(theta).

4. **Complex matrix multiplication.** Implement a 2x2 complex matrix multiplication using your complex arithmetic functions. Verify that a rotation matrix applied to a vector gives the same result as multiplication by e^(i*theta).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Complex number | "Number with i" | a + b*i where i^2 = -1. A 2D point in the complex plane. |
| Real part | "The a in a + bi" | The x-coordinate of the complex point. |
| Imaginary part | "The b in a + bi" | The y-coordinate of the complex point. Not imaginary at all; it is a real number multiplied by i. |
| Complex conjugate | "Flip the sign" | Change the sign of the imaginary part. Reflection across the real axis. |
| Modulus | "Magnitude" | sqrt(a^2 + b^2). Distance from origin. |
| Argument | "Angle" | atan2(b, a). Direction from origin. |
| Euler's formula | "The circle formula" | e^(i*theta) = cos(theta) + i*sin(theta). Links growth to rotation. |
| Polar form | "Magnitude and angle" | z = r * e^(i*theta). Natural for multiplication. |
| De Moivre's formula | "Power of complex" | (cos(theta) + i*sin(theta))^n = cos(n*theta) + i*sin(n*theta). |
| Roots of unity | "Solutions to z^n = 1" | n equally spaced points on the unit circle. Basis of the DFT. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/19-complex-numbers)

---

## Part 2 (ch032): Fourier Transform

> In the beginning, there was time. The Fourier Transform lets you see the frequencies hidden within.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04, 19  
**Time:** ~120 minutes  

## Learning Objectives

- Implement the Discrete Fourier Transform (DFT) and its inverse from scratch using O(n^2) complexity
- Implement the Cooley-Tukey FFT algorithm with O(n log n) complexity and verify correctness against the naive DFT
- Apply the Fourier transform to filter noise from a signal in the frequency domain
- Interpret the magnitude and phase spectra of real-world signals and images

## The Concept

### What Is the Fourier Transform?

The Fourier Transform decomposes a signal into its constituent frequencies. Any function of time can be represented as a sum of sinusoids at different frequencies, each with its own amplitude and phase.

The intuition: instead of tracking how a signal changes over time, track how much energy exists at each frequency. Time domain tells you "when." Frequency domain tells you "what."

### The Continuous Fourier Transform

For a continuous function `f(t)`:

```
F(omega) = integral from -inf to inf of f(t) * e^(-i*omega*t) dt
```

The inverse:

```
f(t) = (1 / 2*pi) * integral from -inf to inf of F(omega) * e^(i*omega*t) d(omega)
```

The Fourier Transform maps from the time domain to the frequency domain. The inverse maps back.

For ML, the discrete version is what matters.

### The Discrete Fourier Transform (DFT)

For a finite sequence of N samples `x[0], x[1], ..., x[N-1]`:

```
X[k] = sum from n=0 to N-1 of x[n] * e^(-i*2*pi*k*n / N)

for k = 0, 1, ..., N-1
```

where `X[k]` is the k-th frequency component.

The inverse:

```
x[n] = (1/N) * sum from k=0 to N-1 of X[k] * e^(i*2*pi*k*n / N)

for n = 0, 1, ..., N-1
```

The DFT is a linear transformation. It can be viewed as multiplying the input vector by the DFT matrix `F` where `F[j][k] = e^(-i*2*pi*j*k/N)`.

### The DFT Matrix

The N x N DFT matrix:

```
F_N = [[1, 1, 1, ..., 1],
       [1, W_N, W_N^2, ..., W_N^(N-1)],
       [1, W_N^2, W_N^4, ..., W_N^(2*(N-1))],
       ...
       [1, W_N^(N-1), W_N^(2*(N-1)), ..., W_N^((N-1)*(N-1))]]

where W_N = e^(-i*2*pi/N)
```

Properties:
- The matrix is symmetric (but not Hermitian)
- The inverse DFT matrix is the complex conjugate of the forward matrix, divided by N
- The DFT matrix is unitary up to scaling: `F * F^H = N * I`

The naive DFT is O(N^2) — multiplying by the N x N matrix. For N=1000, that is 1 million complex operations. Fine. For N=10^6 (a 1-second audio clip at 44.1 kHz), that is 10^12 operations. Not fine. This is why the FFT exists.

### The Fast Fourier Transform (FFT): Cooley-Tukey

The FFT reduces the DFT from O(N^2) to O(N log N) using divide and conquer. The Cooley-Tukey algorithm (the most common) works when N is a power of 2.

The key insight: split the DFT into even-indexed and odd-indexed terms.

```
X[k] = sum from n=0 to N-1 of x[n] * W_N^(k*n)

= sum from m=0 to N/2-1 of x[2m] * W_N^(k*2m) + sum from m=0 to N/2-1 of x[2m+1] * W_N^(k*(2m+1))

= sum from m=0 to N/2-1 of x[2m] * W_(N/2)^(k*m) + W_N^k * sum from m=0 to N/2-1 of x[2m+1] * W_(N/2)^(k*m)
```

This is the sum of two DFTs of length N/2 (even and odd indexed), combined with the twiddle factor `W_N^k`.

The recursive algorithm:

```
1. If N == 1, return x (the DFT of a single value is itself)
2. Compute FFT of even-indexed elements: E = FFT(x[0::2])
3. Compute FFT of odd-indexed elements: O = FFT(x[1::2])
4. For k = 0, ..., N/2 - 1:
     twiddle = e^(-i*2*pi*k/N)
     X[k]       = E[k] + twiddle * O[k]
     X[k + N/2] = E[k] - twiddle * O[k]
```

The recurrence: `T(N) = 2*T(N/2) + O(N) = O(N log N)`.

For N=1024: naive DFT is about 1 million operations, FFT is about 10,000 operations. The difference grows with N.

**Bit-reversal permutation:** The recursion reorders the input. After repeatedly splitting even/odd, the order is the bit-reversal of the original indices. In-place FFT implementations use this to avoid allocating new arrays.

### Magnitude and Phase Spectra

The DFT output `X[k]` is complex. Two views:

**Magnitude spectrum:** `|X[k]| = sqrt(Re(X[k])^2 + Im(X[k])^2)`

Shows how much energy is at each frequency. The magnitude spectrum is symmetric for real inputs: `|X[k]| = |X[N-k]|`.

**Phase spectrum:** `arg(X[k]) = atan2(Im(X[k]), Re(X[k]))`

Shows the phase shift at each frequency. Phase is usually harder to interpret than magnitude.

For real-valued signals, the magnitude spectrum is symmetric and the phase spectrum is antisymmetric:

```
|X[k]| = |X[N-k]|
arg(X[k]) = -arg(X[N-k])
X[0] is real (DC component, the average value)
```

### Nyquist-Shannon Sampling Theorem

A continuous signal must be sampled at at least twice the maximum frequency present in the signal.

```
f_sampling >= 2 * f_max
```

The Nyquist frequency is `f_Nyquist = f_sampling / 2`. Any frequency above Nyquist is aliased to a lower frequency, creating artifacts. This is why anti-aliasing filters exist.

For a signal sampled at 44.1 kHz (CD quality), the maximum frequency you can represent is 22.05 kHz (just above human hearing). Frequencies above 22.05 kHz must be filtered out before sampling.

### Frequency Bins and Resolution

For N samples at sampling rate `f_s`:

- Each DFT index `k` corresponds to frequency `f_k = k * f_s / N`
- Frequency resolution (spacing between bins): `delta_f = f_s / N`
- Range: DC (k=0) to Nyquist (k=N/2) for real signals

To get finer frequency resolution, increase N (sample for longer). To see higher frequencies, increase the sampling rate.

### The Fourier Transform of Images

The 2D DFT extends naturally:

```
F[u, v] = sum from x=0 to M-1 sum from y=0 to N-1 of f[x, y] * e^(-i*2*pi*(u*x/M + v*y/N))
```

Low frequencies (small u, v) correspond to smooth variations (large-scale structure). High frequencies (large u, v) correspond to edges and texture.

In image processing:
- Low-pass filter: keep small u, v, remove high u, v. Blurs the image.
- High-pass filter: keep large u, v, remove small u, v. Sharpens edges.
- Band-pass filter: keep a range of u, v. Extracts specific texture scales.

The 2D FFT can be computed by applying the 1D FFT to each row, then to each column (separable). This is O(N^2 * log N) for an N x N image.

### Convolution Theorem

The most important property for ML:

```
Convolution in time domain = Multiplication in frequency domain

f * g = F^(-1)(F(f) * F(g))

and conversely:

f * g (element-wise in time) ⇔ F(f) * F(g) (convolution in frequency)
```

Why this matters:
- Direct convolution is O(N^2)
- FFT-based convolution: O(N log N) for FFT + O(N) for multiply + O(N log N) for inverse FFT = O(N log N)
- For large kernels (N > ~64), FFT convolution is faster
- This is why modern convolutional neural networks sometimes use FFT-based convolution for large kernels

The convolution theorem also explains how convolutional neural networks work: a small kernel applied everywhere in the spatial domain is equivalent to multiplying the frequency representation by the kernel's Fourier transform.

### How Fourier Analysis Appears in ML

**Feature engineering:** Spectral features (power in frequency bands, spectral centroids, MFCCs for audio) are standard inputs for classical ML models.

**Data augmentation:** Adding noise in specific frequency bands. Mixing signals in the frequency domain for audio augmentation.

**Audio processing:** Convolution reverb, noise reduction, pitch shifting, and source separation all operate in the frequency domain. Spectrograms (time-frequency representations) are the standard input for audio deep learning.

**Image processing:** Many classical filters (blur, sharpen, edge detection) are defined in the frequency domain. JPEG compression uses the Discrete Cosine Transform (a variant).

**Graph Neural Networks:** Spectral GNNs define convolution via the graph Fourier transform (eigenvectors of the Laplacian matrix). The convolution theorem on graphs: convolution = multiplication in the spectral domain.

**Transformers and attention:** The core operation of attention — weighted sum of values — is related to convolution. Some work has analyzed attention through the lens of the Fourier transform.

**Time series forecasting:** Seasonal decomposition, spectral analysis for periodicity detection, and frequency-domain forecasting methods.

## Build It

### Step 1: Naive DFT

```python
import math

def dft(x):
    N = len(x)
    X = []
    for k in range(N):
        s = (0.0, 0.0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            s = (s[0] + x[n][0] * math.cos(angle) - x[n][1] * math.sin(angle),
                 s[1] + x[n][0] * math.sin(angle) + x[n][1] * math.cos(angle))
        X.append(s)
    return X

def idft(X):
    N = len(X)
    x = []
    for n in range(N):
        s = (0.0, 0.0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            s = (s[0] + X[k][0] * math.cos(angle) - X[k][1] * math.sin(angle),
                 s[1] + X[k][0] * math.sin(angle) + X[k][1] * math.cos(angle))
        x.append((s[0] / N, s[1] / N))
    return x
```

### Step 2: Cooley-Tukey FFT

```python
def fft(x):
    N = len(x)
    if N == 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    X = [(0.0, 0.0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = (math.cos(angle), math.sin(angle))
        odd_twiddle = complex_mul(twiddle, odd[k])
        X[k] = complex_add(even[k], odd_twiddle)
        X[k + N // 2] = complex_sub(even[k], odd_twiddle)
    return X
```

### Step 3: Magnitude and phase spectrum

```python
def magnitude_spectrum(X):
    N = len(X)
    mag = [complex_abs(X[k]) for k in range(N // 2 + 1)]
    return mag

def phase_spectrum(X):
    N = len(X)
    phase = [complex_arg(X[k]) for k in range(N // 2 + 1)]
    return phase
```

## Use It

The all implementations from `code/fourier.py` include complete functions:

```python
import math

def complex_add(z1, z2):
    return (z1[0] + z2[0], z1[1] + z2[1])

def complex_sub(z1, z2):
    return (z1[0] - z2[0], z1[1] - z2[1])

def complex_mul(z1, z2):
    a, b = z1
    c, d = z2
    return (a*c - b*d, a*d + b*c)

def complex_abs(z):
    return math.sqrt(z[0]**2 + z[1]**2)

def complex_arg(z):
    return math.atan2(z[1], z[0])

def dft(x):
    N = len(x)
    X = [(0.0, 0.0)] * N
    for k in range(N):
        s = (0.0, 0.0)
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            w = (math.cos(angle), -math.sin(angle))
            s = complex_add(s, complex_mul(x[n], w))
        X[k] = s
    return X

def idft(X):
    N = len(X)
    x = [(0.0, 0.0)] * N
    for n in range(N):
        s = (0.0, 0.0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = (math.cos(angle), math.sin(angle))
            s = complex_add(s, complex_mul(X[k], w))
        x[n] = (s[0] / N, s[1] / N)
    return x

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [complex_mul((math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N)), odd[k]) for k in range(N // 2)]
    X = [(0.0, 0.0)] * N
    for k in range(N // 2):
        X[k] = complex_add(even[k], T[k])
        X[k + N // 2] = complex_sub(even[k], T[k])
    return X

def ifft(X):
    N = len(X)
    conj_X = [(x[0], -x[1]) for x in X]
    conj_x = fft(conj_X)
    return [(x[0] / N, -x[1] / N) for x in conj_x]

def magnitude_spectrum(X, real_signal=True):
    N = len(X)
    if real_signal:
        n_bins = N // 2 + 1
    else:
        n_bins = N
    return [complex_abs(X[k]) for k in range(n_bins)]

def phase_spectrum(X, real_signal=True):
    N = len(X)
    if real_signal:
        n_bins = N // 2 + 1
    else:
        n_bins = N
    return [complex_arg(X[k]) for k in range(n_bins)]

def lowpass_filter(X, cutoff):
    N = len(X)
    filtered = X[:]
    for k in range(cutoff + 1, N - cutoff):
        filtered[k] = (0.0, 0.0)
    return filtered

def highpass_filter(X, cutoff):
    N = len(X)
    filtered = X[:]
    for k in range(cutoff + 1):
        filtered[k] = (0.0, 0.0)
    for k in range(N - cutoff, N):
        filtered[k] = (0.0, 0.0)
    return filtered

def dft_2d(image):
    H = len(image)
    W = len(image[0])
    rows = [dft(image[y]) for y in range(H)]
    return [dft([rows[y][x] for y in range(H)]) for x in range(W)]

def fft_2d(image):
    H = len(image)
    W = len(image[0])
    rows = [fft(image[y]) for y in range(H)]
    transposed = [[rows[y][x] for y in range(H)] for x in range(W)]
    cols = [fft(col) for col in transposed]
    return [[cols[x][y] for y in range(H)] for x in range(W)]

def fft_convolution(signal, kernel):
    N = len(signal) + len(kernel) - 1
    n = 1
    while n < N:
        n *= 2
    signal_padded = signal + [(0.0, 0.0)] * (n - len(signal))
    kernel_padded = kernel + [(0.0, 0.0)] * (n - len(kernel))
    S = fft(signal_padded)
    K = fft(kernel_padded)
    product = [(S[i][0] * K[i][0] - S[i][1] * K[i][1], S[i][0] * K[i][1] + S[i][1] * K[i][0]) for i in range(n)]
    result = ifft(product)
    return result[:N]

def spectrogram(signal, window_size, hop_size):
    n_windows = (len(signal) - window_size) // hop_size + 1
    spec = []
    for i in range(n_windows):
        start = i * hop_size
        window = signal[start:start + window_size]
        window = [(w * (0.5 - 0.5 * math.cos(2 * math.pi * j / (window_size - 1))), 0.0) for j, w in enumerate(window)]
        X = fft(window)
        mag = magnitude_spectrum(X)
        spec.append(mag)
    return spec
```

## Ship It

This lesson produces `code/fourier.py` with DFT, FFT, inverse FFT, filtering, 2D FFT, and spectrogram functions. These reappear in Phase 4 for audio processing, sequence modeling, and graph neural networks.

## Exercises

1. **DFT vs FFT timing.** Implement a timing comparison for N = 2^k for k = 4, 5, 6, 7, 8, 9, 10. Time the naive DFT and the FFT for each N. Plot the results on a log-log scale. Does the FFT curve follow O(N log N)?

2. **FFT correctness.** Generate a test signal: a sum of two sinusoids (e.g., 50 Hz and 120 Hz, sampled at 1000 Hz). Compute the FFT and verify that the magnitude spectrum has peaks at the correct frequencies. Apply the inverse FFT and verify the signal is reconstructed.

3. **Low-pass filter.** Generate a noisy signal (signal + Gaussian noise). Apply an FFT-based low-pass filter. Compare the filtered signal with the original clean signal.

4. **2D FFT on an image.** Create a 64x64 image of a simple shape (circle, square, gradient). Compute its 2D FFT, zero out the high frequencies (above a cutoff), compute the inverse 2D FFT, and show the result.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| DFT | "Discrete Fourier Transform" | Decomposes a finite sequence into frequencies. O(N^2) naive. |
| FFT | "Fast Fourier Transform" | O(N log N) algorithm for DFT. Cooley-Tukey divides into even/odd indices recursively. |
| Twiddle factor | "FFT rotation" | e^(-i*2*pi*k/N), the complex factor combining even and odd DFT halves. |
| Frequency bin | "DFT index" | Each k corresponds to frequency f_k = k * f_s / N. Resolution = f_s / N. |
| Magnitude spectrum | "Amplitude vs frequency" | |X[k]|: how much energy at each frequency. |
| Phase spectrum | "Phase vs frequency" | arg(X[k]): the phase shift at each frequency. |
| Nyquist frequency | "Half the sampling rate" | Maximum representable frequency. Frequencies above this alias to lower frequencies. |
| Convolution theorem | "Multiply in frequency" | f * g = F^(-1)(F(f) * F(g)). The basis of FFT convolution and spectral methods. |
| Aliasing | "Frequency folding" | When frequencies above Nyquist appear as lower frequencies in the sampled signal. |
| Spectrogram | "Time-frequency view" | Sequence of FFTs over sliding windows. Shows how frequency content changes over time. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/20-fourier-transform)

---

## Part 3 (ch033): Graph Theory

> Your data is not a grid of pixels. Your data is a web of relationships. Graph theory is how you navigate it.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04  
**Time:** ~120 minutes  

## Learning Objectives

- Implement adjacency matrices and adjacency lists for graph representation
- Implement BFS and DFS traversal for connectivity and pathfinding
- Compute PageRank via power iteration on a directed graph
- Explain how graph convolutional networks generalize convolution to graph-structured data

## The Concept

### What Is a Graph?

A graph `G = (V, E)` consists of:
- A set of vertices (nodes) `V`
- A set of edges `E`, where each edge connects two vertices

**Types of graphs:**
- **Undirected:** Edges have no direction. `(u, v)` = `(v, u)`. Friendship networks, collaboration graphs.
- **Directed (digraph):** Edges have a direction. `(u -> v)` is not `(v -> u)`. Web links, Twitter follows, citation graphs.
- **Weighted:** Each edge has a number (weight). Road networks (distance), communication networks (bandwidth).
- **Unweighted:** All edges are equal. Social networks (friendship is binary).

**Special graph structures:**
- **Tree:** Connected, acyclic, undirected graph with no cycles. N nodes, N-1 edges.
- **Bipartite graph:** Vertices can be split into two sets such that all edges go between sets. User-item interactions, matching problems.
- **Complete graph:** Every pair of vertices is connected by an edge. N(N-1)/2 edges for undirected.
- **Sparse graph:** E << V^2. Most real-world graphs are sparse (social networks, web graphs).
- **Dense graph:** E ≈ V^2. Complete or near-complete graphs.

### Graph Representations

**Adjacency Matrix:** An `|V| x |V|` matrix A where `A[i][j] = 1` if there is an edge from i to j, 0 otherwise.

- Undirected: symmetric matrix
- Weighted: store weight instead of 1
- Memory: O(V^2). For V=100,000, that is 10^10 entries. Impractical for sparse graphs.

**Adjacency List:** For each vertex, store a list of its neighbors.

- Memory: O(V + E). For sparse graphs, this is much smaller than adjacency matrix.
- Iterating over neighbors: O(degree). Fast for sparse graphs.
- Checking if edge exists: O(degree). Slower than O(1) of adjacency matrix.

**Edge List:** List of all edges, each as (u, v, weight). Simple, used in graph files.

| Operation | Adjacency Matrix | Adjacency List |
|-----------|-----------------|----------------|
| Memory | O(V^2) | O(V + E) |
| Edge existence check | O(1) | O(degree(v)) |
| Iterate neighbors | O(V) | O(degree(v)) |
| Add edge | O(1) | O(1) |
| Remove edge | O(1) | O(degree(v)) |

For ML on graphs, adjacency lists are the standard for sparse graphs (social networks, citation networks, molecular graphs). Adjacency matrices are used when dense linear algebra operations are needed (spectral methods, graph neural networks).

### Graph Traversal: BFS and DFS

**BFS (Breadth-First Search):** Explore all neighbors at the current depth before going deeper.

```
BFS(start):
  queue = [start]
  visited = {start}
  while queue:
    v = queue.pop(0)
    for neighbor in neighbors(v):
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append(neighbor)
```

Properties:
- Finds shortest path in unweighted graphs
- Order by distance from start
- O(V + E) time
- Applications: shortest path, connected components, web crawling

**DFS (Depth-First Search):** Go as deep as possible before backtracking.

```
DFS(start):
  stack = [start]
  visited = {start}
  while stack:
    v = stack.pop()
    for neighbor in neighbors(v):
      if neighbor not in visited:
        visited.add(neighbor)
        stack.append(neighbor)
```

Properties:
- Does not find shortest path
- Uses stack (or recursion)
- O(V + E) time
- Applications: topological sorting, cycle detection, maze solving, connected components

### Connected Components and Shortest Paths

**Connected components (undirected):** The maximal sets of vertices that are reachable from each other. BFS/DFS from each unvisited vertex finds all components.

**Strongly connected components (directed):** Vertices where there is a path in both directions (v -> u and u -> v). Kosaraju's or Tarjan's algorithm finds them in O(V + E).

**Shortest path in unweighted graphs:** BFS gives the shortest path (minimum edges) in O(V+E).

**Dijkstra's algorithm (weighted, non-negative):** Finds shortest path from one source to all vertices.

```
Dijkstra(start):
  dist = [inf] * V
  dist[start] = 0
  pq = [(0, start)]
  while pq:
    d, v = pq.pop_min()
    if d > dist[v]: continue
    for neighbor, weight in edges[v]:
      new_d = d + weight
      if new_d < dist[neighbor]:
        dist[neighbor] = new_d
        pq.push((new_d, neighbor))
```

O((V + E) * log V) with a binary heap. This is the standard shortest-path algorithm for road networks, routing, and many graph problems.

### PageRank

The algorithm that launched Google. It measures the importance of nodes in a directed graph based on the idea that important nodes are linked to by other important nodes.

The PageRank of node v:

```
PR(v) = (1 - d) / N + d * sum over u in in_neighbors(v) of PR(u) / out_degree(u)
```

where `d` is the damping factor (typically 0.85), representing the probability that a random surfer follows a link rather than jumping to a random page.

**Power iteration:** Start with equal ranks `PR(v) = 1/N`, then iterate until convergence:

```
For each node v:
  new_PR(v) = (1 - d) / N + d * sum over u in in_neighbors(v) of PR(u) / out_degree(u)
```

Properties:
- Converges to a unique stationary distribution
- Guaranteed convergence for d < 1
- Each iteration is O(E) for sparse graphs
- Typically converges in 50-100 iterations for web-scale graphs

PageRank is a special case of eigenvector centrality. The PageRank vector is the dominant eigenvector of the Google matrix `G = (1-d)/N * J + d * A^T * D^(-1)`, where D is the diagonal out-degree matrix.

### Graph Laplacian

The Laplacian matrix `L` of an undirected graph:

```
L = D - A
```

where `D` is the diagonal degree matrix (`D[i][i] = degree(v_i)`) and `A` is the adjacency matrix.

Properties of the Laplacian:
- Symmetric and positive semidefinite
- Has eigenvalues `0 = lambda_1 <= lambda_2 <= ... <= lambda_n`
- The smallest eigenvalue is always 0, with eigenvector (1, 1, ..., 1)
- The number of zero eigenvalues equals the number of connected components
- `lambda_2` (the Fiedler value) measures graph connectivity: larger = better connected

**Normalized Laplacian:**

```
L_sym = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
```

Used in spectral clustering and graph neural networks because its eigenvalues are bounded between 0 and 2.

**Why the Laplacian matters for ML:**

The Laplacian's eigenvectors give a natural Fourier basis for the graph. Just as the Fourier transform decomposes a signal into frequencies (Fourier basis = eigenvectors of the 1D Laplacian), the graph Fourier transform decomposes a graph signal into the eigenvectors of the graph Laplacian.

This is the foundation of:
- Spectral clustering (use bottom k eigenvectors for clustering)
- Graph Fourier transform
- Spectral graph convolutional networks (ChebNet, GCN)
- Label propagation and semi-supervised learning

### Spectral Clustering

Algorithm:

```
1. Compute the Laplacian L = D - A
2. Compute the k smallest eigenvectors of L
3. Form matrix U where columns are these eigenvectors
4. Each row of U is a k-dimensional embedding of a vertex
5. Run k-means on these row embeddings
```

Why it works: the eigenvectors of the Laplacian provide a low-dimensional embedding of the graph that preserves local neighborhood structure. Points that are close in the graph are close in the embedding. K-means on this embedding finds clusters that correspond to graph partitions with minimal cut weight.

Spectral clustering can find non-convex clusters that k-means alone cannot. It is the standard method for graph clustering and image segmentation.

### Graph Neural Networks (GNNs)

GNNs extend neural networks to graph-structured data. The core idea: each node's representation is computed by aggregating information from its neighbors.

**Message passing framework:**

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) for u in N(v)}))
```

where `h_v^(k)` is the feature vector of node v at layer k, and `N(v)` is the set of neighbors of v.

**Graph Convolutional Network (GCN):**

```
H^(k+1) = sigma(D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(k) * W^(k))
```

where `A_tilde = A + I` (adds self-loops), `D_tilde = sum of A_tilde rows`, `H^(k)` is the node feature matrix, `W^(k)` is the weight matrix, and `sigma` is an activation function.

The GCN layer is a spectral filter: it applies a localized first-order approximation of a spectral graph convolution. Each layer aggregates information from immediate neighbors. Stacking k layers gives each node information from its k-hop neighborhood.

**Other GNN variants:**
- **GraphSAGE:** Sample a fixed number of neighbors (scalable to large graphs)
- **GAT (Graph Attention):** Learn attention weights for neighbors (weights depend on node features, not just graph structure)
- **GIN (Graph Isomorphism Network):** Maximally expressive GNN that can distinguish different graph structures

### Applications of Graph Theory in ML

**Social network analysis:** Community detection, influence propagation, recommendation (friends-of-friends).

**Recommendation systems:** User-item bipartite graphs. Collaborative filtering via graph embeddings (Node2Vec, GraphSAGE).

**Molecular property prediction:** Molecules are graphs (atoms = nodes, bonds = edges). GNNs predict properties (solubility, toxicity, drug-target affinity).

**Knowledge graphs:** Entities as nodes, relationships as edges. Link prediction, entity classification, question answering.

**Computer vision:** Scene graphs (objects and their relationships), point cloud processing (3D points as graph), image segmentation (pixels as graph).

**Natural language processing:** Syntactic parse trees, dependency graphs, document citation graphs.

**Physics simulation:** Mesh-based simulations (airflow, structural mechanics). GNNs learn physics simulators.

**Traffic prediction:** Road networks as graphs, traffic sensors as nodes. GNNs predict traffic flow.

## Build It

### Step 1: Graph representation

```python
class Graph:
    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.adj_list = [[] for _ in range(n_vertices)]

    def add_edge(self, u, v, weight=1):
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def adjacency_matrix(self):
        A = [[0] * self.n for _ in range(self.n)]
        for u in range(self.n):
            for v, w in self.adj_list[u]:
                A[u][v] = w
        return A
```

### Step 2: BFS and DFS

```python
def bfs(graph, start):
    visited = [False] * graph.n
    queue = [start]
    visited[start] = True
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for neighbor, _ in graph.adj_list[v]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return order

def dfs(graph, start):
    visited = [False] * graph.n
    stack = [start]
    order = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            order.append(v)
            for neighbor, _ in graph.adj_list[v]:
                if not visited[neighbor]:
                    stack.append(neighbor)
    return order
```

### Step 3: PageRank

```python
def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    n = graph.n
    ranks = [1.0 / n for _ in range(n)]
    out_degrees = [len(graph.adj_list[i]) for i in range(n)]
    for _ in range(max_iter):
        new_ranks = [(1.0 - damping) / n for _ in range(n)]
        for u in range(n):
            for v, _ in graph.adj_list[u]:
                new_ranks[v] += damping * ranks[u] / out_degrees[u]
        diff = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if diff < tol:
            break
    return ranks
```

## Use It

The all implementations from `code/graph_theory.py` include complete functions:

```python
import math

class Graph:
    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.adj_list = [[] for _ in range(n_vertices)]

    def add_edge(self, u, v, weight=1):
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def adjacency_matrix(self):
        A = [[0] * self.n for _ in range(self.n)]
        for u in range(self.n):
            for v, w in self.adj_list[u]:
                A[u][v] = w
        return A

    def degree(self, v):
        return len(self.adj_list[v])

    def neighbors(self, v):
        return [n for n, _ in self.adj_list[v]]

def bfs(graph, start):
    visited = [False] * graph.n
    queue = [start]
    visited[start] = True
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for n, _ in graph.adj_list[v]:
            if not visited[n]:
                visited[n] = True
                queue.append(n)
    return order

def dfs(graph, start):
    visited = [False] * graph.n
    stack = [start]
    order = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            order.append(v)
            for n, _ in graph.adj_list[v]:
                if not visited[n]:
                    stack.append(n)
    return order

def connected_components(graph):
    visited = [False] * graph.n
    components = []
    for v in range(graph.n):
        if not visited[v]:
            component = bfs(graph, v)
            components.append(component)
            for u in component:
                visited[u] = True
    return components

def shortest_path_bfs(graph, start, end):
    visited = [False] * graph.n
    parent = [-1] * graph.n
    queue = [start]
    visited[start] = True
    while queue:
        v = queue.pop(0)
        if v == end:
            return reconstruct_path(parent, start, end)
        for n, _ in graph.adj_list[v]:
            if not visited[n]:
                visited[n] = True
                parent[n] = v
                queue.append(n)
    return []

def reconstruct_path(parent, start, end):
    path = []
    v = end
    while v != -1:
        path.append(v)
        v = parent[v]
    path.reverse()
    return path if path[0] == start else []

def dijkstra(graph, start):
    INF = float('inf')
    dist = [INF] * graph.n
    dist[start] = 0
    visited = [False] * graph.n
    pq = [(0, start)]
    while pq:
        pq.sort(key=lambda x: x[0])
        d, v = pq.pop(0)
        if visited[v]:
            continue
        visited[v] = True
        for n, w in graph.adj_list[v]:
            if not visited[n] and d + w < dist[n]:
                dist[n] = d + w
                pq.append((dist[n], n))
    return dist

def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    n = graph.n
    ranks = [1.0 / n] * n
    out_degrees = [len(graph.adj_list[i]) for i in range(n)]
    for _ in range(max_iter):
        new_ranks = [(1.0 - damping) / n] * n
        for u in range(n):
            if out_degrees[u] == 0:
                continue
            for v, _ in graph.adj_list[u]:
                new_ranks[v] += damping * ranks[u] / out_degrees[u]
        diff = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if diff < tol:
            break
    return ranks

def laplacian_matrix(graph):
    A = graph.adjacency_matrix()
    n = graph.n
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        degree = sum(A[i])
        for j in range(n):
            L[i][j] = -A[i][j]
        L[i][i] = degree
    return L

def degree_matrix(graph):
    n = graph.n
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = graph.degree(i)
    return D

def normalized_laplacian(graph):
    n = graph.n
    L = laplacian_matrix(graph)
    D = degree_matrix(graph)
    D_inv_sqrt = [[0] * n for _ in range(n)]
    for i in range(n):
        if D[i][i] > 0:
            D_inv_sqrt[i][i] = 1.0 / math.sqrt(D[i][i])
    L_norm = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += D_inv_sqrt[i][k] * L[k][j]
            L_norm[i][j] = s
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += L_norm[i][k] * D_inv_sqrt[k][j]
            result[i][j] = s
    return result
```

## Ship It

This lesson produces `code/graph_theory.py` with graph representation, traversals, shortest paths, PageRank, and Laplacian utilities. These reappear in Phase 3 for clustering, Phase 4 for graph neural networks, and Phase 5 for advanced GNN architectures.

## Exercises

1. **PageRank on a small graph.** Create a directed graph with 6 nodes and edges representing a mini web. Compute PageRank manually (power iteration). Which nodes have the highest rank? Verify the ranks sum to 1.

2. **Graph Laplacian properties.** Create a graph with two disconnected cliques of size 4 each. Compute the Laplacian and its eigenvalues. How many zero eigenvalues do you see? Connect the cliques with one edge and repeat. How does the second smallest eigenvalue change?

3. **BFS vs DFS order.** Create a graph that is a binary tree of depth 4. Compare the order of nodes visited by BFS and DFS. What differences do you observe?

4. **Spectral clustering.** Generate a graph of two interleaving half-moons (like sklearn's make_moons). Use spectral clustering with the normalized Laplacian to separate them. Compare with k-means directly on the 2D coordinates.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vertex/Node | "A point in the graph" | The fundamental unit. Represents an entity (person, page, atom, sensor). |
| Edge | "A connection" | A relationship between two vertices. Can be directed, undirected, weighted, unweighted. |
| Adjacency list | "Neighbor list" | For each vertex, a list of neighbors. Memory O(V+E). Standard for sparse graphs. |
| Adjacency matrix | "Connection matrix" | V x V matrix where A[i][j] = 1 if edge exists. Memory O(V^2). Standard for dense graphs. |
| BFS | "Breadth-first search" | Explore in layers of increasing distance. Finds shortest paths in unweighted graphs. |
| DFS | "Depth-first search" | Explore as deep as possible before backtracking. Used for topological sorting, cycle detection. |
| PageRank | "Google's algorithm" | Eigenvector centrality for directed graphs. PR(v) = sum of PR(u)/outdeg(u) from in-neighbors, damped. |
| Graph Laplacian | "L = D - A" | Degree minus adjacency. Symmetric, positive semidefinite. Eigenvalues reveal graph structure. |
| Spectral clustering | "Laplacian eigenvectors" | Cluster by embedding nodes in Laplacian eigenvectors then running k-means. Handles non-convex clusters. |
| GNN | "Graph Neural Network" | Neural network for graph data. Each node aggregates features from its neighbors. Layer k gives k-hop information. |
| Message passing | "Neighbor aggregation" | The core GNN operation: update each node's features by combining its features with aggregated neighbor features. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/21-graph-theory)

---

## Part 4 (ch034): Stochastic Processes

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
