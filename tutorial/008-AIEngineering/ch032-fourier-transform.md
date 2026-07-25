# Fourier Transform

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
