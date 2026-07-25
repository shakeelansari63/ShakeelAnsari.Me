# Complex Numbers

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
