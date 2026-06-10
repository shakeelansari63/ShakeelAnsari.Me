# NumPy Arrays

NumPy provides the `ndarray` — a fast, multi-dimensional array. Install with `pip install numpy`.

```python
import numpy as np
```

## Creating Arrays

```python
# From a list
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(a[1])        # 2

# 2D array (matrix)
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(b)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
print(b[1])        # [4 5 6]
print(b[2][1])     # 8
```

## Shape and Dimension

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(a.ndim)    # 1
print(a.shape)   # (5,)
print(b.ndim)    # 2
print(b.shape)   # (3, 3)
print(b.size)    # 9
```

## Data Type

```python
a = np.array([1.2, 2.3, 3.4])
b = np.array([1, 2, 3])
c = np.array([1, 2, 3], dtype=float)

print(a.dtype)   # float64
print(b.dtype)   # int64
print(c.dtype)   # float64
```

## Special Arrays

```python
# Zeros and Ones
zeros = np.zeros((3, 4))   # 3x4 matrix of 0.0
ones = np.ones((4, 3))     # 4x3 matrix of 1.0

# Fill with a value
filled = np.full((3, 5), 4)  # 3x5 matrix of 4s

# Random numbers
random = np.random.random((3, 6))  # 3x6 matrix of random 0-1

# Range
arange = np.arange(0, 51, 5)       # [0, 5, 10, ..., 50]
linspace = np.linspace(0, 50, 11)  # 11 evenly spaced values from 0 to 50
```

## Array Operations

Unlike lists, operations apply element-wise.

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 0])

print(a + b)   # [7 9 11 13 5]
print(a * 2)   # [2 4 6 8 10]
print(a ** 3)  # [1 8 27 64 125]

# Equations
x = np.arange(0, 51, 5)
y1 = x**2 + 2*x
y2 = np.sin(x)
```

## Math Functions

Trigonometric, exponential, logarithmic and power functions apply element-wise.

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Trigonometry (angles in radians)
print(np.sin(a))
print(np.cos(a))
print(np.tan(a))
print(np.arcsin(a))   # Inverse sin (NaN where |values| > 1)
print(np.arccos(a))   # Inverse cos
print(np.arctan(a))   # Inverse tan

# Exponential
print(np.exp(a))      # e^x
print(np.exp(np.pi * 1j))  # e^(pi*i) = -1 (Euler's identity)

# Square root
print(np.sqrt(a))

# Logarithm
print(np.log(a))      # Natural log (base e)
print(np.log10(a))    # Log base 10
print(np.log2(a))     # Log base 2
```

## Aggregate Functions

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(a.sum())    # 45
print(a.max())    # 9
print(a.min())    # 1
print(a.mean())   # 5.0
print(np.median(a))  # 5.0
print(np.std(a))     # 2.58
```

## Reshape and Transpose

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(a.reshape((2, 6)))     # Reshape to 2x6
print(a.reshape((6, 2)))     # Reshape to 6x2
print(a.reshape((2, 2, 3)))  # Reshape to 3D (2x2x3)
print(a.flatten())           # Flatten to 1D
print(a.transpose())         # Transpose rows ↔ columns

# flat iterator — flatten without creating a copy, useful for iteration
for element in a.flat:
    print(element)
```

## Slicing

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(a[0:2])       # First 2 rows
print(a[:, 1:3])    # All rows, columns 1 and 2
print(a[1, :])      # Row at index 1
```

## Joining Arrays

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = np.array([[51, 52, 53, 54], [55, 56, 57, 58], [59, 60, 61, 62]])

# Concatenate along rows (axis=0) — stack vertically
print(np.concatenate((a, b)))             # Shape: (6, 4)
# Concatenate along columns (axis=1) — stack horizontally
print(np.concatenate((a, b), axis=1))     # Shape: (3, 8)
# stack() adds a new dimension
print(np.stack((a, b)))                   # Shape: (2, 3, 4)
# Horizontal stack (same as axis=1 concatenation)
print(np.hstack((a, b)))
# Vertical stack (same as axis=0 concatenation)
print(np.vstack((a, b)))
```

## Splitting Arrays

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# Split along rows (axis=0)
print(np.split(a, 2))       # Split into 2 equal parts along rows
print(np.split(a, 4))       # Split into 4 rows

# Split along columns (axis=1)
print(np.split(a, 4, axis=1))  # Split into 4 columns
print(np.split(a, 2, axis=1))  # Split into 2 columns

# Horizontal split (same as axis=1)
print(np.hsplit(a, 4))
```

## Adding Rows/Columns

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
b = [21, 22, 23, 24]

print(np.append(a, [b], axis=0))    # Append row
print(np.insert(a, 1, b, axis=0))   # Insert row at index 1
print(np.insert(a, 1, b, axis=1))   # Insert column at index 1
```
