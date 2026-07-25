# Introduction to JAX

> PyTorch mutates tensors. TensorFlow builds graphs. JAX compiles pure functions. That last one changes how you think about deep learning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Learning Objectives

- Write pure-function neural network code using JAX's functional API (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Explain the key design difference between PyTorch's eager mutation and JAX's functional compilation model
- Apply jit compilation and vmap vectorization to accelerate training loops compared to naive Python
- Train a simple network in JAX and contrast the explicit state management with PyTorch's object-oriented approach

## The Problem

PyTorch traces operations eagerly, one at a time. Every `tensor + tensor` is a separate kernel launch. This works until you need to train a 540-billion-parameter model across 2,048 TPUs.

Google DeepMind trains Gemini on JAX. Anthropic trained Claude on JAX. JAX treats your training loop as a compilable program, not a sequence of Python calls.

JAX is NumPy with three superpowers: automatic differentiation, JIT compilation to XLA, and automatic vectorization. You write a function that processes one example. JAX gives you a function that processes a batch, computes gradients, compiles to machine code, and runs across multiple devices.

## The Concept

### The JAX Philosophy

JAX is a functional framework. No classes, no mutable state, no `.backward()` method:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| Mutable `model.parameters()` | Immutable pytree of arrays |

### jax.numpy

JAX reimplements the NumPy API on accelerators. Same function names, same broadcasting rules. But arrays live on GPU/TPU, and every operation is traceable by the compiler.

One critical difference: JAX arrays are immutable. `a = a.at[0].set(5)` instead of `a[0] = 5`.

### jax.grad: Functional Autodiff

PyTorch attaches gradients to tensors. JAX attaches gradients to functions.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)  # 6.0

d2f = jax.grad(jax.grad(f))
d2f(3.0)  # 2.0
```

The constraint: `grad` only works on pure functions. No print statements, no mutation, no randomness without explicit keys.

### jit: Compile to XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss
```

On the first call, JAX traces the function and compiles to XLA. Subsequent calls skip Python entirely.

When JIT helps: training steps (same computation repeated thousands of times), inference.
When JIT hurts: functions with Python control flow, one-shot computations, debugging.

### vmap: Automatic Vectorization

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']

batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`vmap` generates fused vectorized code. Composes with `jit` and `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

### pmap: Data Parallelism

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

Replicates the function across all available devices and splits the batch.

### Pytrees: The Universal Data Structure

JAX operates on "pytrees" -- nested combinations of lists, tuples, dicts, and arrays:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Every JAX transformation knows how to traverse pytrees.

### The JAX Ecosystem

| Library | Role |
|---------|------|
| Flax (Google) | Neural network layers |
| Equinox (Kidger) | Pytree-based neural networks |
| Optax (DeepMind) | Optimizers + LR schedules |
| Orbax (Google) | Checkpointing |

### When to Use JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class | Community-maintained |
| GPU support | Good (XLA) | Best-in-class |
| Debugging | Hard (tracing) | Easy (eager) |
| Ecosystem | Research-focused | Massive |
| Large-scale training | Superior (XLA, pmap) | Good (FSDP) |

Use PyTorch unless you have a specific reason for JAX -- TPU access, per-example gradients, massive multi-device training.

### Random Numbers in JAX

No global random state. Every random operation requires an explicit PRNG key:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

## Build It

### Step 1: Setup and Data

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    return X[:60000], y[:60000], X[60000:], y[60000:]
```

### Step 2: Initialize Parameters

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    params = {
        'layer1': {
            'w': jnp.sqrt(2.0/784) * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': jnp.sqrt(2.0/256) * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': jnp.sqrt(2.0/128) * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

### Step 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

### Step 4: JIT-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

### Step 5: Training Loop

```python
optimizer = optax.adam(learning_rate=1e-3)
X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)
batch_size = 128

for epoch in range(10):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        params, opt_state, loss = train_step(params, opt_state,
            X_shuffled[start:start+batch_size], y_shuffled[start:start+batch_size])
        epoch_loss += loss

    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch+1:2d} | Test Acc: {test_acc:.4f}")
```

Notice what is missing: no `.zero_grad()`, no `.backward()`, no `.step()`. The entire update is one composed function call.

## Use It

### Flax

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x); x = nn.relu(x)
        x = nn.Dense(128)(x); x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

### Optax

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3, warmup_steps=1000, decay_steps=50000
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Compiler that fuses operations and generates optimized GPU/TPU kernels |
| JIT | "Just-in-time compilation" | Traces function on first call, compiles to XLA |
| Pure function | "No side effects" | Output depends only on inputs |
| vmap | "Auto-batching" | Transforms per-example function to batch function |
| pmap | "Auto-parallelism" | Replicates function across multiple devices |
| Pytree | "Nested dict of arrays" | Any nested structure JAX can traverse |
| Tracing | "Recording the computation" | Executes with abstract values to build a computation graph |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions |
| Optax | "JAX's optimizer library" | Composable gradient transformations |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX |

## Exercises

1. Add dropout to the MLP. Thread a PRNG key through the forward pass.
2. Use `jax.vmap` to compute per-example gradients. Find which examples have the largest gradients.
3. Replace the manual forward function with a generic `mlp_forward` for any number of layers.
4. Benchmark training step with and without `@jax.jit`. Time 100 steps of each.
5. Implement gradient clipping using `optax.chain`. Plot gradient norm over training.

## Further Reading

- JAX documentation: https://jax.readthedocs.io/
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018)
- Flax documentation: https://flax.readthedocs.io/
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees" (2021)
- "You Don't Know JAX" (Colin Raffel, 2020)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/12-intro-to-jax)
