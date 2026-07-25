# Attention Mechanism — The Breakthrough

The decoder stops squinting at a compressed summary and starts looking at the whole source. Everything after this is attention plus engineering.

Lesson 09 ended on a measured failure. A GRU encoder-decoder goes from 89% accuracy at length 5 to near-chance at length 80. Bahdanau, Cho, and Bengio published a three-line fix in 2014: instead of giving the decoder only the final encoder state, keep every encoder state. At each decoder step, compute a weighted average of encoder states where the weights say "how much does the decoder need to look at encoder position `i` right now?"

## The Concept

At each decoder step `t`:

1. Use the previous decoder hidden state `s_{t-1}` as a **query**.
2. Score it against every encoder hidden state `h_1, ..., h_T`.
3. Softmax the scores to get attention weights `α_{t,1}, ..., α_{t,T}`.
4. Context vector `c_t = Σ α_{t,i} * h_i`.
5. Decoder takes `c_t` plus the previous output token, produces the next token.

## Shapes (the thing that bites everyone)

| Thing | Shape | Notes |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | If BiLSTM, `d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | One vector |
| Attention score `e_{t,i}` | scalar | One per encoder position |
| Context vector `c_t` | `(d_h,)` | Same shape as encoder state |

**Bahdanau (additive) score:** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

**Luong (multiplicative) score** has three variants: `dot` (q^T k), `general` (q^T W k), `concat` (Bahdanau-like).

One gotcha: Bahdanau uses `s_{t-1}` (pre-step state). Luong uses `s_t` (post-step state). Pick one paper and stick to its convention.

## Build It

### Step 1: Additive (Bahdanau) Attention

```python
import numpy as np

def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Check shapes: `encoder_states` has shape `(T_enc, d_h)`. `projected_enc` is `(T_enc, d_attn)`. `scores` is `(T_enc,)`. `context` is `(d_h,)`.

### Step 2: Luong Dot and General

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights

def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Three lines each. Same accuracy on most tasks, a lot less code.

### Step 3: A Worked Numerical Example

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

First row wins. Move the decoder state closer to the third encoder state and watch the weights shift.

### Why This Is the Bridge to Transformers

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder states
- **Value** = encoder states

Self-attention separates K and V. Multi-head attention runs it in parallel with different learned projections. The math is the same. The shapes are the same.

## Use It

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

### The Attention-Weight-as-Explanation Trap

Attention weights look interpretable but are not as reliable as they look. Jain and Wallace (2019) showed distributions can be permuted without changing predictions. Never report attention weights as evidence of reasoning without an ablation or counterfactual check.

## Exercises

1. **Easy.** Implement softmax masking so padding tokens get attention weight zero.
2. **Medium.** Add multi-head attention to the Luong `general` form.
3. **Hard.** Train a GRU encoder-decoder with Bahdanau attention on the copy task. Plot accuracy vs length.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Attention | Weighted average of a value sequence, weights from query-key similarity. |
| Query, Key, Value | Three projections: Q asks, K is what to match, V is what to return. |
| Additive attention | Feed-forward score: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Score is `q^T k` or `q^T W k`. Cheaper, same accuracy. |
| Alignment matrix | Attention weights as a grid. What the model attended to. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/10-attention-mechanism)
