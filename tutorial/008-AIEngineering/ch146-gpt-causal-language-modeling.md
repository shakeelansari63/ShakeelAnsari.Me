# GPT — Causal Language Modeling

> BERT sees both sides. GPT sees only the past. The triangle mask is the most consequential single line of code in modern AI.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## The Problem

A language model answers one question: given the first `t-1` tokens, what is the probability distribution over token `t`? Train on that signal — next-token prediction — and you get a model that can generate arbitrary text one token at a time.

To train it end-to-end on a whole sequence in parallel, you need each position's prediction to depend only on earlier positions. Otherwise the model trivially cheats by looking at the answer.

The causal mask does this. It is a single upper-triangular matrix of `-inf` values added to attention scores before softmax. After softmax, those positions become 0. Each position can attend only to itself and earlier positions. And because you apply it once to the whole sequence, you get N parallel next-token predictions in one forward pass.

GPT-1 (2018) through GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — they are all decoder-only causal transformers with the same core loop. Just bigger, better data, and better RLHF.

## The Concept

### The mask

Given a sequence of length `N`, build an `N × N` matrix:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Add `M` to the raw attention scores before softmax. `exp(-inf) = 0`, so masked positions contribute zero weight. Each row of the attention matrix is a probability distribution over previous positions only.

### Parallel training, serial inference

Training: forward-pass the whole `(N, d_model)` sequence once, compute N cross-entropy losses (one per position), sum, backprop. Parallel along the sequence.

Inference: you generate token by token. Feed `[t1, t2, t3]`, get `t4`. The KV cache saves the hidden states so you don't recompute them each step. But serial depth at inference = output length. That is the autoregressive tax.

### The loss — shift-by-one

Given tokens `[t1, t2, t3, t4]`:

- Input: `[t1, t2, t3]`
- Targets: `[t2, t3, t4]`

For every position `i`, compute `-log P(target_i | inputs[:i+1])`. Sum. This is the cross-entropy for the whole sequence.

### Decoding strategies

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Argmax every step | Deterministic tasks, code completion |
| Temperature | Divide logits by T, sample | Creative tasks, higher T = more diversity |
| Top-k | Sample from top-k tokens only | Kills low-probability tails |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | 2020+ default |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tails |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction |

## Build It

### Step 1: the causal mask

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Add it to attention scores before softmax. That's the entire mechanism.

### Step 2: sampling strategies

```python
def sample_greedy(probs):
    return max(range(len(probs)), key=lambda i: probs[i])

def sample_temperature(logits, t, rng):
    probs = softmax(logits, temperature=t)
    return sample_from_distribution(probs, rng)

def sample_top_k(logits, k, rng, temperature=1.0):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    keep = indexed[:k]
    keep_ids = [i for i, _ in keep]
    keep_logits = [v for _, v in keep]
    probs = softmax(keep_logits, temperature=temperature)
    return keep_ids[sample_from_distribution(probs, rng)]

def sample_top_p(logits, p, rng, temperature=1.0):
    probs = softmax(logits, temperature=temperature)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cum = 0.0
    cutoff = len(indexed)
    for i, (_, pi) in enumerate(indexed):
        cum += pi
        if cum >= p:
            cutoff = i + 1
            break
    kept = indexed[:cutoff]
    total = sum(pi for _, pi in kept)
    renorm = [(idx, pi / total) for idx, pi in kept]
    r = rng.random()
    cum = 0.0
    for idx, pi in renorm:
        cum += pi
        if r <= cum:
            return idx
    return renorm[-1][0]

def sample_min_p(logits, min_p, rng, temperature=1.0):
    probs = softmax(logits, temperature=temperature)
    max_p = max(probs)
    threshold = min_p * max_p
    kept = [(i, pi) for i, pi in enumerate(probs) if pi >= threshold]
    total = sum(pi for _, pi in kept)
    renorm = [(i, pi / total) for i, pi in kept]
    r = rng.random()
    cum = 0.0
    for i, pi in renorm:
        cum += pi
        if r <= cum:
            return i
    return renorm[-1][0]
```

### Step 3: cross-entropy next-token loss

```python
def cross_entropy_shifted(logits_per_pos, target_ids):
    total = 0.0
    count = 0
    for i in range(len(target_ids) - 1):
        probs = softmax(logits_per_pos[i])
        p = probs[target_ids[i + 1]]
        total += -math.log(max(p, 1e-12))
        count += 1
    return total / count
```

## Use It

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=64, temperature=0.7, top_p=0.9, do_sample=True)
print(tok.decode(out[0]))
```

Under the hood, `generate()` runs the forward pass, pulls the final-position logits, samples the next token, appends it, and repeats.

## Ship It

See `outputs/skill-sampling-tuner.md`. The skill picks sampling parameters for a new generation task.

## Exercises

1. **Easy.** Run the code and verify the causal attention matrix is lower-triangular after softmax.
2. **Medium.** Implement beam search for width 4. Compare perplexity of beam-4 vs greedy on 10 short prompts.
3. **Hard.** Implement speculative decoding: tiny 2-layer model as draft, 6-layer model as verifier. Measure wall-clock speedup.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Causal mask | "The triangle" | Upper-triangular `-inf` matrix so position `i` only sees positions `≤ i` |
| Next-token prediction | "The loss" | Cross-entropy of the model's distribution against the true next token |
| Autoregressive | "Generate one at a time" | Feed output back as input |
| Logits | "Pre-softmax scores" | Raw output of the LM head before softmax |
| Temperature | "Creativity knob" | Divide logits by T; T→0 = greedy, T→∞ = uniform |
| Top-p | "Nucleus sampling" | Truncate distribution to smallest set summing to ≥p |
| Min-p | "Better than top-p" | Keep tokens where `p ≥ min_p × max_p` |
| Speculative decoding | "Draft + verify" | Cheap model proposes N tokens; big model verifies in parallel |

## Further Reading

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling)
