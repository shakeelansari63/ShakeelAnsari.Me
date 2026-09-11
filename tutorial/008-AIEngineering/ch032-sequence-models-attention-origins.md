# RNNs, Seq2Seq & the Attention Breakthrough

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch101): CNNs and RNNs for Text

Convolutions learn n-grams. Recurrences remember. Both are superseded by attention. Both still matter on constrained hardware.

TF-IDF and Word2Vec produce flat vectors that ignore word order. Two families of architectures filled that gap before transformers arrived.

## The Concept

**TextCNN** (Kim, 2014). Apply 1D convolutions over sequences of word embeddings. A filter of width 3 is a learnable trigram detector. Max-pool to a fixed-size representation. Flat, parallel, fast.

**RNN.** Process tokens one at a time, maintaining a hidden state. Sequential, memory-bearing. Dominated sequence modeling from 2014 to 2017.

**LSTM** adds gates (input, forget, output) and a cell state that stabilizes gradients through long sequences. **GRU** simplifies LSTM to two gates with similar accuracy and fewer parameters.

**Bidirectional RNNs** run one RNN forward and another backward, concatenating hidden states. Essential for tagging tasks.

## Build It

### Step 1: TextCNN in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

The `transpose(1, 2)` reshapes `[batch, seq_len, embed_dim]` to `[batch, embed_dim, seq_len]` for `nn.Conv1d`.

### Step 2: LSTM Classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool over the sequence usually beats taking the last hidden state for classification.

### Step 3: The Vanishing Gradient Demo

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)

# At weight=0.9 over 100 steps: 0.9^100 ≈ 2.7e-5
```

LSTMs fix this with a **cell state** that runs through the network with only additive interactions.

### Why This Still Was Not Enough

1. Sequential bottleneck: RNNs on 1000-length sequences require 1000 serial steps.
2. Fixed-size context vector in encoder-decoder setups compresses all input into one vector.
3. Distant-dependency accuracy ceiling.

Attention solved all three. Transformers dropped recurrence entirely.

## Use It

When it fits the constraint:

- **Edge/on-device inference.** TextCNN + GloVe is 10-100x smaller than a transformer.
- **Streaming/online classification.** RNN processes one token at a time.
- **Tiny models for baselines.** Train a TextCNN in 5 minutes on a CPU.
- **Sequence labeling with limited data.** BiLSTM-CRF is production-grade NER for 1k-10k sentences.

Everything else goes to a transformer.

## Exercises

1. **Easy.** Train a TextCNN on a 3-class toy dataset. Verify multi-filter-width outperforms single width.
2. **Medium.** Compare max-pool, mean-pool, and last-state pooling for LSTM classifier.
3. **Hard.** Build a BiLSTM-CRF NER tagger. Train on CoNLL-2003.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| TextCNN | Stack of 1D convolutions over word embeddings with global max-pool. |
| RNN | Hidden state updated at each timestep: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN with input/forget/output gates and a cell state. |
| GRU | Two gates instead of three. Similar accuracy, fewer params. |
| Bidirectional | Forward + backward concatenated. Both left and right context. |
| Vanishing gradient | Repeated <1 weight multiplication makes early gradients effectively zero. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text)

---

## Part 2 (ch102): Sequence-to-Sequence Models

Two RNNs pretending to be a translator. The bottleneck they hit is the reason attention exists.

Classification maps a variable-length sequence to a single label. Translation maps a variable-length sequence to another of different length, with no guarantee of length parity. The seq2seq architecture (Sutskever, Vinyals, Le, 2014) cracked this with two RNNs: one reads the source, the other generates the target token by token.

## The Concept

**Encoder.** An RNN that reads the source sentence. Its final hidden state is the **context vector** — a fixed-size summary of the entire input.

**Decoder.** Another RNN initialized from the context vector. At each step it takes the previously generated token and produces a distribution over the target vocabulary. Repeat until `<EOS>`.

**Teacher forcing.** During training, the decoder's input at step `t` is the ground-truth token, not its own previous prediction. This stabilizes training. The gap between training and inference is called **exposure bias**.

**The bottleneck.** Everything the encoder learned must be squeezed into one context vector. Long sentences lose detail. Attention (lesson 10) fixes this by letting the decoder look at every encoder hidden state.

## Build It

### Step 1: An Encoder

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

### Step 2: A Decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

### Step 3: Training Loop with Teacher Forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

### Step 4: Inference Loop (Greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

**Beam search** keeps the top-k partial sequences alive and picks the highest-scoring complete one at the end. Beam width 3-5 is standard.

### Step 5: The Bottleneck, Demonstrated

Train on a copy task. Results show the bottleneck:

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

A single GRU hidden state cannot losslessly memorize a 40-token input.

## Use It

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

### Exposure Bias Mitigations

Scheduled sampling (anneal teacher forcing), minimum risk training (train on BLEU), reinforcement learning fine-tuning (RLHF for LLMs).

## Exercises

1. **Easy.** Implement the toy copy task. Measure accuracy at lengths 5, 10, 20.
2. **Medium.** Add beam search with width 3. Measure BLEU against greedy.
3. **Hard.** Fine-tune `facebook/bart-base` on a 10k-pair paraphrase dataset.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Encoder | Reads source. Produces per-step hidden states and final context vector. |
| Decoder | Generates target tokens one at a time from context vector. |
| Context vector | Final encoder hidden state. Fixed-size bottleneck that attention solves. |
| Teacher forcing | Feed ground-truth previous token at training time. |
| Exposure bias | Model trained on true tokens never practiced recovering from own mistakes. |
| Beam search | Keep top-k partial sequences alive instead of greedy commitment. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence)

---

## Part 3 (ch103): Attention Mechanism — The Breakthrough

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

---

## Part 4 (ch109): Text Generation Before Transformers — N-gram Language Models

If a word is surprising, the model is bad. Perplexity makes surprise a number. Smoothing keeps it finite.

Before transformers, a language model predicted the next word by counting how often it followed the previous n-1 words. That is an n-gram model. It ran every speech recognizer, spell checker, and phrase-based MT system from 1980 through 2015.

## The Concept

**N-gram probability:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Compute from counts: `P(w | context) = count(context, w) / count(context)`.

**The zero-count problem.** Any n-gram not seen in training gets probability zero. A 4-gram model on Brown has 30% of held-out 4-grams unseen.

**Smoothing approaches (in order of sophistication):**

1. **Laplace (add-one).** Adds 1 to every count. Simple, terrible.
2. **Good-Turing.** Reallocate mass from high-frequency events to unseen ones.
3. **Interpolation.** Combine n-gram, (n-1)-gram estimates with weights.
4. **Backoff.** Fall back to shorter context if n-gram count is zero.
5. **Absolute discounting.** Subtract fixed discount D from all counts.
6. **Kneser-Ney.** Absolute discounting + continuation probability (how many contexts a word appears in).

**Perplexity:** `exp(-(1/N) * Σ log P(w_i | context_i))`. Lower is better. A perplexity of 100 means the model is as confused as choosing uniformly among 100 words.

## Build It

### Step 1: Trigram Counts

```python
from collections import Counter, defaultdict

def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts
```

### Step 2: Laplace Smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

### Step 3: Kneser-Ney (Bigram, Interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Three moving parts: `continuation_prob` (the Kneser-Ney innovation), `lambda_prev` (mass freed by discount), and the final probability as discounted main term plus weighted continuation term.

### Step 4: Generating Text with Sampling

```python
import random

def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

### Step 5: Perplexity

```python
import math

def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

For Brown corpus, a well-tuned 4-gram KN model hits ~140 perplexity. A transformer LM hits 15-30. That 10x gap is why the field moved on.

## Exercises

1. **Easy.** Train a trigram LM on 1,000 Shakespeare sentences. Generate 20 sentences.
2. **Medium.** Implement perplexity for your KN model. Compare against Laplace.
3. **Hard.** Build a trigram spell corrector using LM context probability.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| N-gram | Sequence of n consecutive tokens. |
| Smoothing | Reallocating probability mass so unseen events get non-zero prob. |
| Perplexity | `exp(-average log-prob)` on held-out data. Lower is better. |
| Backoff | If trigram count is zero, use bigram. |
| Kneser-Ney | Absolute discounting + continuation probability. |
| Continuation probability | P(w) weighted by number of contexts w appears in. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer)
