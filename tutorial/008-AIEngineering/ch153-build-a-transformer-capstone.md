# Build a Transformer from Scratch — The Capstone

> Thirteen lessons. One model. No shortcuts.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## The Problem

You've read every paper. You've implemented attention, multi-head splits, positional encodings, encoder and decoder blocks, BERT and GPT losses, MoE, KV cache. Now make them work together on a real task.

The capstone: train a small decoder-only transformer end-to-end on a character-level language modeling task. It reads Shakespeare. It generates new Shakespeare. It is small enough to train on a laptop in under 10 minutes.

This is the "nanoGPT" of the course.

## The Concept

The architecture, annotated:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │
│  MultiHeadAttention (causal)      │
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │
│  residual                         │
└───────────────────────────────────┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy
```

### Target metrics

On a Mac M2 laptop, a 4-layer, 4-head, d_model=128 GPT trained for 2,000 steps on `tinyshakespeare.txt`:

- Training loss converges from ~4.2 (random) to ~1.5 in about 6 minutes.
- Sampled output looks Shakespeare-shaped.
- Val loss tracks training loss closely; no overfitting.

## Build It

### Step 1: data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 unique characters. No BPE.

### Step 2: model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)

class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, block_size):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.d_head = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, N, self.h, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.h, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.h, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :N, :N] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(y)

class SwiGLUFFN(nn.Module):
    def __init__(self, d, expansion):
        super().__init__()
        h = int(d * expansion)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, d, h, block_size, expansion):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, h, block_size)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, expansion)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d, h, n_layers, block_size, expansion):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, h, block_size, expansion) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, N = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(N, device=idx.device))
        x = tok + pos
        for b in self.blocks:
            x = b(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
```

### Step 3: training loop

```python
def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([src[i + 1:i + 1 + block_size] for i in ix]).to(device)
    return x, y

model = GPT(vocab_size, d_model, n_heads, n_layers, block_size, ffn_expansion).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

for step in range(max_steps + 1):
    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            x, y = get_batch("train")
            _, train_loss = model(x, y)
            x, y = get_batch("val")
            _, val_loss = model(x, y)
        model.train()
        print(f"  step {step:>4}  train={train_loss.item():.3f}  val={val_loss.item():.3f}")
    if step == max_steps:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

### Step 4: sample

```python
prompt = torch.tensor([[stoi["F"], stoi["i"], stoi["r"], stoi["s"], stoi["t"]]], dtype=torch.long)
out = model.generate(prompt, max_new_tokens=200, temperature=0.9, top_k=10)
print("".join(itos[int(i)] for i in out[0].tolist()))
```

After 2,000 steps:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

## Use It

This capstone is a reference architecture. Three extensions:

1. **Swap the tokenizer.** Use BPE (e.g. `tiktoken.get_encoding("cl100k_base")`).
2. **Train on a bigger corpus.** Use `OpenWebText` or `fineweb-edu`.
3. **Add RoPE + KV cache + Flash Attention.**

## Ship It

See `outputs/skill-transformer-review.md`. The skill reviews a transformer-from-scratch implementation for correctness.

## Exercises

1. **Easy.** Run the model. Verify final-step validation loss is under 2.0.
2. **Medium.** Replace learned positional embeddings with RoPE.
3. **Medium.** Implement a KV cache in the sampling loop.
4. **Hard.** Add a second head predicting the next-plus-one token (MTP).
5. **Hard.** Replace the single FFN per block with a 4-expert MoE.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial uses it |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix |
| bf16 autocast | "Training precision trick" | Forward/back in bf16, optimizer state in fp32 |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0 |
| Cosine LR schedule | "The 2020+ default" | LR ramps up then decays cosine-shaped |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw |

## Further Reading

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)
- Karpathy's nanoGPT (GitHub)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/14-build-a-transformer-capstone)
