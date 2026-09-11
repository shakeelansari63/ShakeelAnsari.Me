# Scaling Laws, KV Cache & Speculative Decoding

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch151): KV Cache, Flash Attention & Inference Optimization

> Training is parallel and FLOP-bound. Inference is serial and memory-bound. Different bottleneck, different tricks.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## The Problem

A naive autoregressive decoder does `O(N²)` work to generate `N` tokens: at each step it recomputes attention over the full prefix. For a 4K-token response that is 16M attention operations, most of them redundant.

On top of that, attention itself moves a lot of data. Standard attention materializes an N×N score matrix, N×d softmax output, N×d final output — too many reads and writes to HBM. For N≥2K, attention becomes memory-bound before it becomes FLOP-bound.

Two optimizations, both from Dao et al., pushed frontier inference from "slow" to "fast":

1. **KV cache.** Store the K and V vectors of every prefix token. Each new token's attention is one query against the cached keys. Inference reduces from `O(N²)` to `O(N)` per generation step.
2. **Flash Attention.** Tile the attention computation so the full N×N matrix never hits HBM. All of softmax + matmul happens in SRAM. 2–4× wall-clock speedup on A100; 5–10× on H100 with FP8.

## The Concept

### KV cache math

Per decoder layer, per token, per head:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
```

For a 7B model with 32 layers, 32 heads, d_head=128, fp16: per token = 16 KB. Per 32K context = 512 MB.

For Llama 3 70B (80 layers, GQA with 8 KV heads): per 32K context = 10.4 GB.

### Flash Attention — the tiling trick

Standard attention:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Three HBM round trips. On H100, HBM bandwidth is 3 TB/s; SRAM is 30 TB/s.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

One HBM trip per tile. Total memory footprint drops from `O(N²)` to `O(N)`.

**Version evolution:**

| Version | Year | Key change | Speedup |
|---------|------|-----------|---------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first | 3× on A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× on H100 |
| Flash 4 | 2026 | Blackwell 5-stage pipeline | Inference-first |

## Build It

### Step 1: KV cache

```python
class KVCache:
    def __init__(self):
        self.K = []
        self.V = []

    def append(self, k, v):
        self.K.append(k)
        self.V.append(v)

    def __len__(self):
        return len(self.K)
```

### Step 2: tiled softmax (Flash-style)

```python
def tiled_softmax_dot(q, Ks, Vs, tile=4):
    d_head = len(Vs[0])
    scale = 1.0 / math.sqrt(len(q))
    m = float("-inf")
    s = 0.0
    out = [0.0] * d_head
    for start in range(0, len(Ks), tile):
        k_block = Ks[start:start + tile]
        v_block = Vs[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) * scale for k in k_block]
        new_m = max(m, *scores)
        if m == float("-inf"):
            exp_old = 0.0
        else:
            exp_old = math.exp(m - new_m)
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(d_head):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Bit-identical output to `softmax(qK) V` in one shot.

### Step 3: compare naive vs cached decoding

```python
def decode_naive(all_K, all_V, all_queries):
    outputs = []
    ops = 0
    for t, q in enumerate(all_queries):
        Ks = all_K[:t + 1]
        Vs = all_V[:t + 1]
        out = attention_full(q, Ks, Vs)
        ops += t + 1
        outputs.append(out)
    return outputs, ops

def decode_cached(all_K, all_V, all_queries):
    cache = KVCache()
    outputs = []
    ops = 0
    for q, k, v in zip(all_queries, all_K, all_V):
        cache.append(k, v)
        out = attention_full(q, cache.K, cache.V)
        ops += len(cache)
        outputs.append(out)
    return outputs, ops
```

Naive: `O(N²)` ops. Cached: `O(N)` ops. Both produce identical output.

### Step 4: KV cache size table

```python
def kv_cache_bytes(N, n_layers, n_heads_kv, d_head, dtype=2):
    return 2 * N * n_layers * n_heads_kv * d_head * dtype
```

For Llama-3-70B at 128K context: ~10+ GB just for KV cache.

## Use It

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
)
```

vLLM production:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

## Ship It

See `outputs/skill-inference-optimizer.md`. The skill picks attention implementation, KV cache strategy, quantization, and speculative decoding.

## Exercises

1. **Easy.** Confirm the naive and cached decoders produce the same output; note the op-count difference.
2. **Medium.** Implement prefix caching: given a prompt P and several completions, run one forward pass over P to fill the KV cache, then branch per-completion.
3. **Hard.** Implement a toy PagedAttention: KV cache in fixed 16-token blocks with a free-list.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| KV cache | "The trick that makes decoding fast" | Stored K and V from every prefix token |
| HBM | "GPU main memory" | High Bandwidth Memory, ~3 TB/s bandwidth |
| SRAM | "On-chip memory" | Per-SM fast memory, ~30 TB/s bandwidth |
| Flash Attention | "Tiled attention kernel" | Computes attention without materializing N×N in HBM |
| Continuous batching | "No-wait batching" | Swap finished sequences out, new ones in |
| PagedAttention | "vLLM's headline" | KV cache in fixed blocks with a page table |
| Prefix caching | "Reuse long prompts" | Cache KV for a shared prefix across requests |
| Speculative decoding | "Draft + verify" | Cheap draft proposes tokens; big model verifies |

## Further Reading

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [Shah et al. (2024). FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Kwon et al. (2023). Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention)

---

## Part 2 (ch152): Scaling Laws

> The 2020 Kaplan paper said: bigger model, lower loss. The 2022 Hoffmann paper said: you were under-training. Compute goes into two buckets — parameters and tokens — and the split is not obvious.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## The Problem

When you have C FLOPs of training compute and want the best model, you face two knobs:

1. **How many parameters (N)?** Bigger model, higher capacity.
2. **How many training tokens (D)?** More data, better use of capacity.

FLOPs scale approximately as `6 × N × D`. You can push N up and D down, or D up and N down. Which is better?

Before 2022, the answer was "push N hard." GPT-3 (2020) was 175B parameters trained on ~300B tokens. A ratio of about 1.7 tokens per parameter. The Kaplan scaling laws backed this up.

Hoffmann et al. (2022), training a small family of models called Chinchilla, found something different: optimal ratio is closer to **20 tokens per parameter**. GPT-3 was 10× undertrained. Chinchilla (70B params, 1.4T tokens) beat GPT-3 (175B, 300B tokens) on every benchmark at 2.5× less inference cost.

2026 is Chinchilla's world — with one important twist. Llama 3 8B was trained on 15 trillion tokens, a ratio of 1,875 tokens per parameter. Inference cost matters more than training cost for models that will be used at scale.

## The Concept

### The Hoffmann law

From the Chinchilla paper, loss follows:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters (non-embedding).
- `D` = training tokens.
- `α ≈ 0.34`, `β ≈ 0.28` (roughly symmetric).
- `E ≈ 1.69`, the irreducible loss ceiling.
- `A ≈ 406`, `B ≈ 411`.

Two terms trade against each other as you scale. Take the derivative w.r.t. `N` at fixed compute (C = 6ND) and solve:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal: 20 tokens per parameter.

### Why over-training anyway

Chinchilla-optimal minimizes training loss per training FLOP. But you pay training cost once; inference cost forever.

For a chatbot that serves a trillion tokens per month, inference dominates total cost. Llama's approach: train smaller, longer. 8B at 15T tokens is deeply inference-optimized.

### The 2026 picture

| Factor | Changed how |
|--------|-------------|
| Data quality | Curating "good" tokens shifts curves by >2× effective compute |
| MoE | Total params decouple from active FLOPs |
| Post-training | Some capabilities shift with SFT+RLHF more than pretraining |
| Multimodality | Image + text tokens scale together |
| Synthetic data | Models generate training data; effective compute can compound |

## Build It

### Step 1: Chinchilla loss

```python
A = 406.4
B_CONST = 410.7
ALPHA = 0.34
BETA = 0.28
E_CONST = 1.69

def chinchilla_loss(N, D, A=A, B=B_CONST, alpha=ALPHA, beta=BETA, E=E_CONST):
    return A / N ** alpha + B / D ** beta + E
```

### Step 2: compute-optimal frontier

```python
def compute_optimal(C_flops, n_grid=200):
    log_N_min = math.log10(1e5)
    log_N_max = math.log10(1e13)
    best = (None, None, float("inf"))
    for i in range(n_grid):
        log_N = log_N_min + (log_N_max - log_N_min) * i / (n_grid - 1)
        N = 10 ** log_N
        D = C_flops / (6 * N)
        if D < 1e6:
            continue
        loss = chinchilla_loss(N, D)
        if loss < best[2]:
            best = (N, D, loss)
    return best
```

For compute budgets from `1e17` to `1e25` FLOPs, verify the ratio `D/N ≈ 20`.

### Step 3: over-training cost

```python
C = 1e24
N_opt, D_opt, L_opt = compute_optimal(C)
N_under = N_opt / 10
D_over = D_opt * 10
L_over = chinchilla_loss(N_under, D_over)
print(f"chinchilla optimal:  loss={L_opt:.3f}")
print(f"over-trained:        loss={L_over:.3f}")
print(f"inference savings:   {N_opt / N_under:.0f}x")
```

### Step 4: compare to real models

```python
models = [
    ("GPT-3 175B",          175e9,  300e9),
    ("Chinchilla 70B",       70e9, 1400e9),
    ("Llama 2 70B",          70e9, 2000e9),
    ("Llama 3 8B",            8e9, 15_000e9),
    ("Llama 3 70B",          70e9, 15_000e9),
    ("DeepSeek-V3 (active)", 37e9, 14_800e9),
    ("Qwen 2.5 72B",         72e9,  18_000e9),
]
for name, N, D in models:
    L = chinchilla_loss(N, D)
    print(f"  {name:<22}  D/N={D/N:>6.1f}  loss={L:>6.3f}")
```

## Use It

Scaling laws tell you:

1. **Whether your fine-tune has enough data.** If your task-specific data is below 20 tokens per param of the base model, expect saturation.
2. **Whether to pick a bigger base model.** If spending all budget on inference, prefer a smaller, longer-trained model.
3. **Where the returns diminish.** Beyond 1000× Chinchilla-optimal, log-loss changes become noise.

## Ship It

See `outputs/skill-training-budget-estimator.md`. The skill picks `(N, D, hours, GPU)` for a new training run.

## Exercises

1. **Easy.** Print Chinchilla-optimal `(N, D)` for compute budgets `1e20`, `1e22`, `1e24`. Compare to the real model table.
2. **Medium.** Implement the Hoffmann loss-as-function-of-compute curve. Identify when `>10^28` FLOPs are needed for the next 0.1 reduction in cross-entropy.
3. **Hard.** Fit your own scaling law on 5 tiny models (100K to 10M params) trained on the same dataset.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Parameters (N) | "Model size" | Non-embedding weight count; determines capacity |
| Tokens (D) | "Training data" | Number of training tokens seen |
| Compute (C) | "FLOPs spent" | Approximately `6 × N × D` |
| Chinchilla-optimal | "D/N ≈ 20" | Ratio minimizing loss per FLOP of pretraining |
| Over-training | "Past Chinchilla" | Spend extra training FLOPs to save inference FLOPs |
| Irreducible loss | "The floor" | The `E` term; the entropy of the data itself |
| Emergent capability | "Sudden jumps at scale" | Often a scorer artifact; continuous loss is smooth |
| Effective compute | "Training-efficiency multiplier" | Better data/optimizer/architecture multiplies a FLOP |

## Further Reading

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/13-scaling-laws)

---

## Part 3 (ch153): Build a Transformer from Scratch — The Capstone

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

---

## Part 4 (ch155): Speculative Decoding — Draft, Verify, Repeat

> Autoregressive decoding is serial. Each token waits for the previous one. Speculative decoding breaks the chain: a cheap model drafts N tokens, the expensive model verifies all N in one forward pass. When the draft is right you paid one big forward for N generations.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## The Problem

A 70B LLM sampling one token takes ~30 ms on an H100. A 3B draft model takes ~3 ms. If we let the 3B draft 5 tokens ahead, then run the 70B *once* to verify all 5, the total is `5×3 + 30 = 45 ms` for up to 5 accepted tokens — versus `5×30 = 150 ms` for straight-line generation. That is the full speculative-decoding pitch: trade a small amount of extra GPU memory (draft model) for 2–4× lower decode latency.

The trick preserves the distribution. Speculative sampling guarantees that the output sequence is **identically distributed** to what the big model would have produced on its own. No quality tradeoff. Just faster.

Four families of draft-verifier pairs dominate 2026 inference:

1. **Vanilla speculative (Leviathan 2023).** Separate draft model + verifier.
2. **Medusa (Cai 2024).** Multiple decoding heads on the verifier predict `t+1..t+k` in parallel.
3. **EAGLE family (Li 2024, 2025).** Lightweight draft that reuses the verifier's hidden states.
4. **Lookahead decoding (Fu 2024).** Jacobi iteration; no draft model required.

## The Concept

### The core algorithm

Given a verifier `M_q` and a cheaper draft `M_p`:

1. Let `x_1..x_k` be the prefix already decoded.
2. **Draft**: use `M_p` to autoregressively propose `d_{k+1}, ..., d_{k+N}` with probabilities `p_1..p_N`.
3. **Verify in parallel**: run `M_q` once on `x_1..x_k, d_{k+1}, ..., d_{k+N}`, getting probabilities `q_1..q_{N+1}`.
4. **Accept/reject**: for each `i`, accept with probability `min(1, q_i(d_i) / p_i(d_i))`.
5. On first rejection at position `j`: sample from residual `(q_j - p_j)_+` normalized.
6. On accepting all `N`: sample one bonus token from `q_{N+1}`.

### What determines speedup

Let `α` = expected acceptance rate. Per step:

- Naive: 1 big-model call per token.
- Speculative: 1 big-model call per `(1 - α^{N+1}) / (1 - α)` tokens.

At `α = 0.75` and `N = 5`: ~3× fewer big-model calls.

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft | Fast prototype, no training | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, max speed | 3–4× |
| Lookahead | No draft, no training | 1.3–1.6× |

## Build It

### Step 1: the rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` is a uniform random number. This Bernoulli decision preserves the verifier's distribution exactly.

### Step 2: residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0.0:
        return list(q)
    return [r / s for r in raw]
```

Subtract `p` from `q` element-wise, clamp negatives to zero, renormalize.

### Step 3: one speculative step (N draft tokens)

```python
def spec_step_n(q, p, N, rng):
    accepted = 0
    for _ in range(N):
        d = sample(p, rng)
        p_prob = p[d]
        q_prob = q[d]
        u = rng.random()
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            accepted += 1
        else:
            return sample(residual(q, p), rng), accepted
    bonus = sample(q, rng)
    return bonus, accepted + 1
```

Five accepted → one bonus → six tokens produced in one verifier pass.

### Step 4: measure acceptance rate

```python
def acceptance_rate(q, p, n_samples, rng):
    hits = 0
    for _ in range(n_samples):
        d = sample(p, rng)
        u = rng.random()
        q_prob = q[d]
        p_prob = p[d]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            hits += 1
    return hits / n_samples
```

### Step 5: verify distribution equivalence

```python
def run_distribution_check(q, p, n_samples, rng):
    spec_counts = [0] * len(q)
    direct_counts = [0] * len(q)
    for _ in range(n_samples):
        d, _ = spec_step_one_token(q, p, rng)
        spec_counts[d] += 1
        direct_counts[sample(q, rng)] += 1
    return spec_counts, direct_counts
```

A chi-square test confirms the speculative histogram matches the direct sample within sampling error.

### Step 6: expected tokens per verify call

```python
def expected_tokens_per_verify(alpha, N):
    if alpha >= 1.0:
        return N + 1
    if alpha == 0:
        return 1
    return (1 - alpha ** (N + 1)) / (1 - alpha)
```

At `α=0.85` and `N=5`: ~4.1 tokens per verifier call ≈ 4× fewer big-model forwards.

## Use It

Production:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --num-speculative-tokens 5

# vLLM with vanilla draft
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

**When NOT to spec-decode:**
- Single-sequence generation of 1–5 tokens (overhead dominates).
- Wildly creative / high-temperature sampling (α drops).
- Memory-constrained deployments (draft model adds VRAM).

## Ship It

See `outputs/skill-spec-decode-picker.md`. The skill picks a speculative decoding strategy and tuning parameters.

## Exercises

1. **Easy.** Confirm the speculative token distribution matches the verifier's direct-sample distribution on 50,000 tokens within chi-square p > 0.05.
2. **Medium.** Plot speedup as a function of `N` for `α = 0.5, 0.7, 0.85`. Identify optimal `N` for each.
3. **Hard.** Implement a tiny Medusa: take the capstone GPT, add 3 extra LM heads predicting t+2, t+3, t+4.
4. **Hard.** Implement KV rollback: feed 5 draft tokens, simulate rejection at position 3, verify cache reads correctly.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | Smaller model proposing candidate tokens |
| Verifier | "The big one" | Target model whose distribution we preserve |
| Acceptance rate (α) | "How often the draft is right" | Per-token acceptance probability, 0.7–0.9 typical |
| Residual distribution | "The rejection fallback" | `(q - p)_+` normalized; preserves verifier distribution |
| Bonus token | "The free one" | Extra token sampled when all N drafts are accepted |
| Medusa | "Draft-less speculative" | Multiple LM heads predicting future tokens |
| EAGLE | "Hidden-state draft" | Tiny draft conditioned on verifier's hidden states |
| Lookahead decoding | "Jacobi iteration" | Self-speculation with no draft model |
| Tree attention | "Verify many candidates at once" | Branching verification for multiple continuations |
| KV rollback | "Undo rejected drafts" | Scratch buffer; commit on acceptance, discard on reject |

## Further Reading

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al. (2023). Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774)
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [Li et al. (2025). EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference](https://arxiv.org/abs/2402.02057)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/16-speculative-decoding)
