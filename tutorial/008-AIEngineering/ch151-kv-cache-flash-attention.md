# KV Cache, Flash Attention & Inference Optimization

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
