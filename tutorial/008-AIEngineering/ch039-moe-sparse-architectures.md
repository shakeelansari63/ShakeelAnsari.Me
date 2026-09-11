# MoE, Audio Transformers & Sparse Attention

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch149): Audio Transformers — Whisper Architecture

> Audio is an image of frequency over time. Whisper is a ViT that eats mel spectrograms and speaks back.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## The Problem

Before Whisper (OpenAI, Radford et al. 2022), state-of-the-art automatic speech recognition (ASR) meant wav2vec 2.0 and HuBERT — self-supervised feature extractors plus a fine-tuned head. High quality, expensive data pipelines, domain-brittle.

Whisper made three bets:

1. **Train on everything.** 680,000 hours of weakly-labeled audio scraped from the internet across 97 languages.
2. **Multi-task single model.** One decoder trained jointly on transcription, translation, voice activity detection, language ID, and timestamping via task tokens.
3. **Standard encoder-decoder transformer.** Encoder consumes log-mel spectrograms. Decoder produces text tokens autoregressively.

## The Concept

### Step 1 — resample + window

Audio at 16 kHz. Clip/pad to 30 seconds. Compute log-mel spectrogram: 80 mel bins, 10 ms stride → ~3,000 frames × 80 features.

### Step 2 — convolutional stem

Two Conv1D layers with kernel 3 and stride 2 reduce the 3,000 frames to 1,500.

### Step 3 — encoder

A 24-layer (for large) transformer encoder over 1,500 timesteps. Sinusoidal positional encoding, self-attention, GELU FFN. Produces 1,500 × 1,280 hidden states.

### Step 4 — decoder

A 24-layer transformer decoder. Autoregressively produces tokens from a BPE vocabulary that is a superset of GPT-2's with audio-specific special tokens.

### Step 5 — task tokens

The decoder prompt starts with control tokens:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

or

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

You control task by prefix.

### Whisper sizes

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Turbo | 809M | 32 | 1280 | 20 | ~6 GB |

Large-v3-turbo (2024) cut the decoder from 32 layers to 4. 8× faster decoding with <1 WER point regression.

## Build It

### Step 1: synthesize audio

```python
SAMPLE_RATE = 16000
FRAME_SIZE = 400   # 25 ms at 16 kHz
HOP = 160          # 10 ms at 16 kHz
TARGET_FRAMES = 3000

def sine_wave(freq, duration_s, sr=SAMPLE_RATE):
    n = int(duration_s * sr)
    return [math.sin(2 * math.pi * freq * i / sr) for i in range(n)]
```

### Step 2: log-mel spectrogram (simplified)

```python
def frame_signal(x, frame_size=FRAME_SIZE, hop=HOP):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames

def frame_energy(frame):
    e = sum(v * v for v in frame)
    return math.log(e + 1e-9)
```

Frame = 25 ms, hop = 10 ms. Matches Whisper's windowing. Per-frame energy stands in for mel bins.

### Step 3: pad to 30 s

```python
def pad_or_clip(frames, target):
    if len(frames) >= target:
        return frames[:target]
    pad_frame = [0.0] * len(frames[0]) if frames else [0.0] * FRAME_SIZE
    return frames + [pad_frame] * (target - len(frames))
```

### Step 4: build the prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

That is the whole task-control surface. A 4-token prefix.

## Use It

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
```

Faster, OpenAI-compatible:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

## Ship It

See `outputs/skill-asr-configurator.md`. The skill picks an ASR model, decoding parameters, and preprocessing pipeline for a new speech application.

## Exercises

1. **Easy.** Confirm the frame count for a 1-second signal at 16 kHz with 10 ms hop is ~100 frames. For 30 seconds: ~3,000 frames.
2. **Medium.** Build the full log-mel spectrogram using `numpy.fft`. Verify 80 mel bins match `librosa.feature.melspectrogram(n_mels=80)`.
3. **Hard.** Implement streaming inference: chunk audio into 10 s windows with 2 s overlap, run Whisper on each chunk, merge transcripts.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | "Audio image" | 2D representation: frequency bins × time frames; log-scaled energy per cell |
| Log-mel | "What Whisper sees" | Mel spectrogram passed through log |
| Frame | "One time slice" | A 25 ms window of samples; overlapping at 10 ms stride |
| Task token | "Prompt prefix for speech" | Special tokens like `<\|transcribe\|>` / `<\|translate\|>` |
| Whisper-turbo | "Small decoder, full encoder" | large-v3 encoder + 4-layer decoder; 8× faster |
| Faster-whisper | "The production wrapper" | CTranslate2 reimplementation; int8 quantization |

## Further Reading

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription](https://arxiv.org/abs/2410.15608)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/10-audio-transformers-whisper)

---

## Part 2 (ch150): Mixture of Experts (MoE)

> A dense 70B transformer activates every parameter for every token. A 671B MoE activates only 37B per token and beats it on every benchmark. Sparsity is the most important scaling idea of the decade.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## The Problem

A dense transformer's FLOPs at inference equal its parameter count (times 2 for forward pass). Scale up a dense model and every token pays the full bill. By 2024 the frontier was hitting a compute wall: to be meaningfully smarter, you needed exponentially more FLOPs per token.

Mixture of Experts breaks this link. Replace each FFN with `E` independent experts + a router that picks `k` experts per token. Total parameters = `E × FFN_size`. Active parameters per token = `k × FFN_size`. Typical 2026 configuration: `E=256`, `k=8`. Storage scales with `E`, compute scales with `k`.

The 2026 frontier is almost entirely MoE: DeepSeek-V3 (671B total / 37B active), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2.

## The Concept

### The FFN swap

Dense transformer block:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block:

```
h = x + attn(norm(x))
scores = router(norm(h))
top_k = argmax_k(scores)
h = h + sum_{e in top_k}( gate(scores[e]) * Expert_e(norm(h)) )
```

### The load-balancing problem

If the router puts 90% of tokens through expert 3, the other experts starve. Three fixes:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Add a penalty proportional to the variance in expert usage.
2. **Expert capacity + token dropping** (early Switch). Each expert processes at most `C × N/E` tokens; overflow skips the layer.
3. **Auxiliary-loss-free balancing** (DeepSeek-V3). Add a learned per-expert bias that shifts selection. No penalty on the main objective.

### Shared experts

DeepSeek-V2/V3 splits experts into *shared* and *routed*. Every token passes through all shared experts. Routed experts are picked via top-k.

### The cost profile

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

## Build It

### Step 1: the router

```python
def route(hidden, W_router, top_k, bias):
    E = len(W_router)
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(E)]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(E), key=lambda i: -biased[i])[:top_k]
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias affects selection, not gate weight. That is the DeepSeek-V3 trick.

### Step 2: auxiliary-loss-free balancing

```python
def update_bias(bias, usage_counts, target, gamma):
    for e in range(len(bias)):
        if usage_counts[e] > target:
            bias[e] -= gamma
        elif usage_counts[e] < target:
            bias[e] += gamma
    return bias
```

### Step 3: run 100 tokens through the router

```python
def run_epoch(tokens, experts, W_router, top_k, bias):
    usage = [0] * len(experts)
    for x in tokens:
        out = [0.0] * d_hidden
        top_idx, gates = route(x, W_router, top_k, bias)
        for e_idx, gate in zip(top_idx, gates):
            h = apply_expert(x, experts[e_idx])
            for j in range(d_hidden):
                out[j] += gate * h[j]
        for e in top_idx:
            usage[e] += 1
    return usage
```

Track which experts fire how often. Without the bias, usage is skewed. With bias update, usage converges to uniform.

### Step 4: param count comparison

```python
def dense_active_params(n_experts, expert_params, top_k, d_model):
    total = n_experts * expert_params
    active = top_k * expert_params
    return total, active
```

DeepSeek-V3-shaped: 256 routed + 1 shared, 8 active, d_model=7168. The total parameter count is eye-watering. The active count is a seventh of a dense Llama 3 70B.

## Use It

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

**When to pick MoE:** You want frontier quality at lower inference cost per token, have VRAM / expert-parallel infrastructure, and your workload is token-heavy.

**When NOT to pick MoE:** Edge deployment, latency-critical single-user serving, small models (<7B).

## Ship It

See `outputs/skill-moe-configurator.md`. The skill picks E, k, and shared-expert layout for a new MoE.

## Exercises

1. **Easy.** Watch how the auxiliary-loss-free bias update evens out expert usage over 50 iterations.
2. **Medium.** Replace the learned router with a hash-based router. Compare quality and balance.
3. **Hard.** Implement GRPO-style "rollout-matched routing": log which experts fire during inference, force same routing during gradient computation.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | "One FFN among many" | An independent feed-forward network |
| Router | "The gate" | Tiny linear layer that scores each token against each expert |
| Top-k routing | "k active experts per token" | Each token goes through exactly k experts |
| Auxiliary loss | "Load-balance penalty" | Extra loss term penalizing skewed expert usage |
| Auxiliary-loss-free | "DeepSeek-V3's trick" | Balance via per-expert bias on selection only |
| Shared expert | "Always on" | Expert through which every token passes |
| Expert parallelism | "Shard by expert" | Distribute experts to different GPUs |
| Sparsity | "Active params < total params" | The ratio `k × expert_size / (E × expert_size)` |

## Further Reading

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Fedus, Zoph, Shazeer (2022). Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/11-mixture-of-experts)

---

## Part 3 (ch198): Differential Attention (V2)

> Softmax attention spreads a small amount of probability over every non-matching token. Over 100k tokens that noise adds up. Differential Transformer computes attention as the difference of two softmaxes, subtracting the shared noise floor.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 10 · 14 (architecture walkthroughs)
**Time:** ~60 minutes

## Learning Objectives

- State why softmax attention has a noise floor and why it grows with context length
- Derive the differential attention formula and explain why subtraction cancels shared noise
- Walk the V1-to-V2 diff and explain each change
- Implement differential attention and verify noise cancellation on synthetic data

## The Problem

Softmax can never produce exact zeros. Every non-matching token gets some positive mass. At 128k tokens, even 0.001% per token sums to ~12% of the total probability. The model learns to route around a noise floor that grows with context.

Empirically: hallucinated citations in long-context RAG, lost-in-the-middle failures on 100k-token retrieval, subtle accuracy degradation past 32k.

DIFF V1 (ICLR 2025) measured the gap: Differential Transformers hit lower perplexity and higher long-context accuracy than same-size baselines. DIFF V2 (January 2026) made it production-ready.

## The Concept

### The Noise Floor

For query `q` and keys `K = [k_1, ..., k_N]`:

`w_i = exp(q·k_i / √d) / Σ_j exp(q·k_j / √d)`

No `w_i` is ever zero. Each unrelated token contributes `O(1/N)`. Total noise: `O(1)`.

### The Differential Idea

Split each head's Q and K into two: `Q = (Q_1, Q_2)`, `K = (K_1, K_2)`. Compute two attention maps:

```
A1 = softmax(Q_1 K_1^T / √d)
A2 = softmax(Q_2 K_2^T / √d)
DiffAttn = (A1 - λ * A2) V
```

The subtraction cancels whatever noise the two maps share (the 127k unrelated tokens). Signal -- peaked weight on relevant tokens -- only cancels if it appears in both at the same magnitude, which training prevents.

`λ` is a learnable per-head scalar: `λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init`. Can be negative.

### V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Head dim | Halved (parameter neutral) | Same as baseline |
| Q heads | Same as baseline | Doubled |
| KV cache | Loaded twice per decode | Loaded once |
| FlashAttention | Custom kernel needed | Compatible |
| Per-head RMSNorm | Required for stability | Removed (simpler init) |
| Decode speed | Slower than baseline | Matches baseline |

V2 doubles Q heads while keeping KV heads the same, borrowing parameters from the up-projection. After subtraction, the extra dimension projects back down.

### When to Reach For It

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Cleaner attention, fewer hallucinations |
| Needle-in-haystack (32k+) | Substantial accuracy lift |
| Short chat (< 4k) | Indistinguishable from baseline |

## Build It

### Step 1: Standard Softmax Attention

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2: Split Q, K into Two Halves

V1 style: halve head dimension. V2 style: double Q heads. The toy uses V1 for clarity.

### Step 3: Two Softmax Branches + Subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Output weights can be negative -- V projection absorbs the sign.

### Step 4: Noise Cancellation Measurement

Build synthetic sequence of length 1024. Place signal at known position, rest is noise. Compare standard vs differential attention weight on the signal position. DIFF produces 3-10x higher signal-to-noise ratio.

### Step 5: V1 vs V2 Parameter Accounting

For hidden=4096, heads=32, d_head=128: V2 adds roughly `hidden * hidden` extra per attention block (doubling Q size).

## Use It

DIFF V2 integration is underway in vLLM and SGLang as of April 2026. Reach for it when training a new model targeting 64k+ context. Apply LoRA on Q projections to approximate DIFF on existing weights.

## Ship It

This lesson produces `outputs/skill-diff-attention-integrator.md` -- integration plan for adding DIFF attention to pre-training or fine-tuning.

## Exercises

1. Run the implementation and verify signal-to-noise is higher for DIFF. Vary noise amplitude to find crossover.
2. Compute parameter-count delta from baseline to V1 and V2 for 7B-class model.
3. Read DIFF V1 Section 3 and V2 Section 2. Explain why V1 needed per-head RMSNorm and V2 removed it.
4. Sweep λ from 0 to 1. Measure signal-to-noise. Find the optimal λ.
5. Extend to GQA + DIFF V2: pick 8 KV heads, 32 Q heads. Show KV cache matches baseline.

## Further Reading

- Ye et al., "Differential Transformer" (arXiv:2410.05258, ICLR 2025)
- Microsoft unilm, "Differential Transformer V2" (HuggingFace blog, January 2026)
- Liu et al., "Lost in the Middle" (arXiv:2307.03172)

---

## Part 4 (ch199): Native Sparse Attention (DeepSeek NSA)

> At 64k tokens, attention eats 70-80% of decode latency. DeepSeek's NSA (ACL 2025 Best Paper) runs three parallel attention branches -- compressed coarse-grained, selectively retained fine-grained, and sliding windows -- combined through a learned gate.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 14 (architecture walkthroughs)
**Time:** ~60 minutes

## Learning Objectives

- State the three NSA attention branches and what each captures
- Explain why NSA is "natively trainable" where prior sparse-attention methods were inference-only
- Compute the attention savings vs full attention at 64k context
- Implement the three-branch combination and verify gating weights behave

## The Problem

Full attention at sequence N costs `O(N^2)` time and `O(N)` KV cache. At 64k tokens, attention accounts for 70-80% of decode latency.

Prior sparse attention falls into two buckets: fixed-pattern (sliding window, strided) that fails on long-range recall, and inference-time pruning (H2O, StreamingLLM) that recovers only partial speedup because the model was not trained to route through the sparse pattern.

NSA is both: a sparsity pattern learned during pre-training, implemented as a kernel-aligned algorithm.

## The Concept

### Three Parallel Branches

For each query, NSA runs attention three times against different views:

1. **Compressed branch.** Tokens grouped into blocks of size `l` (typically 32-64). Each block compressed to one summary token via a learned MLP. Query attends over compressed tokens for a coarse-grained view.

2. **Selected branch.** Using compressed-branch attention scores, top-k most relevant blocks are identified. Fine-grained (uncompressed) tokens from those blocks are loaded and attended.

3. **Sliding-window branch.** Query attends to the most recent `W` tokens (typically 512) for local context.

Output = `g_cmp * out_cmp + g_sel * out_sel + g_win * out_win`, gates from a small MLP on the query.

### Why It Is Natively Trainable

The selection step (top-k blocks) is discrete, breaking gradient flow. NSA sidesteps this: the compressed-branch attention is a differentiable coarse-grained attention on the whole sequence. The top-k just reuses top attention scores to pick which fine-grained blocks to load. Gradients flow through compressed-branch scores. Top-k is a no-op on the forward graph -- it only controls memory loading.

### Compute Budget

With `N=64k, l=64, k=16, b=64, w=512`:

- Compressed: `N/l = 1000` keys per query
- Selected: `k*b = 1024` keys per query
- Sliding: `w = 512` keys per query
- Total: 2536 vs 64000 for full attention -- 25x reduction

At 128k: 3536 vs 128000 -- 36x reduction. Benefit grows with sequence length.

### Hardware Alignment

Loads queries by GQA groups (outer loop), fetches sparse KV blocks per group (inner loop), runs attention on SRAM. KV loads amortized across the group. Reported 9x faster than FlashAttention on 64k decodes.

### Comparison

| Method | Differentiable | Real speedup | Long-range recall |
|--------|---------------|-------------|-------------------|
| Sliding window only | yes | yes | fails |
| KV pruning | N/A (inference) | yes | partial |
| NSA | natively | 9x at 64k | matches full attention |

## Build It

### Step 1: Compress Tokens into Blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2-4: Three Branch Attentions

Compressed attention against compressed keys. Top-k selection based on compressed scores. Slide window attention on last W tokens.

### Step 5: Gate + Combine

Small MLP on query produces three gate weights. Weighted sum of branch outputs.

### Step 6: Compute Counting

On 1024-token synthetic with l=32, k=4, w=128: NSA sees 32 + 128 + 128 = 288 keys per query vs 1024 for full -- 3.5x fewer.

## Use It

NSA ships in DeepSeek's own long-context pre-training. Integration in vLLM/SGLang is experimental as of April 2026.

When to reach: pre-training targeting 64k+ context with a serious compute budget. Inference of DeepSeek's own long-context checkpoints.

When not: serving existing dense-attention models (cannot retrofit). Context under 16k.

## Ship It

This lesson produces `outputs/skill-nsa-integrator.md` -- NSA integration plan for long-context pre-training.

## Exercises

1. Sweep (l, k, w) across presets and find the lowest compute that maintains 95% recall on needle-in-haystack.
2. Replace mean-pool compressor with a trained MLP and measure perplexity gap.
3. Implement the gate MLP and show it behaves sensibly.
4. Compute KV cache for NSA-enabled 70B at 128k vs full attention and MLA.
5. Read NSA Section 4 and explain why compressed-branch scores are reused for selection.

## Further Reading

- Yuan et al., "Native Sparse Attention" (arXiv:2502.11089, ACL 2025 Best Paper)
- DeepSeek-V3 Technical Report (arXiv:2412.19437)
- Moonshot AI, "MoBA: Mixture of Block Attention" (arXiv:2502.13189)
- Dao et al., "FlashAttention-2" (arXiv:2307.08691)

---

## Part 5 (ch203): Jamba -- Hybrid SSM-Transformer

> AI21's Jamba puts Transformer and Mamba layers in the same model: 1 Transformer layer for every 7 Mamba layers, MoE on every other block, and a 256k context window that fits on a single 80GB GPU.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (NSA)
**Time:** ~60 minutes

## Learning Objectives

- Explain the three primitives in a Jamba block and the 1:7:even interleaving recipe
- State what an SSM's recurrence looks like and why it enables constant-memory inference
- Compute the KV cache footprint of a Jamba model at 256k vs a pure Transformer
- Name the three Mamba-3 innovations and the problem each targets

## The Problem

Attention is quadratic. SSMs are linear. At 256k tokens, a Transformer attention map is 65B entries per head; an SSM's recurrent state is fixed-size.

Pure-SSM models match Transformer perplexity at small scales but lag on state-tracking and in-context retrieval. SSMs compress history into a fixed state, and when history is long, information leaks.

The fix: use both. Jamba is the first production-grade hybrid at scale (52B total, 12B active, 256k context).

## The Concept

### An SSM in One Page

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Computing `y_t` needs only `h_{t-1}` and `x_t`. Memory is constant. Inference is O(1) per token.

S4 used structured A matrix. Mamba made A, B, C data-dependent ("selective"). Mamba-2 simplified further. Mamba-3 adds complexity in specific places.

### The Jamba Block

Interleaves layers by two numbers:
- `l = 8`: 1 Transformer for every 7 Mamba layers
- `e = 2`: every other layer applies MoE

Layer sequence within a block:
```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (| marks MoE)
```

At 4 blocks (32 layers): 28 Mamba + 4 Attention. 16 use MoE.

### Why 1:7

AI21 ran ablations. Too much attention (1:1): quality up but memory degrades. Too little (1:15): memory great but retrieval fails. Sweet spot: 1:7 or 1:8. Transformer layers handle exact recall and state tracking. Mamba layers handle cheap bulk processing.

### The Memory Budget

For Jamba-1 (32 layers: 28 Mamba + 4 Attention, hidden 4096, 32 heads):
- KV cache (attention only): 2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB
- SSM state: 28 * 4096 * 16 * 2 = 3.7 MB (fixed, does not grow with sequence)

Compare pure Transformer (32 layers, full MHA): 2 * 32 * 32 * 128 * 256k * 2 = 128 GB. Even vs GQA(8): 32 GB. Jamba's hybrid is 8x to 16x smaller.

### Mamba-3 (ICLR 2026)

Three innovations:

1. **Exponential-trapezoidal discretization.** More expressive recurrence than Mamba-2's Euler method.
2. **Complex-valued state update.** Re-adds complex values (removed in Mamba-2). Equivalent to data-dependent rotary embedding on the state. Restores state-tracking.
3. **Multi-input multi-output (MIMO) projections.** Matrix-valued projections instead of per-feature scalar. Improves modeling and hardware utilization.

At 1.5B: +0.6 points over Gated DeltaNet. MIMO variant adds +1.2 more.

### When to Reach for Hybrid

Win when: context is 64k+, tasks mix short-range structure with long-range recall, single-GPU memory where Transformer KV cache alone would not fit.

Lose when: context under 16k, tasks needing everywhere-to-everywhere attention (cross-document references), scaling to trillion-parameter frontier (pure Transformer + MLA + MoE currently winning).

## Build It

The code is a memory calculator for hybrid architectures. Given SSM-Transformer ratio, hidden size, and layer count, it computes KV cache at target context, SSM state memory, and total memory for pure Transformer, Jamba 1:7 hybrid, and pure SSM.

## Use It

Run the calculator to reproduce Jamba's 8x memory reduction claim. Most production servers (vLLM, SGLang) support Jamba. At 256k, Jamba's memory advantage shows in concurrent-request throughput.

## Ship It

This lesson produces `outputs/skill-hybrid-picker.md` -- recommends between pure Transformer, Jamba hybrid, and pure SSM based on workload.

## Exercises

1. Run calculator for 32-layer pure Transformer (hidden 4096, 32 heads) and Jamba hybrid at 256k. Verify 8x reduction.
2. Modify calculator for 1:3 and 1:15 ratios. Plot KV cache vs ratio. Find where KV cache equals SSM state.
3. Read Section 3 of Jamba paper. Explain why AI21 chose Mamba-1 over Mamba-2.
4. Compute parameter overhead of MoE-every-other-layer in Jamba 1.5 Large. Compare active ratio to DeepSeek-V3.
5. Read Mamba-3 Section 3. Explain why complex-valued state update is equivalent to data-dependent rotary embedding.

## Further Reading

- Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model" (arXiv:2403.19887)
- AI21, "Jamba 1.5: Hybrid Transformer-Mamba at Scale" (arXiv:2408.12570)
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling" (arXiv:2312.00752)
- Gu, Dao, "Mamba-2" (arXiv:2405.21060)
- Lahoti et al., "Mamba-3" (arXiv:2603.15569, ICLR 2026)
