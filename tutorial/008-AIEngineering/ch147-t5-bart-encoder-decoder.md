# T5, BART — Encoder-Decoder Models

> Encoders understand. Decoders generate. Put them back together and you get a model built for input → output tasks: translate, summarize, rewrite, transcribe.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## The Problem

Decoder-only GPT and encoder-only BERT each strip down the 2017 architecture for a different goal. But many tasks are naturally input-output:

- Translation: English → French.
- Summarization: 5,000-token article → 200-token summary.
- Speech recognition: audio tokens → text tokens.
- Structured extraction: prose → JSON.

For these, encoder-decoder makes the cleanest fit. The encoder produces a dense representation of the source. The decoder generates the output, cross-attending to that representation at every step. Training is shift-by-one on the output side.

Two papers defined the modern playbook:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." Every NLP task reframed as text-in, text-out. Pretrained on masked span prediction.
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." Denoising autoencoder: corrupt input in multiple ways, ask the decoder to reconstruct the original.

## The Concept

### The forward loop

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Crucially, the encoder runs once per input. The decoder runs autoregressively but cross-attends to the *same* encoder output at every step.

### T5 pretraining — span corruption

Pick random spans of the input (average length 3 tokens, 15% total). Replace each span with a unique sentinel: `<extra_id_0>`, `<extra_id_1>`, etc. The decoder outputs only the corrupted spans with their sentinel prefix:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

### BART pretraining — multi-noise denoising

BART tries five noising functions:

1. Token masking.
2. Token deletion.
3. Text infilling (mask a span, decoder inserts the right length).
4. Sentence permutation.
5. Document rotation.

### When to pick each variant in 2026

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | Yes, usually | Clear source sequence; beam search works |
| Speech-to-text | Yes (Whisper) | Input modality differs from output |
| Chat / reasoning | No, decoder-only | No persistent "input" |
| Code completion | Usually no | Decoder-only with long context wins |
| Summarization | Either works | BART beats early decoder-only; modern LLMs match |
| Structured extraction | Either | T5 is clean because "text → text" is general |

## Build It

### Step 1: T5 span corruption

```python
def sentinel(i):
    return f"<extra_id_{i}>"

def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    n = len(tokens)
    n_mask = max(1, int(round(n * mask_rate)))
    n_spans = max(1, int(round(n_mask / mean_span)))
    positions = list(range(n))
    rng.shuffle(positions)
    starts = []
    used = [False] * n
    span_lengths = []
    remaining = n_mask
    for _ in range(n_spans):
        if remaining <= 0:
            break
        random_order = list(range(n))
        rng.shuffle(random_order)
        for start in random_order:
            if used[start]:
                continue
            length = max(1, int(rng.gauss(mean_span, 1.0)))
            length = min(length, remaining, n - start)
            if length < 1:
                continue
            if any(used[i] for i in range(start, start + length)):
                continue
            for i in range(start, start + length):
                used[i] = True
            starts.append(start)
            span_lengths.append(length)
            remaining -= length
            break
    ordered = sorted(zip(starts, span_lengths), key=lambda x: x[0])
    source = []
    target = []
    prev_end = 0
    for idx, (start, length) in enumerate(ordered):
        source.extend(tokens[prev_end:start])
        source.append(sentinel(idx))
        target.append(sentinel(idx))
        target.extend(tokens[start:start + length])
        prev_end = start + length
    source.extend(tokens[prev_end:])
    target.append(sentinel(len(ordered)))
    return source, target
```

### Step 2: verify round-trip

```python
def round_trip(source, target):
    spans = {}
    current_key = None
    current_span = []
    for tok in target:
        if tok.startswith("<extra_id_"):
            if current_key is not None:
                spans[current_key] = current_span
            current_key = tok
            current_span = []
        else:
            current_span.append(tok)
    out = []
    for tok in source:
        if tok.startswith("<extra_id_"):
            out.extend(spans.get(tok, []))
        else:
            out.append(tok)
    return out
```

### Step 3: BART noise functions

```python
def token_mask(tokens, rate=0.15, rng=None, mask_token="<mask>"):
    return [mask_token if rng.random() < rate else t for t in tokens]

def token_delete(tokens, rate=0.15, rng=None):
    return [t for t in tokens if rng.random() >= rate]

def text_infill(tokens, rate=0.15, mean_span=3.0, rng=None, mask_token="<mask>"):
    out = []
    i = 0
    n = len(tokens)
    budget = int(n * rate)
    while i < n:
        if budget > 0 and rng.random() < 0.3:
            span_len = max(1, min(int(rng.gauss(mean_span, 1.0)), budget, n - i))
            out.append(mask_token)
            budget -= span_len
            i += span_len
        else:
            out.append(tokens[i])
            i += 1
    return out

def sentence_permute(sentences, rng=None):
    sents = list(sentences)
    rng.shuffle(sents)
    return sents

def document_rotate(tokens, rng=None):
    if len(tokens) <= 1:
        return tokens
    pivot = rng.randrange(1, len(tokens))
    return tokens[pivot:] + tokens[:pivot]
```

## Use It

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

The T5 trick: the task name goes into the input text. Same model handles dozens of tasks because each task is text-in, text-out.

## Ship It

See `outputs/skill-seq2seq-picker.md`. The skill picks between encoder-decoder and decoder-only for a new task.

## Exercises

1. **Easy.** Apply span corruption to a 30-token sentence, verify that concatenating the non-sentinel source tokens with the decoded target spans reproduces the original.
2. **Medium.** Implement BART's `text_infill` noise: replace random spans with a single `<mask>` token.
3. **Hard.** Fine-tune `flan-t5-small` on a tiny English → pig-Latin corpus. Compare against fine-tuning `Llama-3.2-1B`.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | Two stacks: bidirectional encoder for input, causal decoder with cross-attention |
| Cross-attention | "Where source talks to target" | Decoder's Q × encoder's K/V |
| Span corruption | "T5's pretraining trick" | Replace random spans with sentinel tokens; decoder outputs the spans |
| Denoising objective | "BART's game" | Apply noise function to input, train decoder to reconstruct the clean sequence |
| Sentinel token | "The `<extra_id_N>` placeholder" | Special tokens that tag corrupted spans |
| Flan | "Instruction-tuned T5" | T5 fine-tuned on >1,800 tasks |
| Beam search | "Decoding strategy" | Keep top-k partial sequences at each step |

## Further Reading

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)

---

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder)
