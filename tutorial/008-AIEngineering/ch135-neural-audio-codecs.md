# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split

> 2026 audio generation is almost all tokens. EnCodec, SNAC, Mimi, and DAC turn continuous waveforms into discrete sequences that a transformer can predict. The semantic-vs-acoustic token split is the most important architectural shift since the Transformer for audio.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Learning Objectives

- Implement Residual Vector Quantization (RVQ) from scratch
- Encode and decode audio with EnCodec and Mimi
- Explain why the semantic-acoustic token split (Mimi, SpeechTokenizer) is crucial for generative models
- Measure reconstruction error as a function of bitrate and codebook count

## The Problem

Language models work on discrete tokens. Audio is continuous. If you want an LLM-style model for speech / music — MusicGen, Moshi, Sesame CSM — you first need a **neural audio codec**: a learned encoder that discretizes audio into a small vocabulary of tokens, and a matching decoder that reconstructs the waveform.

Two families:

1. **Reconstruction-first codecs** — EnCodec, DAC. Optimize perceptual audio quality.
2. **Semantic-first codecs** — Mimi (Kyutai), SpeechTokenizer. Force the first codebook to encode linguistic / phonetic content.

The 2024-2026 insight: **a pure reconstruction codec gives you blurry speech when you try to generate from text.** Separating semantic from acoustic is what makes Moshi and Sesame CSM work.

## The Concept

### Residual Vector Quantization (RVQ)

Rather than one big codebook, all modern audio codecs use **RVQ**: a cascade of small codebooks. The first codebook quantizes the encoder output; the second quantizes the residual; etc. Each codebook is 1024 codes. 8 codebooks = effective vocabulary of 1024^8.

### The four codecs that matter in 2026

**EnCodec (Meta, 2022).** The baseline. Encoder-decoder over waveform, RVQ bottleneck. 24 kHz, default 4 codebooks @ 1.5 kbps. Used by MusicGen.

**DAC (Descript, 2023).** RVQ with L2-normalized codebooks, periodic activation functions. Highest reconstruction fidelity. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** Multi-scale RVQ — coarse codebooks at ~12 Hz, fine at 50 Hz. Used by Orpheus-3B.

**Mimi (Kyutai, 2024).** 12.5 Hz frame rate, 8 codebooks @ 4.4 kbps. Codebook 0 is **distilled from WavLM**. Powers Moshi.

### Frame rates matter

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

### Semantic vs acoustic tokens

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

**Semantic token (codebook 0 in Mimi).** Encodes what was said — phonemes, words, content. **Acoustic tokens (codebooks 1-7).** Encode timbre, speaker identity, prosody.

### 2026 reconstruction quality

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

## Build It

### Step 1: toy RVQ from scratch

```python
import math, random

def learn_codebook(values, size, iterations=20):
    rng = random.Random(0)
    lo, hi = min(values), max(values)
    centroids = [lo + (hi - lo) * rng.random() for _ in range(size)]
    for _ in range(iterations):
        buckets = [[] for _ in range(size)]
        for v in values:
            idx = min(range(size), key=lambda i: abs(centroids[i] - v))
            buckets[idx].append(v)
        for i in range(size):
            if buckets[i]:
                centroids[i] = sum(buckets[i]) / len(buckets[i])
    return sorted(centroids)

def quantize_with_codebook(values, codebook):
    indices, residuals = [], []
    for v in values:
        idx = min(range(len(codebook)), key=lambda i: abs(codebook[i] - v))
        indices.append(idx)
        residuals.append(v - codebook[idx])
    return indices, residuals

def rvq_encode(values, codebook_size=8, n_codebooks=4):
    residuals = list(values)
    codebooks, all_indices = [], []
    for cb_i in range(n_codebooks):
        cb = learn_codebook(residuals, codebook_size, seed=cb_i)
        codebooks.append(cb)
        indices, residuals = quantize_with_codebook(residuals, cb)
        all_indices.append(indices)
    return all_indices, codebooks
```

### Step 2: encode with EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

### Step 3: the semantic-acoustic split (Mimi-style)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

### Step 4: AR LM over codec tokens

For a 10 s speech clip at Mimi's 12.5 Hz × 8 codebooks:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens is a trivial context for a transformer. A 256M-parameter transformer can generate 10 seconds of speech in milliseconds.

## Use It

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC or Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library | EnCodec + T5 condition |

Rule of thumb: **if you're building a generative model, start with Mimi or SNAC.**

## Pitfalls

- **Too many codebooks.** Stop at 8-12; diminishing returns.
- **Frame-rate mismatch.** Training LM on 12.5 Hz Mimi then fine-tuning on 50 Hz EnCodec fails silently.
- **Assuming all codebooks equal.** Codebook 0 carries content; losing it destroys intelligibility.
- **Using reconstruction quality as the only metric.** A codec with great reconstruction may be useless for LM-based generation.

## Ship It

Save as `outputs/skill-codec-picker.md`. Pick a codec for a given generative or compression task.

## Exercises

1. **Easy.** Run `code/main.py`. Implements a toy scalar + residual quantizer and measures reconstruction error as you add codebooks.
2. **Medium.** Install `encodec` and compare 1, 4, 8, 32 codebooks on a held-out speech clip.
3. **Hard.** Load Mimi. Encode a clip. Replace codebook 0 with random integers; decode. Then replace codebook 7 similarly. Codebook 0 corruption should destroy intelligibility.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | Cascade of small codebooks; each quantizes the previous residual. |
| Frame rate | Codec speed | How many token-frames per second. Lower = faster LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook distilled from SSL features; encodes content. |
| Acoustic codebooks | Everything else | Timbre, prosody, noise, fine detail. |
| PESQ / ViSQOL | Perceptual quality | Objective metrics correlating with MOS. |
| EnCodec | Meta codec | The RVQ baseline; used by MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/13-neural-audio-codecs)
