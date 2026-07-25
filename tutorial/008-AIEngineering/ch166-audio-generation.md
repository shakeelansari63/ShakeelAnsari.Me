# Audio Generation

> Audio is a 1-D signal at 16-48 kHz. A five-second clip is 80-240k samples. No transformer attends to that sequence directly. The solution for every production audio model in 2026 is the same: a neural codec (Encodec, SoundStream, DAC) compresses audio to discrete tokens at 50-75 Hz, and a transformer or diffusion model generates tokens.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## The Problem

Three audio generation tasks:

1. **Text-to-speech.** Given text, produce speech. Clean speech is narrow-band and has strong phonetic structure — solved well by transformer-over-tokens.
2. **Music generation.** Given a prompt (text, melody, chord progression, genre), produce music. Much broader distribution.
3. **Audio effects / sound design.** Given a prompt, produce ambient sound or Foley.

All three run on the same substrate: neural audio codec + token-AR or diffusion generator.

## The Concept

```mermaid
graph LR
    A[Waveform<br>16-48 kHz] --> B[Encoder Conv]
    B --> C[RVQ Layer 1]
    C --> D[RVQ Layer 2]
    D --> E[...]
    E --> F[Tokens at 50-75 Hz]
    F --> G[Generator]
    G --> H[Token Sequence]
    H --> I[Decoder Conv]
    I --> J[Output Waveform]
    G -- Option 1 --> K[Token-AR Transformer]
    G -- Option 2 --> L[Latent Diffusion / Flow]
```

### Neural Audio Codecs

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). A convolutional encoder compresses waveform to a per-timestep vector; residual vector quantization (RVQ) converts each vector to a cascade of K codebook indices.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Two Generative Paradigms on Top

**Token-autoregressive.** Flatten RVQ tokens into a sequence, run a decoder-only transformer. MusicGen uses "delayed parallel" to emit K codebook streams in parallel with per-stream offsets. VALL-E generates speech tokens from a text prompt + 3-second voice sample.

**Latent diffusion.** Pack codec tokens as continuous latents or model them with categorical diffusion. Stable Audio 2.5 uses flow matching on continuous audio latents. AudioLDM 2 uses text-to-mel-to-audio diffusion.

The 2024-2026 trend: flow matching is winning for music (faster inference, cleaner samples) while token-AR still dominates speech because it is naturally causal and streams well.

## Production Landscape

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |

## Build It

`code/main.py` simulates the core idea: train a tiny next-token transformer on synthetic "audio token" sequences generated from two distinct "styles".

### Step 1: synthetic audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: train a tiny token predictor

A bigram-style predictor conditioned on style. The point is the pattern: codec tokens → cross-entropy training → autoregressive sampling.

```python
def init_counts():
    return [[[1.0 for _ in range(VOCAB)] for _ in range(VOCAB)] for _ in range(NUM_STYLES)]

def update_counts(counts, sequence, style):
    for i in range(len(sequence) - 1):
        counts[style][sequence[i]][sequence[i + 1]] += 1.0
```

### Step 3: sample conditionally

Given the style token and a starting token, sample the next token from the predicted distribution.

```python
def generate(counts, style, start, length, rng, temperature=1.0):
    out = [start]
    for _ in range(length - 1):
        p = probs(counts, style, out[-1])
        if temperature != 1.0:
            p = [pi ** (1 / temperature) for pi in p]
            total = sum(p)
            p = [x / total for x in p]
        out.append(sample_from(p, rng))
    return out
```

## Pitfalls

- **Codec quality caps output quality.** If the codec can't represent a sound faithfully, no amount of generator quality helps. DAC is the current open best.
- **RVQ error accumulation.** Each RVQ layer models the residual of the previous. Errors on layer 1 propagate.
- **Musical structure.** 30 seconds of tokens is 20k+ tokens at 75 Hz. Hard for transformers.
- **Clean-data appetite.** Music generators need tens of thousands of hours of licensed music.
- **Voice cloning ethics.** A 3-second sample plus a text prompt is enough for VALL-E / XTTS / ElevenLabs to clone a voice.

## Use It — 2026 Stack

| Task | 2026 stack |
|------|------------|
| Commercial TTS | ElevenLabs, OpenAI TTS, or Azure Neural |
| Voice cloning (consent-verified) | XTTS v2 (open) or ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API, Suno, or Udio |
| Music with lyrics | Suno v4 or Udio v1.5 |
| Sound effects / Foley | AudioCraft 2, ElevenLabs SFX, or Stable Audio Open |
| Real-time voice agent | GPT-4o realtime or Gemini Live |
| Open-weights music research | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |

## Production Note

Audio is the one output modality users expect to arrive *as it is generated*, not all-at-once. For 16kHz audio tokenized at ~75 tokens/second (Encodec), the server must generate ≥75 tokens/sec per user to keep playback smooth.

- **Flow-matching audio models cannot stream trivially.** Stable Audio 2.5 and AudioCraft 2 render a fixed clip length in one pass. To stream, you chunk the clip and overlap boundaries, adding 100-300ms of latency overhead vs a codec AR model.
- If the product is "live voice chat" or "real-time music continuation", pick the codec AR path. If it is "render a 30-second clip on submit", flow-matching wins on quality and total latency.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Codec | "Neural compression" | Encoder / decoder for audio; typical output is 50-75 Hz tokens. |
| RVQ | "Residual VQ" | Cascade of K quantizers; each models the residual of the previous. |
| Token | "One codec symbol" | Discrete index into a codebook; 1024 or 2048 typical. |
| Delayed parallel | "Offset codebooks" | Emit K token streams with staggered offsets to reduce sequence length. |
| Flow matching | "The 2024 win for audio" | Straighter-path alternative to diffusion; faster sampling. |
| Voice prompt | "3-second sample" | Speaker embedding or token prefix that steers the cloned voice. |
| Mel spectrogram | "The visual" | Log-magnitude perceptual spectrogram; used by many TTS systems. |
| Vocoder | "Mel to wave" | Neural component that converts mel spectrograms back to audio. |

## Exercises

1. **Easy.** Run `code/main.py` and set style explicitly. Verify the generated sequences match the style's pattern.
2. **Medium.** Add delayed parallel decoding: simulate 2 streams of tokens that must stay offset by 1 step.
3. **Hard.** Use HuggingFace transformers to run MusicGen-small locally. Generate a 10-second clip with three different prompts; A/B for style adherence.

## Further Reading

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312)
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546)
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111)
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284)
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/08-generative-ai/11-audio-generation)
