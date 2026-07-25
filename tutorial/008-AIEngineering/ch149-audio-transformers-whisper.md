# Audio Transformers — Whisper Architecture

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
