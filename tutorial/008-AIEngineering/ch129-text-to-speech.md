# Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro

> ASR inverts speech to text; TTS inverts text to speech. The 2026 stack is three parts: text → tokens, tokens → mel, mel → waveform. Each part has a default model that fits in a laptop.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Learning Objectives

- Run Kokoro-82M for real-time English TTS on CPU
- Run F5-TTS for zero-shot voice cloning from a 5-second reference
- Phonemize input text using the phonemizer library
- Evaluate TTS quality using UTMOS, CER round-trip, and SECS

## The Problem

You have a string: "Please remind me to water the plants at 6 pm." You need a 3-second audio clip that sounds natural, has correct prosody, and runs in under 300 ms on a CPU.

Modern TTS pipelines:

1. **Text frontend.** Normalize text, convert to phonemes or subword tokens.
2. **Acoustic model.** Text → mel spectrogram. Tacotron 2, FastSpeech 2, VITS, F5-TTS, Kokoro.
3. **Vocoder.** Mel → waveform. WaveNet, HiFi-GAN, BigVGAN, neural codec vocoders.

## The Concept

**Tacotron 2 (2017).** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder emits mel frames. Slow, wobbly on long text.

**FastSpeech 2 (2020).** Non-autoregressive. Duration predictor outputs how many mel frames each phoneme gets. 1-pass, 10× faster than Tacotron.

**VITS (2021).** Jointly trains encoder + flow-based duration + HiFi-GAN vocoder end-to-end with variational inference. High quality, single model.

**F5-TTS (2024).** Diffusion transformer over flow matching. Natural prosody, zero-shot voice cloning with 5 seconds of reference audio. 335M params.

**Kokoro (2024).** Small (82M), CPU-runnable, best-in-class English TTS for real-time use.

### Vocoder evolution

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated | discrete tokens, bit-efficient |

### Evaluation

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |

## Build It

### Step 1: phonemize input

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes are the universal bridge.

### Step 2: run Kokoro (2026 CPU default)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

### Step 3: run F5-TTS with voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

### Step 4: the full pipeline

```python
import soundfile as sf

text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
sf.write("out.wav", wav, 24000)
```

## Use It

The 2026 stack:

| Situation | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
| Voice cloning from 5 s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
| Low-resource language | Train VITS on 5–20 h target-lang data |

Open-source leader as of 2026: **F5-TTS for quality, Kokoro for efficiency**.

## Pitfalls

- **No text normalizer.** "Dr. Smith" reads as "Doctor" or "Drive"? Normalize BEFORE phonemizer.
- **OOV proper nouns.** Ship a fallback grapheme-to-phoneme model for unknown tokens.
- **Clipping.** Always `np.clip(wav, -1, 1)`.
- **Sample-rate mismatch.** Kokoro outputs 24 kHz; downstream may expect 16 kHz.

## Ship It

Save as `outputs/skill-tts-designer.md`. Design a TTS pipeline for a given voice, latency, and language target.

## Exercises

1. **Easy.** Run `code/main.py`. Builds a phoneme dictionary from a toy vocab, estimates duration per phoneme, and prints a fake "mel" schedule.
2. **Medium.** Install Kokoro, synthesize the same sentence at voice `af_bella` and `am_adam`. Compare audio durations and subjective quality.
3. **Hard.** Record a 5-second reference clip of yourself. Use F5-TTS to clone it. Report SECS between reference and cloned output.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Sound unit | Abstract sound class; 39 in English (ARPABet). |
| Duration predictor | How long each phoneme lasts | Non-AR model output; integer frames per phoneme. |
| Vocoder | Mel → waveform | Neural net mapping mel-spec to raw samples. |
| HiFi-GAN | Standard vocoder | GAN-based; dominant 2020–2024. |
| MOS | Subjective quality | 1–5 mean opinion score from human raters. |
| SECS | Voice-clone metric | Cosine similarity between target and output speaker embedding. |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/07-text-to-speech)
