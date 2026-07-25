# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify

> Voice cloning shipped faster than defenses. 2026 production voice systems need two things: a detector (AASIST, RawNet2) that classifies real vs fake speech, and a watermark (AudioSeal) that survives compression and editing.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## Learning Objectives

- Implement a spectral-feature toy detector for synthetic speech
- Embed and detect an AudioSeal watermark in generated audio
- Evaluate detection performance using Equal Error Rate (EER)
- Design a production defense pipeline with watermark + consent + C2PA provenance

## The Problem

Three related defenses:

1. **Anti-spoofing / deepfake detection.** Given an audio clip, is it synthetic or real?
2. **Audio watermarking.** Embed an imperceptible signal in generated audio that a detector can extract.
3. **Authenticated provenance.** Cryptographic signing of audio files + metadata (C2PA).

## The Concept

### ASVspoof 5 — the 2024-2025 benchmark

Key changes: crowdsourced data, ~2000 speakers, 32 attack algorithms (TTS + VC + adversarial). SOTA: ~7.23% EER.

### AASIST and RawNet2 — detection model families

**AASIST** (2021, updated through 2026). Graph-attention on spectral features. SOTA on ASVspoof 5 countermeasure.

**RawNet2.** Convolutional front-end over raw waveform + TDNN backbone.

**NeXt-TDNN + SSL features.** ECAPA-style + WavLM + focal loss. 0.42% EER on ASVspoof 2019 LA.

### AudioSeal — the 2024 watermark default

Meta's **AudioSeal** (Jan 2024, v0.2 Dec 2024):

- **Localized.** Detects per-frame at 16 kHz sample resolution.
- **Robust.** Survives MP3 / AAC compression, EQ, speed-shift ±10%, noise +10 dB SNR.
- **Fast.** Detector runs at 485× realtime.
- **Capacity.** 16-bit payload per utterance.

### The gap adversaries exploit

From AudioMarkBench: "under pitch shift, all watermarks show Bit Recovery Accuracy below 0.6." **Pitch-shift is the universal attack.** Ship detection alongside watermarking.

### C2PA / Content Authenticity Initiative

Cryptographically signed metadata about creation tool, author, date. Trivially bypassed by re-encoding; always use with watermarking.

## Build It

### Step 1: a simple spectral-feature detector (toy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Synthetic speech often has unusually flat high-frequency energy.

### Step 2: AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
```

### Step 3: evaluation — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4: the production integration

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

## Use It

| Use case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | AudioSeal embed on every output (non-negotiable) |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
| Call-center fraud detection | AASIST on 20% sample of incoming calls |
| Podcast authenticity | C2PA signing on upload, AudioSeal if AI-generated |

## Pitfalls

- **Watermark without detector ever running.** Ship the detector in your CI.
- **Detection without calibration.** AASIST trained on ASVspoof LA overfits; calibrate on your domain.
- **Pitch-shift gap.** Aggressive pitch shift removes most watermarks. Have a detection fallback.
- **Metadata strip-and-rehost.** Always add watermark alongside C2PA.
- **Liveness as detection.** Ask user to say a random phrase. Prevents replay but not real-time cloning.

## Ship It

Save as `outputs/skill-spoof-defender.md`. Pick detection model, watermark, provenance manifest, and operational playbook.

## Exercises

1. **Easy.** Run `code/main.py`. Toy detector + toy watermark embed/detect on synthetic audio.
2. **Medium.** Install `audioseal`, embed a 16-bit payload in a TTS output. Corrupt with noise and measure Bit Recovery Accuracy.
3. **Hard.** Fine-tune RawNet2 or AASIST on ASVspoof 2019 LA. Test on F5-TTS-generated clips — see how OOD detection degrades.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | The benchmark | Biennial challenge; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Classifier: real speech vs synthetic / converted. |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
| AudioSeal | Meta watermark | Localized, 16-bit payload, 485× faster than WavMark. |
| Bit Recovery Accuracy | Watermark survival | Fraction of payload bits recovered after attack. |
| C2PA | Provenance manifest | Cryptographic metadata about creation / authorship. |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking)
