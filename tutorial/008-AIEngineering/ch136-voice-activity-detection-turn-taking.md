# Voice Activity Detection & Turn-Taking — Silero, Cobra, and the Flush Trick

> Every voice agent lives or dies on two decisions: is the user speaking now, and are they done? VAD answers the first. Turn-detection (VAD + silence-hangover + semantic endpoint model) answers the second.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## Learning Objectives

- Implement a three-tier VAD cascade: energy gate, Silero, semantic turn detector
- Build a turn-detection state machine with configurable silence hangover
- Understand how the Kyutai flush trick reduces conversational latency
- Calibrate VAD threshold, minimum speech duration, and pre-roll buffer

## The Problem

Three distinct decisions a voice agent makes on every 20 ms chunk:

1. **Is this frame speech?** — VAD.
2. **Has the user started a new utterance?** — onset detection.
3. **Has the user finished?** — end-pointing (turn-end).

## The Concept

### The three-tier VAD cascade

**Tier 1: energy gate.** Cheapest. Threshold RMS at -40 dBFS.

**Tier 2: Silero VAD** (2020-2026, MIT). 1M parameters. Trained on 6000+ languages. ~1 ms per 30 ms chunk. 87.7% TPR at 5% FPR.

**Tier 3: semantic turn detector.** LiveKit's turn-detection model or your own small classifier. Distinguishes "pause mid-sentence" from "done talking."

### Key parameters

- **Threshold.** Silero > 0.5 (default) or > 0.3 (sensitive).
- **Minimum speech duration.** Reject < 250 ms (coughs, chair noise).
- **Silence hangover.** Wait 500-800 ms before declaring end-of-turn.
- **Pre-roll buffer.** Keep 300-500 ms before VAD fires. Prevents "hey" clipping.

### The flush trick (Kyutai 2025)

Streaming STT models have a look-ahead delay. When VAD fires end-of-speech, send a **flush signal** to the STT that forces immediate output. End-to-end: 125 ms VAD + flush STT = conversational latency.

### 2026 VAD comparison

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | ~95% | ~10 ms | MIT-ish |

Silero is the right default. Cobra is the compliance upgrade.

## Build It

### Step 1: the energy gate

```python
import math

def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2: Silero VAD in Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3: turn-end state machine

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4: the flush trick skeleton

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT must support flush. Whisper streaming does not; Kyutai STT and Deepgram do.

## Use It

| Situation | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (phone) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |

## Pitfalls

- **Fixed threshold.** Works in quiet, fails in noisy. Calibrate on-device.
- **Too-short silence hangover.** Agent interrupts mid-sentence. 500-800 ms.
- **Too-long hangover.** Feels sluggish. A/B test.
- **No pre-roll buffer.** First 200-300 ms of user audio lost.
- **Ignoring semantic endpointing.** "Hmm, let me think..." contains long pauses.

## Ship It

Save as `outputs/skill-vad-tuner.md`. Pick VAD model, threshold, hangover, pre-roll, and turn-detection strategy.

## Exercises

1. **Easy.** Run `code/main.py`. Simulates a speech + silence + speech + coughs sequence and tests three VAD tiers.
2. **Medium.** Install `silero-vad`, process a 5-min recording, tune threshold to minimize both first-word clips and false triggers.
3. **Hard.** Build a mini turn-detector: Silero VAD + a 3-layer MLP on the last 10 words' embeddings. Beat Silero-only by 10% F1.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | Voice detector | Binary per-frame: is this speech? |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Time to wait before declaring turn end; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Keep 300-500 ms audio before VAD fires. |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms instead of 500 ms. |
| Semantic endpoint | "Did they mean to stop?" | ML classifier that looks at words, not just silence. |
| TPR @ FPR 5% | ROC point | Standard VAD benchmark. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/14-voice-activity-detection-turn-taking)
