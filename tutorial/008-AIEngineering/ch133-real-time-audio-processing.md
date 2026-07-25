# Real-Time Audio Processing

> Batch pipelines process a file. Real-time pipelines process the next 20 milliseconds before the next 20 arrive. Every conversational AI, broadcast studio, and telephony bot lives and dies by this latency budget.

**Type:** Build
**Languages:** Python, Rust
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Learning Objectives

- Implement a ring buffer for low-latency audio capture
- Build a VAD gate that gates downstream processing on speech presence
- Design an interruption handler for barge-in during TTS playback
- Budget end-to-end latency for a real-time voice pipeline

## The Problem

Human conversational turn-taking latency is ~230 ms. Anything above 500 ms feels robotic.

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) clocked 200 ms full-duplex. GPT-4o-realtime clocks ~320 ms.

## The Concept

**Frame / chunk / window.** Real-time audio flows as fixed-size blocks. Common choice: 20 ms (320 samples at 16 kHz).

**Ring buffer.** Fixed-size circular buffer. Producer thread writes new frames, consumer thread reads. Size ≈ maximum-latency × sample-rate.

**VAD (Voice Activity Detection).** Gates downstream work when nobody is speaking. Silero VAD 4.0 runs <1 ms per 30 ms frame on CPU.

**Streaming ASR.** Models that emit partial transcripts as audio arrives. Parakeet-CTC-0.6B in streaming mode does 2–5% WER at 320 ms latency.

**Interruption.** When the user speaks while the assistant is talking, you must detect the barge-in, stop TTS, discard remaining LLM output. All within 100 ms.

**WebRTC Opus transport.** 20 ms frames, 48 kHz, adaptive bitrate 8–128 kbps.

**Jitter buffer.** Network packets arrive out of order. 60–80 ms typical.

## Build It

### Step 1: ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

### Step 2: VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Replace with Silero VAD in production:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3: streaming ASR

```python
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4: interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

## Use It

The 2026 stack:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Pitfalls

- **Buffering 500 ms to be safe.** The buffer *is* your latency floor. Shrink it.
- **Not pinning threads.** Audio callback on a low-priority thread = glitches.
- **TTS chunks too small.** Sub-200 ms chunks make vocoder artifacts audible. 320 ms is the sweet spot.
- **No jitter buffer.** Real networks are jittery.
- **Single-shot error handling.** One exception kills the session.

## Ship It

Save as `outputs/skill-realtime-designer.md`. Design a real-time audio pipeline with concrete latency budgets per stage.

## Exercises

1. **Easy.** Run `code/main.py`. Simulates a ring buffer + energy VAD; prints stage latencies for a fake 10-second stream.
2. **Medium.** Using `sounddevice`, build a passthrough loop that processes your mic in 20 ms frames and prints VAD state.
3. **Hard.** Build a full duplex echo test with `aiortc`: browser → WebRTC → Python → WebRTC → browser. Measure glass-to-glass latency.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/11-real-time-audio-processing)
