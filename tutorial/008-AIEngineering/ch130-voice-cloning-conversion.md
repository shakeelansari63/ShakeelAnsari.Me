# Voice Cloning & Voice Conversion

> Voice cloning reads your text in someone else's voice. Voice conversion rewrites your voice into someone else's while preserving what you said. Both hang on the same decomposition: separate speaker identity from content.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Learning Objectives

- Perform zero-shot voice cloning with F5-TTS using a 5-second reference
- Perform voice conversion using KNN-VC with a target speaker pool
- Embed and detect an AudioSeal watermark in generated speech
- Implement a consent gate that verifies signed authorization before cloning

## The Problem

In 2026, a 5-second audio clip is enough to produce a high-quality clone of anyone's voice with a consumer GPU. Two closely-related tasks:

- **Voice cloning (TTS-side):** text + 5-second reference voice → audio in that voice.
- **Voice conversion (speech-side):** source audio (person A saying X) + reference voice of person B → audio of B saying X.

Key constraint in 2026: **watermarking and consent gates are legally required in the EU (AI Act, enforceable August 2026) and in California (AB 2905, effective 2025)**.

## The Concept

**Zero-shot cloning.** Pass a 5-second clip to a model trained on thousands of speakers. The speaker encoder maps the clip to a speaker embedding; the TTS decoder conditions on that embedding plus text.

**Few-shot fine-tuning.** Record 5-30 minutes of the target voice. LoRA-fine-tune a base model for an hour.

**Voice conversion (VC).** Two families:

- **Recognition-synthesis.** Run ASR-like model to extract content representation (PPGs), then resynthesize with target speaker embedding.
- **Disentanglement.** Train an autoencoder that separates content, speaker, and prosody in latent space.

**Neural codec-based cloning (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — treat audio as discrete tokens from EnCodec, train a large AR or flow-matching model.

### The ethics bit

**Watermarking.** AudioSeal embeds a ~16-bit ID imperceptibly. Survives re-encoding, streaming, and common edits.

**Consent gates.** Must pair every cloned output with a verifiable consent record.

**Detection.** AASIST, RawNet2 ship as detectors. ASVspoof 2025 challenge published EERs of 0.8–2.3% against ElevenLabs, VALL-E 2, and Bark outputs.

### Numbers (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 is generally indistinguishable from the target for most listeners.

## Build It

### Step 1: zero-shot clone with F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Reference transcript must exactly match the audio.

### Step 2: voice conversion with KNN-VC

```python
from knnvc import KNNVC
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC runs WavLM to extract per-frame embeddings, then replaces each source frame with its nearest neighbor in the pool.

### Step 3: embed a watermark

```python
from silentcipher import SilentCipher

sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)
```

~32 bits of payload, detectable after MP3 re-encode.

### Step 4: consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Use It

| Situation | Pick |
|-----------|------|
| 5-sec zero-shot clone, open-source | F5-TTS or OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (rewriting) | KNN-VC or Diff-HierVC |
| Deepfake detection | Wav2Vec2-AASIST |

## Pitfalls

- **Misaligned reference transcript.** F5-TTS requires reference text to match reference audio exactly.
- **Reverberant reference.** Record dry, close-mic.
- **Emotional mismatch.** Training reference "cheerful" produces cheerful clones of everything.
- **No watermark.** Legally unshippable in EU from Aug 2026.

## Ship It

Save as `outputs/skill-voice-cloner.md`. Design a cloning or conversion pipeline with consent gate + watermark + quality target.

## Exercises

1. **Easy.** Run `code/main.py`. Demonstrates the speaker-embedding swap by computing the cosine between two "speakers" pre and post swap.
2. **Medium.** Use OpenVoice v2 to clone your own voice. Measure SECS between reference and clone. Measure CER via Whisper.
3. **Hard.** Apply SilentCipher watermark to 20 clones, run them through 128 kbps MP3 encode+decode, detect the payload. Report bit-accuracy.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 seconds is enough | Pretrained model + speaker embedding; no training. |
| PPG | Phonetic posteriorgram | Per-frame ASR posteriors used as language-agnostic content rep. |
| KNN-VC | Nearest-neighbor conversion | Replace each source frame with nearest target-pool frame. |
| Neural codec TTS | VALL-E style | AR model over EnCodec/SoundStream tokens. |
| Watermark | Inaudible signature | Bits embedded in audio, survive re-encode. |
| SECS | Cloning fidelity | Cosine between target and clone speaker embeddings. |
| AASIST | Deepfake detector | Anti-spoof model; detects synthesized speech. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/08-voice-cloning-conversion)
