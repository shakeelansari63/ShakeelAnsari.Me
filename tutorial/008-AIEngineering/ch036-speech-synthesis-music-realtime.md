# TTS, Voice Cloning, Music & Real-Time Audio

> Combined lessons (9 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch129): Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro

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

---

## Part 2 (ch130): Voice Cloning & Voice Conversion

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

---

## Part 3 (ch131): Music Generation — MusicGen, Stable Audio, Suno, and the Licensing Earthquake

> 2026 music generation: Suno v5 and Udio v4 dominate commercial; MusicGen, Stable Audio Open, and ACE-Step lead open-source. The technical problem is mostly solved. The legal problem (Warner Music $500M settlement, UMG settlement) reshaped the field in 2025-2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## Learning Objectives

- Generate instrumental music with MusicGen from text prompts and melody conditioning
- Evaluate music generation quality using Frechet Audio Distance (FAD)
- Understand the 2026 legal landscape for AI music generation
- Design a safe-to-ship music generation pipeline with appropriate licensing

## The Problem

Text → a 30-second to 4-minute music clip, with lyrics, vocals, and structure. Three sub-problems:

1. **Instrumental generation.** Text like "lo-fi hip-hop drums with warm keys" → audio.
2. **Song generation (with vocals + lyrics).** "Country song about rainy Texas nights" → full song.
3. **Conditional / controllable.** Extend an existing clip, regenerate a bridge, swap genre, stem-separate.

## The Concept

### Token LM over neural-codec tokens

Meta's **MusicGen** (2023, MIT): condition on text/melody embeddings, autoregressively predict EnCodec tokens (32 kHz, 4 codebooks), decode with EnCodec. 300M - 3.3B params.

**ACE-Step** (open-source, 4B XL released April 2026) extends this for full-song lyric-conditioned generation.

### Diffusion over mels or latents

**Stable Audio (2023)** and **Stable Audio Open (2024)**: latent diffusion on compressed audio. Excels at loops, sound design, ambient textures.

### Hybrid (production) — Suno, Udio, Lyria

Closed weights. Likely AR codec LM + diffusion-based vocoder with specialized voice / drum / melody heads.

### Evaluation

- **FAD (Fréchet Audio Distance).** Embedding-level distance between generated vs real audio using VGGish or PANNs features.
- **Musicality (subjective).** Human preference.
- **CLAP score.** Text-audio alignment.

### 2026 model map

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | > 2 min | yes | Apache-2.0 |
| YuE | 7B | > 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |

### The legal landscape (2025-2026)

- **Warner Music vs Suno settlement.** $500M.
- **EU AI Act** + **California SB 942**: AI-generated music must be disclosed.

Safe-to-ship patterns:

1. Generate instrumental only (MusicGen, Stable Audio Open).
2. Use commercial APIs with per-generation license.
3. Train on owned or licensed catalog.
4. Tag generations with watermarks + metadata.

## Build It

### Step 1: generate with MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

### Step 2: melody conditioning

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody takes a chromagram and preserves the tune while swapping timbre.

### Step 3: FAD evaluation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 4: LLM + music workflow

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Use It

| Goal | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full songs with vocals (commercial) | Suno v5 or Udio v4 with explicit license |
| Full songs with vocals (open) | ACE-Step XL or YuE |
| Short ad jingle | MusicGen melody-conditioned on hummed reference |

## Pitfalls

- **Copyright-laundering prompts.** "In the style of Taylor Swift" — add your own filter list.
- **Repetition / drift past 30 s.** Crossfade multiple generations.
- **Tempo drift.** Use BPM tags and post-filter with `librosa.beat_track`.
- **Vocal intelligibility.** Open models are mushy on words. Use commercial API if lyrics matter.
- **Mono output.** Upgrade with stereo reconstruction.

## Ship It

Save as `outputs/skill-music-designer.md`. Pick model, license strategy, length / structure plan, and disclosure metadata.

## Exercises

1. **Easy.** Run `code/main.py`. It produces a "generative" chord progression + drum pattern as ASCII symbols.
2. **Medium.** Install `audiocraft`, generate 10-second clips across 4 genre prompts with MusicGen-small, measure FAD against a reference genre set.
3. **Hard.** Using ACE-Step (or MusicGen-melody), generate three variations of the same tune with different timbre prompts. Compute CLAP similarity to verify alignment.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | Audio FID | Fréchet distance between embedding distributions of real vs generated. |
| Chromagram | Melody as pitches | 12-dim per-frame vector; input to melody conditioning. |
| Stems | Instrument tracks | Separated bass / drums / vocals / melody as WAV. |
| Inpainting | Regen a section | Mask a time window; model regenerates just that. |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding. |
| EnCodec | Music codec | Meta's neural codec used by MusicGen; 32 kHz, 4 codebooks. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/09-music-generation)

---

## Part 4 (ch132): Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 2026 audio-language models reason over speech + environmental sound + music. Qwen2.5-Omni-7B matches GPT-4o Audio on MMAU-Pro. Audio Flamingo Next beats Gemini 2.5 Pro on LongAudioBench. The gap between open and closed is essentially closed — except on multi-audio tasks.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## Learning Objectives

- Query Qwen2.5-Omni-7B for audio understanding and reasoning
- Implement the projector pattern that bridges an audio encoder to an LLM
- Benchmark a model on MMAU-Pro and interpret per-category accuracy
- Understand where LALMs fail and why multi-audio tasks are unsolved

## The Problem

You have 5 seconds of audio: dog barks, someone yells "stop!", then silence. Useful questions span multiple axes — transcription, semantic reasoning, music reasoning, long-audio retrieval. A single model that answers all of these with one prompt is an **audio-language model** (LALM / ALM).

## The Concept

### The three-component template

Every 2026 LALM has the same skeleton:

1. **Audio encoder.** Whisper encoder · BEATs · CLAP · WavLM · or a custom encoder.
2. **Projector.** Linear or MLP bridging audio-encoder features into the LLM's token embedding space.
3. **LLM.** Llama / Qwen / Gemma-based decoder. Takes interleaved text + audio tokens; generates text.

Training:

- **Stage 1.** Freeze encoder + LLM; train projector only on ASR / captioning data.
- **Stage 2.** Full / LoRA fine-tune on instruction-following audio tasks.
- **Stage 3 (optional).** Voice-in / voice-out adds a speech decoder.

### The 2026 model map

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark reality check (2026)

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |

The **multi-audio column is damning for everyone.** Random chance on 4-option multiple choice = 25%.

## Build It

### Step 1: query Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2: the projector pattern

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

The projector is usually 1-3 linear layers.

### Step 3: benchmarking MMAU-Pro

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Report per-category separately.

## Use It

| Task | 2026 pick |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni or GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 |
| Call-center audit | Gemini 2.5 Pro via API |

## Pitfalls

- **Over-trust on multi-audio.** Random-chance-level performance is real.
- **Long-audio degradation.** Past 10 minutes, speaker attribution breaks.
- **Hallucinations on silence.** Same Whisper-style issue. VAD-gate.
- **Benchmark cherry-picking.** Run MMAU-Pro multi-audio subset yourself.

## Ship It

Save as `outputs/skill-alm-picker.md`. Pick LALM + benchmark subset + output-modality for a given task.

## Exercises

1. **Easy.** Run `code/main.py` to see a toy projector pattern + fake LALM routing.
2. **Medium.** Score Qwen2.5-Omni-7B on 100 MMAU-Pro speech items. Compare to the paper's reported number.
3. **Hard.** Build a minimal audio-captioning baseline: BEATs encoder + 2-layer projector + frozen Llama-3.2-1B. Fine-tune only the projector on AudioCaps.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Small MLP mapping audio features into LLM embedding space. |
| MMAU | The benchmark | 10k audio-QA pairs across speech, sound, music. |
| MMAU-Pro | Harder MMAU | 1800 multi-audio / reasoning-heavy questions. |
| LongAudioBench | Long-form eval | Multi-minute clips with semantic queries. |
| Voice-in / voice-out | Speech-native | Model ingests speech and emits speech without text detour. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/10-audio-language-models)

---

## Part 5 (ch133): Real-Time Audio Processing

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

---

## Part 6 (ch134): Build a Voice Assistant Pipeline — The Phase 6 Capstone

> Everything from lessons 01-11, stitched together. Build a voice assistant that listens, reasons, and talks back. In 2026 that is a solved engineering problem, not a research problem — but the integration details decide whether it ships.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## Learning Objectives

- Wire VAD-gated turn capture with a ring buffer and pre-roll
- Connect streaming STT → LLM with tool calling → streaming TTS
- Implement barge-in interruption handling
- Measure end-to-end latency and identify bottlenecks

## The Problem

Build an end-to-end assistant:

1. Captures mic input (16 kHz mono).
2. Detects start/end of user speech.
3. Transcribes streaming.
4. Passes transcript to an LLM that can call tools.
5. Streams LLM text to a TTS.
6. Plays audio back to the user.
7. Stops if the user interrupts.

Latency target: first TTS audio byte within 800 ms of the user finishing their utterance.

## The Concept

![Voice assistant pipeline](https://mermaid.ink/svg/pako:eNp1UstugzAQ_BXL5x4q8QNcKkWqWvXQHtqj5cDFLq4FY9k4ShTl32toeKSpNwfW7MyOdo-Maw2SMN1zLRV9D4MS9K9BvUo2VnqjvFeoG1tLNrHWCjZBcmqYM_4Ucf5TQsr4HwXnkZT3CzE1xrlBsYFJqyHrLLMB7DXUvE6cMw-mhT0h3fcs7w9nY8hLhPQeQfo3MhkDakCWsFd5uIO8Hw7eLBhx-4RqG8F8QkGixZ4Vl6j8Wz47PEDKj3sxOabv06TLXH7MQ3PYnujIS3ux0SNEGqR_b3mK9w5P_vjF5XQH3tWPN2JgGg0)

### The seven components

1. **Audio capture.** Mic → 16 kHz mono → 20 ms chunks.
2. **VAD.** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang-over 500 ms.
3. **Streaming STT.** Whisper-streaming, Parakeet-TDT, or Deepgram Nova-3.
4. **LLM with tool calling.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash.
5. **Streaming TTS.** Kokoro-82M or Cartesia Sonic.
6. **Playback.** Speaker out.
7. **Interruption handler.** If VAD fires during TTS playback, stop everything.

### The three failure modes

1. **First-word clip.** VAD starts too late. Start threshold at 0.3, not 0.5.
2. **Mid-response interrupt confusion.** Wire VAD → cancel-LLM.
3. **Silence hallucination.** Always VAD-gate.

### 2026 production reference stacks

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | Industry default |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; lesson 15 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## Build It

### Step 1: mic capture with chunking

```python
import sounddevice as sd
import queue

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2: VAD-gated turn capture

```python
import collections

def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4: tool calling

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5: interruption handling

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Use It

See `code/main.py` for a runnable simulation. Replace stubs with:

- `silero-vad` for VAD
- `deepgram-sdk` or `openai-whisper` for STT
- `openai` (`gpt-4o`) or `anthropic` for LLM
- `kokoro` or `cartesia` for TTS
- `sounddevice` for I/O

## Pitfalls

- **Logging PII forever.** Full-turn audio is PII. 30-day retention, encrypted at rest.
- **No barge-in.** Users will interrupt. Your assistant must stop talking.
- **TTS that blocks.** Use async or a separate thread.
- **No tool-call error handling.** Return error + retry once, then gracefully degrade.
- **No wake-word option.** Add a wake-word gate (Porcupine or openWakeWord).

## Ship It

Save as `outputs/skill-voice-assistant-architect.md`. Given budget + scale + language + compliance constraints, produce a full stack spec.

## Exercises

1. **Easy.** Run `code/main.py`. It simulates one full turn end-to-end with stub modules and prints per-stage latency.
2. **Medium.** Replace the STT stub with a real Whisper model on a pre-recorded `.wav`. Measure WER and end-to-end latency.
3. **Hard.** Add tool calling: implement `get_weather` and `set_timer`. Verify that when the user says "set a 5 minute timer" the right function fires.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | A user + assistant round-trip | One VAD-bounded user speech + one LLM-TTS response. |
| Barge-in | Interruption | User speaks while assistant talks; assistant stops. |
| Wake word | "Hey assistant" | Short keyword detector. |
| End-pointing | Turn ending | VAD + min-silence decision that user has finished. |
| Pre-roll | Pre-speech buffer | Keep 200-400 ms of audio before VAD fires. |
| Tool call | Function invocation | LLM emits JSON; runtime dispatches; result feeds back in-loop. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/12-voice-assistant-pipeline)

---

## Part 7 (ch135): Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split

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

---

## Part 8 (ch137): Streaming Speech-to-Speech — Moshi, Hibiki, and Full-Duplex Dialogue

> 2024-2026 redefined voice AI. Moshi ships a single model that listens and speaks simultaneously at 200 ms latency. Hibiki does speech-to-speech translation chunk-by-chunk. Both abandon the ASR → LLM → TTS pipeline for a unified full-duplex architecture over Mimi codec tokens.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Learning Objectives

- Explain the Moshi full-duplex architecture and its parallel Mimi streams
- Simulate the two-stream + inner-monologue loop
- Understand why the semantic-acoustic codec split enables full-duplex
- Compare Moshi vs pipelined voice assistants for latency and tool-calling

## The Problem

Every voice agent built from Lessons 11 + 12 has a fundamental latency floor around 300-500 ms. Moshi asks a different question: what if there is no pipeline? What if one model takes audio in and emits audio out directly, continuously?

The answer is **full-duplex speech-to-speech**. Theoretical latency 160 ms. Practical latency 200 ms on a single L4 GPU.

## The Concept

### The Moshi architecture

**Inputs.** Two Mimi codec streams, both at 12.5 Hz × 8 codebooks:

- Stream 1: user audio (Mimi-encoded, constantly arriving)
- Stream 2: Moshi's own audio (generated by Moshi)

**The transformer.** A 7B-parameter Temporal Transformer processes both streams and a text "inner monologue" stream. At each 80 ms step, it:

1. Consumes the latest user Mimi tokens (8 codebooks).
2. Consumes the most recent Moshi Mimi tokens (8 codebooks, as produced).
3. Generates the next Moshi text token (inner monologue).
4. Generates the next Moshi Mimi tokens (8 codebooks via a small Depth Transformer).

All three streams run in parallel. Moshi can hear the user while speaking, interrupt itself, and back-channel.

**The depth transformer.** Within a frame, the 8 codebooks are predicted sequentially by a small 2-layer transformer.

### Why inner-monologue text helps

Without explicit text, the model has to implicitly model language in its acoustic stream. Moshi forces it to emit text tokens alongside audio. The text stream is essentially the transcript of what Moshi is saying.

### Hibiki: streaming speech-to-speech translation

Same architecture, trained on translation pairs. Source audio in, target-language audio out, continuously. Hibiki-Zero (Feb 2026) eliminates the need for word-level aligned training data using GRPO reinforcement learning.

### The broader Kyutai stack (2026)

- **Moshi** — full-duplex dialogue (French first, English well-supported)
- **Hibiki / Hibiki-Zero** — simultaneous speech translation
- **Kyutai STT** — streaming ASR
- **Kyutai Pocket TTS** — 100M-param TTS runs on CPU

### 2026 performance numbers

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |

## Build It

### Step 1: the interface

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Step 2: the full-duplex loop

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

### Step 3: the training objective (conceptual)

For every 80 ms frame `t`:

- Input: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Predict: `moshi_text[t]`, then `moshi_mimi[t, codebook_0..7]`

### Step 4: where Moshi wins and where it doesn't

Moshi wins: sub-250 ms end-to-end, natural back-channels, no pipeline glue.

Moshi does not win: tool calling, long reasoning, factual accuracy on niche topics, enterprise agent use cases.

## Use It

| Situation | Pick |
|-----------|------|
| Lowest-latency voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent with tools | Pipeline (Lesson 12), not Moshi |
| Custom-voice TTS in context | Sesame CSM |

## Pitfalls

- **Limited tool calling.** Combine with pipeline for tools.
- **Specific-voice conditioning.** Cloning is a separate training run.
- **Language coverage.** French + English excellent; others limited.
- **Resource cost.** Each Moshi session holds a GPU slot.

## Ship It

Save as `outputs/skill-duplex-pipeline.md`. Pick pipeline vs full-duplex architecture for a voice-agent workload.

## Exercises

1. **Easy.** Run `code/main.py`. It simulates the two-stream + inner-monologue architecture symbolically.
2. **Medium.** Pull Moshi from HuggingFace, run the server, test one conversation. Measure wall-clock latency.
3. **Hard.** Take your Lesson 12 pipeline agent and compare P50 latency vs Moshi on 20 matched test utterances.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Full-duplex | Hear-and-speak at once | Two audio streams active simultaneously on the same model. |
| Inner monologue | Model's text stream | Moshi emits text tokens alongside its audio output. |
| Depth transformer | Inter-codebook predictor | Small transformer that predicts 8 codebooks within one 80 ms frame. |
| Mimi | Kyutai's codec | 12.5 Hz × 8 codebooks; semantic+acoustic; powers Moshi. |
| Streaming S2S | Audio → audio live | Chunk-by-chunk translation/dialogue, no pipeline stages. |
| Back-channeling | "Mhm" reactions | Moshi can emit small acknowledgments without breaking its turn. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki)

---

## Part 9 (ch138): Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify

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
