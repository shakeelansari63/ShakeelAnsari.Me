# Music Generation — MusicGen, Stable Audio, Suno, and the Licensing Earthquake

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
