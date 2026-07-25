# Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards

> You cannot ship what you cannot measure. This lesson names the 2026 metrics for every audio task: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language (MMAU, LongAudioBench), music (FAD, CLAP), and speaker (EER).

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## Learning Objectives

- Compute WER with proper normalization for ASR evaluation
- Measure TTS quality using UTMOS, SECS, and WER round-trip
- Calculate EER for speaker verification and FAD for music generation
- Navigate the 2026 open leaderboards for every audio task

## The Problem

Every audio task has multiple metrics, each measuring a different axis. The 2026 canonical list:

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER, RTFx, first-token latency |
| TTS | MOS / UTMOS | SECS, WER-on-ASR-round-trip, TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS, CER |
| Speaker verification | EER | minDCF, FAR / FRR at operating point |
| Diarization | DER | JER, speaker confusion |
| Audio classification | top-1, mAP | macro F1, per-class recall |
| Music generation | FAD | CLAP, listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench, AudioCaps FENSE |

## The Concept

### ASR metrics

**WER (Word Error Rate).** `(S + D + I) / N`. Lowercase, strip punctuation, normalize numbers before scoring.

**CER (Character Error Rate).** Same formula, character-level. Used for tone languages where word segmentation is ambiguous.

**RTFx (inverse real-time factor).** Audio seconds processed per wall-clock second. Parakeet-TDT hits 3380x.

### TTS metrics

**MOS (Mean Opinion Score).** 1-5 human rating. Gold standard but slow. 20+ listeners per sample, 100+ samples per model.

**UTMOS (2022-2026).** Learned MOS predictor. Correlates ~0.9 with human MOS. F5-TTS: 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** ECAPA embedding cosine between reference and cloned output. > 0.75 = recognizable clone.

**WER-on-ASR-round-trip.** Run Whisper over TTS output, compute WER against input text. 2026 SOTA: < 2% CER.

### Speaker verification

**EER (Equal Error Rate).** Threshold where FAR = FRR. ECAPA on VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Weighted cost at a chosen operating point.

### Diarization

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. AMI meetings: ~10-20%.

### Music generation

**FAD (Frechet Audio Distance).** Distance between VGGish-embedding distributions of real vs generated audio.

**CLAP Score.** Text-audio alignment score using CLAP embeddings.

### Audio-language benchmarks

**MMAU-Pro.** 1800 hard items. Random chance 25%. Gemini 2.5 Pro ~60%; multi-audio ~22%.

**LongAudioBench.** Multi-minute clips with semantic queries.

### The 2026 leaderboards

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | huggingface.co/spaces/hf-audio/open_asr_leaderboard |
| TTS Arena (HF) | English TTS | huggingface.co/spaces/TTS-AGI/TTS-Arena |
| MMAU-Pro | LALM reasoning | mmaubenchmark.github.io |
| SpeakerBench / VoxSRC | Speaker recognition | voxsrc.github.io |
| HEAR benchmark | Self-supervised audio | hearbenchmark.com |

## Build It

### Step 1: WER with normalization

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3: SECS for voice cloning

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4: FAD for music generation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5: EER for speaker verification

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Use It

Three cardinal rules:

1. **Normalize before scoring.** Report the normalization rule.
2. **Report distributions, not averages.** P50/P95/P99 for latency. Per-class recall for classification.
3. **Run one canonical public benchmark.** Lets reviewers compare apples-to-apples.

## Pitfalls

- **UTMOS extrapolation.** Trained on VCTK-style clean speech; scores noisy / cloned / emotional audio poorly.
- **MOS panel bias.** 20 AMT workers != 20 target users.
- **FAD depends on reference set.** Compare against the same reference distribution across models.
- **Aggregate WER.** A 5% WER overall can hide 30% WER on accented speech. Report by demographic slice.
- **Public benchmark saturation.** Build an in-house held-out set that reflects your traffic.

## Ship It

Save as `outputs/skill-audio-evaluator.md`. Pick metrics, benchmarks, and reporting format for any audio model release.

## Exercises

1. **Easy.** Run `code/main.py`. Compute WER / CER / EER / SECS / FAD-ish / MMAU-ish on toy inputs.
2. **Medium.** Build a TTS round-trip WER harness. Run Kokoro or F5-TTS output through Whisper. Compute WER over 50 prompts.
3. **Hard.** Score your Lesson 10 LALM choice on MMAU-Pro speech + multi-audio subsets (50 items each). Report per-category accuracy.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | `(S+D+I)/N` at word level after normalization. |
| CER | Character WER | For tone languages or char-level systems. |
| MOS | Human opinion | 1-5 rating; 20+ listeners x 100 samples. |
| UTMOS | ML MOS predictor | Learned model; correlates ~0.9 with human MOS. |
| SECS | Voice-clone similarity | ECAPA cosine between reference and clone. |
| EER | Speaker verif score | Threshold where FAR = FRR. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music-gen quality | Frechet distance on VGGish embeddings. |
| RTFx | Throughput | Audio seconds per wall-clock second. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/17-audio-evaluation-metrics)
