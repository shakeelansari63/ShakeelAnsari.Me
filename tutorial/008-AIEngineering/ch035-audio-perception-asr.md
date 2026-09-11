# Audio Features, Classification, ASR & Whisper

> Combined lessons (8 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch123): Audio Fundamentals — Waveforms, Sampling, Fourier Transform

> Waveforms are the raw signal. Spectrograms are the representation. Mel features are the ML-friendly form. Every modern ASR and TTS pipeline walks this ladder, and the first rung is understanding sampling and Fourier.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## Learning Objectives

- Synthesize a sine wave from first principles using only the math stdlib
- Compute the Discrete Fourier Transform by hand and explain what each bin represents
- Detect the dominant frequency of a signal using the DFT peak
- Explain the Nyquist-Shannon theorem and demonstrate aliasing experimentally

## The Problem

A microphone produces a pressure-vs-time signal. Your neural net consumes tensors. Between them sits a stack of conventions that, when violated, produce silent bugs: the model trains fine but the WER doubles, or TTS ships a hiss, or a voice cloning system memorizes the microphone instead of the speaker.

Every bug in speech systems traces back to one of three questions:

1. What sample rate was the data recorded at, and what does the model expect?
2. Is the signal aliased?
3. Are you operating on raw samples or on a frequency representation?

Get these right and the rest of Phase 6 is tractable. Get them wrong and even Whisper-Large-v4 produces garbage.

## The Concept

**Waveform.** A one-dimensional array of floats in `[-1.0, 1.0]`. Indexed by sample number. To convert to seconds, divide by the sample rate: `t = n / sr`. A 10-second clip at 16 kHz is an array of 160,000 floats.

**Sampling rate (sr).** How many samples per second. Common rates in 2026:

| Rate | Use |
|------|-----|
| 8 kHz | Telephony, legacy VOIP. Nyquist at 4 kHz kills consonants. Avoid for ASR. |
| 16 kHz | ASR standard. Whisper, Parakeet, SeamlessM4T v2 all consume 16 kHz. |
| 22.05 kHz | TTS vocoder training for older models. |
| 24 kHz | Modern TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD audio, music. |
| 48 kHz | Film, pro audio, high-fidelity TTS (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** A sample rate of `sr` can unambiguously represent frequencies up to `sr/2`. The `sr/2` boundary is the *Nyquist frequency*. Energy above Nyquist gets *aliased* — folded down into lower frequencies — and corrupts the signal. Always low-pass filter before downsampling.

**Bit depth.** 16-bit PCM (signed int16, range ±32,767) is the universal exchange format. 24-bit for music, 32-bit float for internal DSP. Libraries like `soundfile` read int16 but expose float32 arrays in `[-1, 1]`.

**Fourier Transform.** Any finite signal is a sum of sinusoids at different frequencies. The Discrete Fourier Transform (DFT) computes, for `N` samples, `N` complex coefficients — one per frequency bin. `bin k` maps to frequency `k · sr / N` Hz. Magnitude is amplitude at that frequency, angle is phase.

**FFT.** Fast Fourier Transform: an `O(N log N)` algorithm for the DFT when `N` is a power of 2. Every audio library uses FFT under the hood. A 1024-sample FFT at 16 kHz gives 512 usable frequency bins spanning 0–8 kHz at 15.6 Hz resolution.

**Framing + window.** We do not FFT an entire clip. We chop it into overlapping *frames* (typically 25 ms with 10 ms hop), multiply each frame by a window function (Hann, Hamming) to kill edge discontinuities, then FFT each frame. This is the Short-Time Fourier Transform (STFT).

## Build It

### Step 1: synthesize a sine wave from first principles

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

A 440 Hz sine (concert A) at 16 kHz for 1 second is 16,000 floats. Write with `wave.open(..., "wb")` using 16-bit PCM encoding.

### Step 2: compute the DFT by hand

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — fine for `N=256` to confirm correctness, useless for real audio. Real code calls `numpy.fft.rfft` or `torch.fft.rfft`.

### Step 3: find the dominant frequency

```python
def magnitudes(spectrum):
    return [math.sqrt(re * re + im * im) for re, im in spectrum]

def peak_freq(samples, sr):
    mags = magnitudes(dft(samples))
    half = len(mags) // 2
    mags = mags[:half]
    k = max(range(len(mags)), key=lambda i: mags[i])
    return k * sr / len(samples), k
```

Magnitude peak index `k_star` maps to frequency `k_star * sr / N`. Running this on the 440 Hz sine should return a peak at bin `440 * N / sr`.

### Step 4: demonstrate aliasing

Sample a 7 kHz sine at 10 kHz (Nyquist = 5 kHz). The 7 kHz tone is above Nyquist and folds to `10 − 7 = 3 kHz`. The FFT peak appears at 3 kHz. This is the classic aliasing demo and the reason every DAC/ADC ships with a brick-wall low-pass filter.

```python
def downsample_naive(samples, factor):
    return samples[::factor]
```

```python
# 24 kHz 7 kHz tone, decimated to 8 kHz without low-pass:
#   peak after decimation: 1000.0 Hz (from folding)
#   lesson: always low-pass filter before decimating
```

### Step 5: WAV round-trip with stdlib

```python
import struct, wave

def write_wav(path, samples, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
            for s in samples
        )
        w.writeframes(frames)

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    ints = struct.unpack("<" + "h" * n, raw)
    return [x / 32768.0 for x in ints], sr
```

## Use It

The stack you will actually ship in 2026:

| Task | Library | Why |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | Fastest, stable, returns float32. |
| Resample | `torchaudio.transforms.Resample` or `librosa.resample` | Correct anti-aliasing built in. |
| STFT / Mel | `torchaudio` or `librosa` | GPU-friendly; PyTorch ecosystem. |
| Real-time streaming | `sounddevice` or `pyaudio` | Cross-platform PortAudio bindings. |
| Inspect a file | `ffprobe` or `soxi` | CLI, fast, reports sr/channels/codec. |

Decision rule: **match sample rate before you match anything else**. Whisper expects 16 kHz mono float32. Pass it 44.1 kHz stereo and you will get garbage that looks like a model bug.

## Pitfalls that still ship in 2026

- **Sample-rate mismatch.** Model trained at 16 kHz, inferencing at 44.1 kHz. Silent failure mode.
- **Aliasing from naive decimation.** Downsampling without a low-pass filter folds high-frequency energy into lower bins.
- **Bit-depth truncation.** Writing float32 to int16 without dithering causes quantization noise.
- **Assuming stereo compatibility.** Most ASR models expect mono. Mix down before inference.

## Ship It

Save as `outputs/skill-audio-loader.md`. The skill helps you check that audio input matches the expectations of the downstream model and resamples correctly when it does not.

## Exercises

1. **Easy.** Synthesize a 1-second mix of 220 Hz + 440 Hz + 880 Hz at 16 kHz. Run DFT. Confirm three peaks at the expected bins.
2. **Medium.** Record a 3-second WAV of your voice at 48 kHz. Downsample to 16 kHz using `torchaudio.transforms.Resample` (with anti-aliasing), then to 16 kHz using naive decimation (every third sample). FFT both. Where does the aliasing appear?
3. **Hard.** Build the STFT from scratch using only `math` and the DFT from Step 3. Frame size 400, hop 160, Hann window. Plot magnitudes with `matplotlib.pyplot.imshow`. This is the spectrogram of Lesson 02.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | How many samples per second | Frequency in Hz at which the ADC measures the signal. |
| Nyquist | The max frequency you can represent | `sr/2`; energy above it aliases back down. |
| Bit depth | Resolution of each sample | `int16` = 65,536 levels; `float32` = 24-bit precision in `[-1, 1]`. |
| DFT | The Fourier transform for sequences | `N` samples → `N` complex frequency coefficients. |
| FFT | The fast DFT | `O(N log N)` algorithm requiring `N` = power of 2. |
| Bin | Frequency column | `k · sr / N` Hz; resolution = `sr / N`. |
| STFT | Spectrogram under the hood | Framed + windowed FFT over time. |
| Aliasing | Weird frequency ghosts | Energy above Nyquist mirroring down to lower bins. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/01-audio-fundamentals)

---

## Part 2 (ch124): Spectrograms, Mel Scale & Audio Features

> Neural nets do not consume raw waveforms well. They consume spectrograms. They consume mel spectrograms even better. Every ASR, TTS, and audio classifier in 2026 lives or dies by this single preprocessing choice.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## Learning Objectives

- Implement the Short-Time Fourier Transform (STFT) from scratch using framing, windowing, and DFT
- Build a mel filterbank that maps linear-frequency STFT bins to the perceptually-motivated mel scale
- Produce log-mel spectrograms and MFCCs using only stdlib math
- Explain the tradeoffs of frame size, hop length, and mel count for ASR vs TTS

## The Problem

Take a 10-second 16 kHz clip. That is 160,000 floats, all in `[-1, 1]`, almost perfectly uncorrelated with the label "dog barking" or "the word cat". The raw waveform has the information but in a form the model cannot easily extract. Two identical phonemes spoken 100 ms apart have completely different raw samples.

A spectrogram fixes this. It collapses the temporal detail where human perception ignores it (microsecond jitter) and preserves the structure where perception attends (which frequencies are energetic, over time windows of ~10–25 ms).

Mel spectrograms push further. Humans perceive pitch logarithmically: 100 Hz vs 200 Hz sounds "the same distance apart" as 1000 Hz vs 2000 Hz. The mel scale warps the frequency axis to match.

## The Concept

**STFT (Short-Time Fourier Transform).** Slice the waveform into overlapping frames (typical: 25 ms window, 10 ms hop = 400 samples / 160 samples at 16 kHz). Multiply each frame by a window function (Hann is the default). FFT each frame. Stack the magnitude spectra into a matrix of shape `(n_frames, n_freq_bins)`. That is your spectrogram.

**Log-magnitude.** Raw magnitudes span 5-6 orders of magnitude. Take `log(|X| + 1e-6)` or `20 * log10(|X|)` to compress dynamic range.

**Mel scale.** Frequency `f` in Hz maps to mel `m` by `m = 2595 * log10(1 + f / 700)`. The mapping is roughly linear below 1 kHz and roughly logarithmic above. 80 mel bins covering 0–8 kHz is the standard ASR input.

**Mel filterbank.** A set of triangular filters spaced equally on the mel scale. Each filter is a weighted sum of adjacent FFT bins. Multiplying the STFT magnitude by the filterbank matrix gives the mel spectrogram in one matmul.

**Log-mel spectrogram.** `log(mel_spec + 1e-10)`. Whisper's input. Parakeet's input. SeamlessM4T's input. The universal 2026 audio frontend.

**MFCCs.** Take the log-mel spectrogram, apply a DCT (type II), keep the first 13 coefficients. Decorrelates the features and compresses further. Dominant feature until about 2015 when CNNs/Transformers on raw log-mels caught up.

## Build It

### Step 1: frame the waveform

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

A 10-second 16 kHz clip with `frame_len=400, hop=160` yields 998 frames.

### Step 2: Hann window

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Multiply element-wise before the FFT. Removes spectral leakage caused by truncating at non-zero endpoints.

### Step 3: STFT magnitude

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Production uses `torch.stft` or `librosa.stft` (FFT-backed, vectorized).

### Step 4: mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mels covering 0–8 kHz with `n_fft=400` gives an `(80, 201)` matrix.

### Step 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

### Step 6: MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Apply DCT to each log-mel frame, keep the first 13 coefficients. That is your MFCC matrix. The first coefficient is usually dropped (it encodes overall energy).

## Use It

The 2026 stack:

| Task | Features |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop for fine temporal control |
| Audio classification (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels or raw-waveform SSL |
| Music (MusicGen, Stable Audio 2) | EnCodec discrete tokens (not mels) |
| Keyword spotting | 40 MFCCs for tiny devices |

Rule of thumb: **if you are not working on music, start with 80 log-mels.** The burden of proof is on any deviation.

## Pitfalls that still ship in 2026

- **Mel count mismatch.** Training with 80 mels, inference with 128 mels. Silent failure. Log the feature shape at both ends.
- **Sample-rate mismatch upstream.** Mels computed at 22.05 kHz look different from 16 kHz. Fix SR *before* featurization.
- **dB vs log.** Whisper expects log-mel, not dB-mel. Some HF pipelines autodetect; your custom code will not.
- **Normalization drift.** Per-utterance normalization during training, global normalization during inference. Production bug that doubles WER.
- **Leakage from padding.** Zero-padding the end of a clip produces a flat spectrum in the trailing frames. Pad symmetrically or replicate.

## Ship It

Save as `outputs/skill-feature-extractor.md`. The skill picks feature type, mel count, frame/hop, and normalization for a given model target.

## Exercises

1. **Easy.** Run `code/main.py`. It synthesizes a chirp (frequency swept 200 → 4000 Hz) and prints the argmax mel bin per frame. Plot (optional) and confirm it matches the sweep.
2. **Medium.** Re-run with `n_mels` in `{40, 80, 128}` and `frame_len` in `{200, 400, 800}`. Measure sharp-peak bandwidth across the time axis. Which combo resolves the chirp the best?
3. **Hard.** Implement `power_to_db` and compare ASR accuracy of a tiny CNN classifier on AudioMNIST using (a) raw log-mel, (b) dB-mel with `ref=max`, (c) MFCC-13 + delta + delta-delta. Report top-1 accuracy.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Frame | A slice | 25 ms chunk of waveform fed to one FFT. |
| Hop | Stride | Samples between consecutive frames; 10 ms is ASR default. |
| Window | Hann/Hamming thing | Point-wise multiplier that tapers the frame edges to zero. |
| STFT | Spectrogram generator | Framed + windowed FFT; yields time × frequency matrix. |
| Mel | Warped frequency | Log-perception scale; `m = 2595·log10(1 + f/700)`. |
| Filterbank | The matrix | Triangular filters that project STFT onto mel bins. |
| Log-mel | Whisper's input | `log(mel_spec + eps)`; standardized in 2026. |
| MFCC | Old-school feature | DCT of log-mel; 13 coeffs, decorrelated. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/02-spectrograms-mel-features)

---

## Part 3 (ch125): Audio Classification — From k-NN on MFCCs to AST and BEATs

> Everything from "dog barking vs siren" to "which language is this" is audio classification. The features are mels. The architecture moves each decade. The evaluation stays AUC, F1, and per-class recall.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## Learning Objectives

- Build a k-NN classifier on MFCC features and evaluate it on synthetic audio data
- Implement a 2D CNN on log-mel spectrograms for classification
- Use pre-trained AST and BEATs models from HuggingFace for transfer learning
- Apply SpecAugment and Mixup to combat class imbalance

## The Problem

You get a 10-second clip. You want to know: "what is it?" Urban sound (siren, drill, dog), speech command (yes/no/stop), language ID (en/es/ar), speaker emotion (angry/neutral), or environmental sound (indoor/outdoor, babble). All of these are *audio classification*.

The core difficulty is not the network. It is data. Audio datasets have brutal class imbalance, strong domain shift (clean vs noisy), and label noise. The 80% of the problem is curation, augmentation, and evaluation, not swapping CNN for Transformer.

## The Concept

**k-NN on MFCCs (the 1990s baseline).** Flatten MFCCs per clip, compute cosine similarity to a labeled bank, return majority vote of the top K. Surprisingly strong on clean, small datasets (Speech Commands, ESC-50). Runs with no GPU.

**2D CNN on log-mels (2015-2019).** Treat the `(T, n_mels)` log-mel as an image. Apply ResNet-18 or VGG-style. Global mean pool the time axis. Softmax over classes.

**Audio Spectrogram Transformer, AST (2021-2024).** Patchify the log-mel (e.g. 16×16 patches), add position embeddings, feed to a ViT. State of the art on AudioSet (mAP 0.485) for supervised learning.

**BEATs (2024+).** Self-supervised pretraining on millions of hours. Fine-tune on your task with 1-10% of the supervised data. BEATs-iter3 beats AST by 1-2 mAP on AudioSet while using 1/4 the compute.

### Class imbalance is the real challenge

ESC-50: 50 classes, 40 clips each — balanced, easy. UrbanSound8K: 10 classes, imbalanced 10:1. AudioSet: 632 classes with a 100,000:1 long tail. Techniques that work:

- Balanced sampling during training (not in evaluation).
- Mixup: linearly interpolate two clips (and their labels) as augmentation.
- SpecAugment: mask random time and frequency bands. Simple; critical.

### Evaluation

- Multiclass exclusive (Speech Commands): top-1 accuracy, top-5 accuracy.
- Multiclass multi-label (AudioSet, UrbanSound-style): mean average precision (mAP).
- Heavily imbalanced: per-class recall + macro F1.

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## Build It

### Step 1: featurize

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2: fixed-length summary

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Simple but strong: mean + variance across time gives a 26-dim fixed embedding for a 13-coef MFCC.

### Step 3: k-NN

```python
import math
from collections import Counter

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4: upgrade to CNN on log-mels

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M parameters. Trains in ~10 min on ESC-50 with a single RTX 4090. 80%+ accuracy.

### Step 5: the 2026 default — fine-tune BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

## Use It

The 2026 stack:

| Situation | Start with |
|-----------|-----------|
| Tiny dataset (<1000 clips) | k-NN on MFCC means (your baseline) + audio augmentation |
| Medium dataset (1K–100K) | BEATs or AST fine-tune |
| Large dataset (>100K) | Train from scratch or fine-tune Whisper-encoder |
| Real-time, edge | 40-MFCC CNN, quantized to int8 (KWS-style) |
| Multi-label (AudioSet) | BEATs-iter3 with BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID, SpeechBrain VoxLingua107 baseline |

Decision rule: **start with a frozen backbone, not a fresh model**. Fine-tuning a BEATs head gets you 95% of SOTA in hours, not weeks.

## Ship It

Save as `outputs/skill-classifier-designer.md`. Pick architecture, augmentations, class-balance strategy, and eval metric for a given audio classification task.

## Exercises

1. **Easy.** Run `code/main.py`. It trains the k-NN MFCC baseline on a 4-class synthetic dataset (pure tones at different pitches). Report confusion matrix.
2. **Medium.** Replace `summarize` with [mean, var, skew, kurtosis]. Does 4-moment pooling beat mean+var on the same synthetic dataset?
3. **Hard.** Using `torchaudio`, train a 2D CNN on ESC-50 fold 1. Report 5-fold cross-validation accuracy. Add SpecAugment (time mask = 20, freq mask = 10) and report the delta.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| AudioSet | The ImageNet of audio | Google's 2M-clip, 632-class weakly-labeled YouTube dataset. |
| ESC-50 | Small classification benchmark | 50 classes × 40 clips of environmental sounds. |
| AST | Audio Spectrogram Transformer | ViT on log-mel patches; 2021 SOTA. |
| BEATs | Self-supervised audio | Microsoft model, iter3 leads AudioSet as of 2026. |
| Mixup | Pair augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Mask-based augmentation | Zero-out random time and frequency bands of the spectrogram. |
| mAP | Main multi-label metric | Mean average precision across classes and thresholds. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/03-audio-classification)

---

## Part 4 (ch126): Speech Recognition (ASR) — CTC, RNN-T, Attention

> Speech recognition is audio classification at every timestep, glued together by a sequence model that knows English and silence. CTC, RNN-T, and attention are the three ways to do it. Pick one and understand why.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## Learning Objectives

- Implement greedy and beam-search decoding for CTC-based ASR
- Compute Word Error Rate (WER) using Levenshtein distance at the word level
- Run Whisper inference on a WAV file and interpret its output
- Explain the tradeoffs between CTC, RNN-T, and attention encoder-decoder architectures

## The Problem

You have a 10-second 16 kHz clip. You want a string: "turn on the kitchen lights". The challenge is structural: audio frames do not align one-to-one with characters. The word "okay" might take 200 ms or 1200 ms. Silence punctuates the utterance. The number of output tokens is not known in advance.

Three formulations solve this:

1. **CTC (Connectionist Temporal Classification).** Emit per-frame token probabilities including a special *blank*. Collapse repeats and blanks at decode time. Non-autoregressive, fast. Used by wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Joint network predicts next token given encoder frame and previous tokens. Streamable. Used by Google's on-device ASR, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Encoder compresses audio to hidden states, decoder cross-attends to generate tokens autoregressively. Used by Whisper, SeamlessM4T.

## The Concept

**CTC intuition.** Let the encoder output `T` frame-level distributions over `V+1` tokens (V chars + blank). For a target string `y` of length `U < T`, any frame alignment that collapses to `y` counts. CTC loss sums over all such alignments. Inference: per-frame argmax, collapse repeats, remove blanks.

**RNN-T intuition.** Adds a *predictor* network that embeds the token history and a *joiner* that combines predictor state with encoder frame into a joint distribution over `V+1` (the `+1` is a null / no-emit). Explicitly models the conditional dependence CTC ignored.

**Attention encoder-decoder.** Encoder (6-32 transformer layers) over log-mel frames. Decoder (6-32 transformer layers) cross-attends to encoder outputs to generate tokens autoregressively. No alignment constraint.

### WER: the one number

**Word Error Rate** = `(S + D + I) / N`, where S=substitutions, D=deletions, I=insertions, N=reference word count.

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

## Build It

### Step 1: greedy CTC decode

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Two rules: collapse consecutive repeats, drop blanks. Example: `a a _ _ a b b _ c` → `a a b c`.

### Step 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Production uses prefix tree beam search with LM fusion; this is the conceptual skeleton.

### Step 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Step 4: inference against Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

### Step 5: streaming with Parakeet

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

## Use It

The 2026 stack:

| Situation | Pick |
|-----------|------|
| English, offline, max quality | Whisper-large-v3-turbo |
| Multilingual, robust | SeamlessM4T v2 |
| Streaming, low latency | Parakeet-TDT-1.1B or Riva |
| Edge, mobile, <500 ms latency | Whisper-Tiny quantized or Moonshine (2024) |
| Long-form | Whisper with VAD-based chunking (WhisperX) |
| Domain-specific (medical, legal) | Fine-tune wav2vec 2.0 + domain LM fusion |

## Pitfalls that still ship in 2026

- **No VAD.** Running Whisper on silence produces hallucinations ("Thanks for watching!"). Always gate with VAD.
- **Character vs word vs subword WER.** Report word-level WER *after* normalization (lowercase, punctuation stripped).
- **Language ID drift.** Whisper's auto LID mis-routes noisy clips; force `language="en"` when you know.
- **Long clips without chunking.** Whisper has a 30-second window. Use `chunk_length_s=30, stride=5` for anything longer.

## Ship It

Save as `outputs/skill-asr-picker.md`. Pick model, decoding strategy, chunking, and LM fusion for a given deployment target.

## Exercises

1. **Easy.** Run `code/main.py`. It greedily decodes a hand-crafted CTC output and computes WER against a reference.
2. **Medium.** Implement the prefix-tree beam search in Step 2 properly (account for the blank merge rule). Compare with greedy on a 10-example synthetic dataset.
3. **Hard.** Use `whisper-large-v3-turbo` on LibriSpeech test-clean. Compute WER on the first 100 utterances. Compare with published numbers.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| CTC | The blank-token loss | Marginal over all frame-to-token alignments; non-AR. |
| RNN-T | The streaming loss | CTC + next-token predictor; handles word-order. |
| Attention enc-dec | Whisper-style | Encoder + cross-attending decoder; best offline quality. |
| WER | The number you report | `(S+D+I)/N` at word level. |
| Blank | The emptiness | Special token in CTC signalling "no emission this frame". |
| LM fusion | External language model | Add weighted LM log-probs during beam search. |
| VAD | The silence gate | Voice activity detector; trims non-speech. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/04-speech-recognition-asr)

---

## Part 5 (ch127): Whisper — Architecture & Fine-Tuning

> Whisper is a 30-second-window transformer encoder-decoder, trained on 680k hours of multilingual weakly-supervised audio-text pairs. One architecture, multiple tasks, robust across 99 languages. The 2026 reference ASR.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Learning Objectives

- Run Whisper inference with optimal parameters to avoid hallucinations
- Implement chunked long-form transcription with WhisperX
- Apply LoRA fine-tuning to adapt Whisper for a domain-specific task
- Inspect cross-attention weights to understand what the decoder attends to

## The Problem

Whisper is not a pipeline you can treat as a black box forever. Domain shift kills it — technical jargon, speaker accents, proper nouns, short clips, silence. You need to know:

1. What it actually is inside.
2. How to give it chunked, streaming, or long-form audio correctly.
3. When to fine-tune and how.

## The Concept

**Architecture.** Standard transformer encoder-decoder.

- Input: 30-second log-mel spectrogram, 80 mels, 10 ms hop → 3000 frames.
- Encoder: conv-downsample (stride 2) + `N` transformer blocks. For Large-v3: 32 layers, 1280-dim, 20 heads.
- Decoder: `N` transformer blocks with causal self-attn + cross-attn to encoder output.
- Output: BPE tokens over a 51,865-token vocab.

Large-v3 has 1.55B params. Turbo uses a 4-layer decoder (from 32), cutting latency 8× with a <1% WER hit.

**The prompt format.** Whisper is steered by special tokens:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — language tag; forces translation-vs-transcription.
- `<|transcribe|>` or `<|translate|>` — English output from any input, or verbatim.
- `<|notimestamps|>` — skip word-level timestamps (faster).

**30-second window.** Everything is pinned to 30 seconds. Longer clips need chunking.

**Log-mel normalization.** `(log_mel - mean) / std` where the stats come from Whisper's own corpus. You *must* use Whisper's preprocessing, not `librosa`.

### Variants in 2026

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |

### Fine-tuning

Canonical workflow in 2026:

1. Collect 10–100 hours of target-domain audio with aligned transcripts.
2. Run `transformers.Seq2SeqTrainer` with `generate_with_loss` callback.
3. Parameter-efficient: LoRA on `q_proj`, `k_proj`, `v_proj` reduces GPU memory 4× with <0.3 WER cost.
4. Freeze the encoder if you have <10 hours. Only tune the decoder.
5. Use Whisper's own tokenizer and prompt format.

## Build It

### Step 1: run Whisper out of the box

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,
)
print(result["text"])
```

Key defaults you should always override: `temperature=0.0`, `condition_on_previous_text=False`, and `no_speech_threshold=0.6`.

### Step 2: chunked long-form

```python
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX adds (1) Silero VAD gating, (2) word-level alignment via wav2vec 2.0, (3) diarization.

### Step 3: fine-tune with LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# ~3M trainable / 809M total
```

Then standard Trainer loop. Checkpoint every 1000 steps.

### Step 4: inspect cross-attention weights

```python
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
```

Visualize with a heatmap — you will see diagonal alignment as decoder steps scan through encoder frames.

## Use It

The 2026 stack:

| Situation | Pick |
|-----------|------|
| General English, offline | Large-v3-turbo via `whisperx` |
| Mobile / edge | Whisper-Tiny quantized (int8) or Moonshine |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
| Low-resource language | Fine-tune Medium or Turbo with LoRA |
| Streaming (2 s latency) | Whisper-Streaming or Parakeet-TDT |
| Word-level timestamps | WhisperX (forced alignment via wav2vec 2.0) |

## Pitfalls that still ship in 2026

- **Hallucinated text on silence.** Whisper trained on captions includes "Thanks for watching!", "Subscribe!". Always VAD-gate.
- **`condition_on_previous_text` cascade.** One hallucination pollutes subsequent windows. Set `False`.
- **Short-clip padding.** A 2-second clip padded to 30 seconds can hallucinate in the trailing silence.
- **Wrong mel stats.** Using librosa's mels instead of Whisper's produces near-random output.

## Ship It

Save as `outputs/skill-whisper-tuner.md`. Design a Whisper fine-tune or inference pipeline for a given domain.

## Exercises

1. **Easy.** Run `code/main.py`. It tokenizes a Whisper-style prompt, computes decoded shape budgets, and prints the chunk schedule for a 10-minute clip.
2. **Medium.** Install `faster-whisper`, transcribe a 10-minute podcast, compare WER against a human transcript. Try `language="auto"` vs forced `language="en"`.
3. **Hard.** Using HF `datasets`, pick a language Whisper struggles with (e.g., Urdu), fine-tune Medium with LoRA for 2 epochs on 2 hours, and report WER delta.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 30-sec window | Whisper's limit | Hard input cap; chunk longer audio. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` kicks off the decoder prompt. |
| Timestamps token | Temporal alignment | Every 0.02 s offset is a special token in the 51k vocab. |
| Turbo | The fast variant | 4-decoder layers, 8× faster, <1% WER regression. |
| WhisperX | The long-form wrapper | VAD + Whisper + wav2vec alignment + diarization. |
| LoRA fine-tune | Efficient tuning | Add low-rank adapters to attention; train ~0.3% of params. |
| Hallucination | The silent failure | Whisper produces fluent English from noise/silence. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/05-whisper-architecture-finetuning)

---

## Part 6 (ch128): Speaker Recognition & Verification

> ASR asks "what did they say?" Speaker recognition asks "who said it?" The math looks the same — embeddings plus cosine — but every production decision hinges on a single EER number.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## Learning Objectives

- Build a speaker embedding from MFCC statistics and compute cosine similarity
- Implement the Equal Error Rate (EER) metric from same-speaker and different-speaker score distributions
- Use SpeechBrain's ECAPA-TDNN for production speaker verification
- Perform diarization with pyannote.audio 3.1

## The Problem

A user says a passphrase. You want to know: is this the person they claim to be (*verification*, 1:1), or is it the first person in your enrollment bank (*identification*, 1:N)? Or neither — is this an unknown speaker (*open-set*)?

The metric is **EER** — Equal Error Rate. Set your decision threshold so False Accept Rate = False Reject Rate. The crossover is EER.

## The Concept

**The pipeline.** Enrollment: record 5–30 seconds of the target speaker; compute a fixed-dimension embedding (192-d for ECAPA-TDNN, 256-d for WavLM-large). Verification: get the test utterance embedding; compute cosine similarity; compare to a threshold.

**ECAPA-TDNN (2020, still dominant 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. 1D conv blocks with squeeze-excitation, multi-head attention pooling, followed by a linear layer to 192-d.

**WavLM-SV (2022+).** Fine-tune a pretrained WavLM-large SSL backbone with AAM loss. Higher quality but slower.

**x-vector (baseline).** TDNN + statistics pooling. Classic; still useful on CPU / edge.

### Scoring

- **Cosine** between enrollment and test embeddings.
- **PLDA (Probabilistic LDA).** Project embeddings into a latent space where same-speaker vs different-speaker has a closed-form likelihood ratio.
- **Score normalization.** `S-norm` or `AS-norm`: normalize each score against a cohort of imposter means and stds.

### Numbers you should know (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization

"Who spoke when" in a multi-speaker clip. Pipeline: VAD → segment → embed each segment → cluster (agglomerative or spectral) → smooth boundaries. Modern stack: `pyannote.audio` 3.1.

## Build It

### Step 1: toy embedding from MFCC statistics

```python
import math

def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

### Step 2: cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3: EER from similarity pairs

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

### Step 4: production with SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25
```

### Step 5: diarize with pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Use It

The 2026 stack:

| Situation | Pick |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST or RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Pitfalls

- **Channel mismatch.** Model trained on VoxCeleb (web video) ≠ phone-call audio.
- **Short utterances.** EER degrades sharply below 3 seconds of test audio.
- **Enrollment with noise.** One noisy enrollment poisons the anchor. Use ≥3 clean samples and average.
- **Fixed threshold across conditions.** Always tune the threshold on a held-out dev set.
- **Cosine on non-normalized embeddings.** L2-normalize first.

## Ship It

Save as `outputs/skill-speaker-verifier.md`. Pick model, enrollment protocol, threshold-tuning plan, and fraud safeguards.

## Exercises

1. **Easy.** Run `code/main.py`. Builds synthetic "speakers" (different tone profiles), enrolls, computes EER on a 100-pair trial list.
2. **Medium.** Use SpeechBrain ECAPA on 30 VoxCeleb1 utterances (5 speakers × 6 each). Compute EER with cosine vs PLDA.
3. **Hard.** Build the full enroll → diarize → verify pipeline with `pyannote.audio`. Evaluate DER on AMI dev set.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | The headline metric | Threshold where False Accept = False Reject. |
| Verification | 1:1 | "Is this Alice?" |
| Identification | 1:N | "Who is speaking?" |
| Open-set | Unknown possible | Test set can contain unenrolled speakers. |
| Enrollment | Registering | Computing a speaker's reference embedding. |
| AAM-softmax | The loss | Softmax with additive angular margin; forces cluster separation. |
| PLDA | Classic scoring | Probabilistic LDA; likelihood-ratio scoring on top of embeddings. |
| DER | Diarization metric | Diarization Error Rate — miss + false alarm + confusion. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/06-speech-and-audio/06-speaker-recognition-verification)

---

## Part 7 (ch136): Voice Activity Detection & Turn-Taking — Silero, Cobra, and the Flush Trick

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

---

## Part 8 (ch139): Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards

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
