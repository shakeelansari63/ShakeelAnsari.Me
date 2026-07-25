# Whisper — Architecture & Fine-Tuning

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
