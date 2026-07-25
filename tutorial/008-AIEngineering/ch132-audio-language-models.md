# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

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
