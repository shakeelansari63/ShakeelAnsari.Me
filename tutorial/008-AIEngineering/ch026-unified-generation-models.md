# Transfusion, Show-o, Janus & Any-to-Any Models

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch234): Transfusion: Autoregressive Text + Diffusion Image in One Transformer

> Chameleon and Emu3 bet everything on discrete tokens. They work, but the quantization bottleneck is visible — the image quality plateaus below continuous-space diffusion models. Transfusion (Meta, Zhou et al., August 2024) takes the opposite bet: keep images continuous, drop the VQ-VAE entirely, and train one transformer with two losses. Text tokens get next-token-prediction. Image patches get a flow-matching / diffusion loss. Both objectives optimize the same weights. The architecture underlying Stable Diffusion 3 (MMDiT) is a close cousin. This lesson reads the Transfusion thesis, builds a toy two-loss trainer, and traces the attention mask that lets one transformer do both jobs.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- Wire a transformer that runs two losses (NTP on text tokens, diffusion MSE on image patches) on one backbone.
- Explain why bidirectional attention across image patches plus causal attention over text tokens is the right mask choice.
- Compare Transfusion-style (continuous images, diffusion loss) to Chameleon-style (discrete images, NTP) on compute, quality, and code complexity.
- Name MMDiT's contribution: modality-specific weights at each block, joint attention at the residual stream.

## The Problem

The discrete vs continuous image tokens debate is older than LLMs. Continuous representations (raw pixels, VAE latents) preserve detail. Discrete tokens (VQ indices) fit the transformer's native vocabulary but lose detail at the quantization step.

Chameleon / Emu3 went discrete: one loss, one architecture, but image fidelity capped by tokenizer quality.

Diffusion models went continuous: exceptional image quality, but a separate model from the LLM, complex noise-schedule engineering, and no clean integration with text generation.

Transfusion asks: can we have both? Keep images continuous, still train one model, use two losses stitched into one gradient step.

## The Concept

### The two-loss architecture

A single decoder-only transformer processes a sequence that contains:

- Text tokens (discrete, from BPE vocab).
- Image patches (continuous, 16×16 pixel blocks projected into hidden dim via linear embedding — same as a ViT encoder's input).
- `<image>` and `</image>` tags marking where continuous patches live.

Forward pass runs once. The loss picks one of two heads per token:

- For text tokens: standard cross-entropy on the vocab-logits head.
- For image patches: diffusion loss on continuous patches — predict the noise that was added to each patch.

The gradient flows through the shared transformer body. Both losses improve the shared weights simultaneously.

### Attention mask: causal text + bidirectional image

Text tokens must be causal — you cannot let a text token attend to future text, or teacher forcing breaks. Image patches, however, represent one snapshot; they should attend to each other bidirectionally within the same image block.

The mask:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Implemented as a block-triangular mask at training and inference.

### Diffusion loss inside the transformer

The diffusion loss is standard: add noise to an image patch, ask the model to predict the noise (or the clean patch, equivalently). Transfusion's version uses flow matching — predict the velocity field from noisy to clean.

During training:
1. For each image patch x0, sample a random timestep t.
2. Sample noise ε, compute xt = (1-t) * x0 + t * ε (linear interpolation for flow matching).
3. The transformer predicts v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backprop alongside text NTP losses from the same sequence.

At inference, generation is:
- Text tokens: standard autoregressive sampling.
- Image patches: diffusion sampling loop (10-30 steps typical) conditioned on the prior text tokens.

### MMDiT: Stable Diffusion 3's variant

Stable Diffusion 3 (Esser et al., March 2024) shipped MMDiT (Multimodal Diffusion Transformer) around the same time as Transfusion. The architectures are siblings.

MMDiT's key differences:

- Modality-specific weights per block. Each transformer block has separate Q, K, V, and MLP weights for text tokens vs image patches. Attention is joint (cross-modality); everything else is modality-specific.
- Rectified flow training. A specific flow-matching variant with known sampling and simpler math than DDPM.
- Scale. MMDiT is the backbone for SD3 (2B and 8B param variants). Transfusion's paper scales to 7B.

Both converge on the same core idea: one transformer runs NTP on text and diffusion on continuous image representations.

### Why this beats Chameleon-style

The quality gap between continuous-diffusion and discrete-NTP on image generation is measurable. Transfusion paper reports:

- At 7B params, beats a same-size Chameleon-style model on FID by 3-5 points.
- No tokenizer training required — the image encoder is simpler (Linear projection to hidden, same as a ViT's input layer).
- Inference can parallelize image patch denoising, unlike autoregressive image tokens.

Downside: Transfusion is a dual-loss model, making training dynamics trickier. Loss weights need tuning. Schedule mismatch between NTP and diffusion can cause one head to dominate.

### What sits downstream

Janus-Pro (Lesson 12.15) refines Transfusion's idea by decoupling the vision encoder for understanding and generation — SigLIP for one, VQ for the other — while sharing the transformer body. Show-o (Lesson 12.14) swaps diffusion for discrete-diffusion (masked prediction). The unified-generation family branches rapidly after Transfusion.

2026 production VLMs that emit images — Gemini 3 Pro, GPT-5, Claude Opus 4.7's image generation path — almost certainly use some descendant of this family. Details are proprietary.

## Use It

`code/main.py` builds a toy Transfusion on a tiny MNIST-like problem:

- Text captions are short integer sequences describing a digit (0-9).
- Images are 4×4 grids of bytes.
- A pair of shared-weight linear projections acts as the transformer stand-in; NTP loss on text, MSE loss on noisy patches.
- Training loop alternates the two losses, attention mask is explicit.
- Generation produces a text caption and a 4×4 image in one forward pass.

The transformer is a toy. The two-loss plumbing, attention mask construction, and inference loop are the real artifacts.

## Ship It

This lesson produces `outputs/skill-two-loss-trainer-designer.md`. Given a new multimodal training task (text + image, text + audio, text + video), it designs the two-loss schedule (loss weights, mask shape, shared vs modality-specific blocks) and flags implementation risks.

## Exercises

1. A Transfusion-style model trains 70% text tokens and 30% image patches. The image diffusion loss is ~10× the text NTP loss in magnitude. What loss weights balance them?

2. Implement the block-triangular mask for a sequence: `[T, T, <image>, P, P, P, P, </image>, T]`. Mark each entry 0 or 1.

3. MMDiT has modality-specific QKV weights. What parameter count overhead does this add vs Transfusion's fully-shared transformer? At 7B params, is it worth it?

4. Generation: given a text prompt, the model runs NTP for 50 tokens, then hits `<image>`, then runs diffusion on 256 patches over 20 denoise steps. How many forward passes total?

5. Read SD3 paper Section 3. Describe rectified flow and why it converges in fewer inference steps than DDPM.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | A single transformer optimizes both cross-entropy on text tokens and MSE on continuous image patches in the same gradient step |
| Flow matching | "Rectified flow" | Diffusion variant that predicts a velocity field from noise to clean data; simpler math than DDPM |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3's architecture: joint attention, modality-specific MLPs and norms |
| Block-triangular mask | "Causal text + bidirectional image" | Attention mask that is causal across text but bidirectional within image regions |
| Continuous image representation | "No VQ" | Image patches as real-valued vectors, not integer codebook indices |
| Velocity prediction | "v-parameterization" | Network output is the velocity field between noise and data, not the noise itself |

## Further Reading

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion)

---

## Part 2 (ch235): Show-o and Discrete-Diffusion Unified Models

> Transfusion mixes continuous and discrete representations. Show-o (Xie et al., August 2024) goes the other way: text tokens use causal next-token prediction, image tokens use masked discrete diffusion in the spirit of MaskGIT. Both sit inside one transformer with a hybrid attention mask. The result unifies VQA, text-to-image, inpainting, and mixed-modality generation on one backbone, one tokenizer per modality, one loss formulation (next-token extended to masked prediction). This lesson walks the Show-o design — why masked discrete diffusion is a parallel, few-step image generator — and contrasts with Transfusion and Emu3.

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## Learning Objectives

- Explain masked discrete diffusion: the schedule that masks tokens uniformly then asks the transformer to recover them.
- Compare parallel image decoding (Show-o, MaskGIT) to autoregressive image decoding (Chameleon, Emu3) on speed and quality.
- Name the three tasks Show-o handles in one checkpoint: T2I, VQA, image inpainting.
- Pick a masking schedule (cosine, linear, truncated) and reason about its effect on sample quality.

## The Problem

Transfusion's two-loss training works but has trickier dynamics — the continuous diffusion loss lives on a different numerical scale from the discrete NTP loss. Balancing loss weights is a hyperparameter search. The architecture is effective but complex.

Show-o's answer: keep both modalities discrete (like Chameleon), but generate images in parallel via masked discrete diffusion instead of sequentially. The training objective becomes a single masked-token-prediction that generalizes next-token-prediction naturally.

## The Concept

### Masked discrete diffusion (MaskGIT)

The original Chang et al. (2022) MaskGIT trick is elegant. Start from a fully-masked image (every token is the special `<MASK>` id). At each step, predict all masked tokens in parallel, then keep the top-K most confident predictions and re-mask the rest. After ~8-16 iterations, all tokens are filled in. The schedule of how many tokens to unmask per step is tuned — cosine schedules work well.

Training is simple: sample a masking ratio uniformly from [0, 1], apply it to the image's VQ tokens, train the transformer to recover the masked ones. Exactly what BERT did for text, scaled to image generation.

### Show-o: one transformer, hybrid mask

Show-o puts MaskGIT inside a causal-language-model transformer. The attention mask is:

- Text tokens: causal (standard LLM).
- Image tokens: full bidirectional within the image block (so the masked tokens can see every other image token during prediction).
- Text-to-image: text attends to prior images, image attends to prior text.

Training alternates between:
1. Standard NTP on text sequences.
2. T2I samples: text → image with masked image tokens, masked-token-prediction loss.
3. VQA samples: image → text with masked text tokens (really just NTP).

The unified loss is cross-entropy on `<MASK>` tokens, which covers both text NTP (only the last token is "masked") and image masked-diffusion (random subset is masked).

### Parallel sampling

Show-o generates an image in ~16 steps instead of ~1000 (autoregressive per token) or ~20 (diffusion). At each step, predict all masked tokens in parallel; commit the top-K confident; repeat.

Compare:
- Chameleon / Emu3 (autoregressive over tokens): N_tokens forward passes, typically 1024-4096 per image.
- Transfusion (continuous diffusion): ~20 steps, each a full transformer pass.
- Show-o (masked discrete diffusion): ~16 steps, each a full transformer pass.

Show-o is faster than Chameleon at similar-scale models, roughly matches Transfusion step count with lower per-step cost (discrete vocab logits vs continuous MSE loss).

### Tasks in one checkpoint

Show-o supports four tasks at inference, selected by prompt format:

- Text generation: standard autoregressive text output.
- VQA: image in, text out.
- T2I: text in, image out via masked discrete diffusion.
- Inpainting: image with some tokens masked, fill in.

The inpainting capability comes for free from the masked-prediction training. Mask a region of the VQ-token grid, feed the rest plus a text prompt, predict the masked tokens.

### Masking schedule

The schedule of how many tokens to unmask per step shapes quality. Show-o recommends cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

At step 0, all tokens masked (ratio 1.0). At step T, none masked. Cosine concentrates mass on mid-range ratios where prediction is most informative. Linear schedules also work but plateau faster.

### Show-o2

Show-o2 (2025 follow-up, arXiv 2506.15564) scales Show-o: larger LLM base, better tokenizer, improved mask schedule. Same architectural pattern.

### Where Show-o sits

In the 2026 taxonomy:

- Discrete tokens + NTP: Chameleon, Emu3. Simple but slow inference.
- Discrete tokens + masked diffusion: Show-o, MaskGIT, LlamaGen, Muse. Parallel sampling, still lossy by tokenizer.
- Continuous + diffusion: Transfusion, MMDiT, DiT. Highest quality, more complex training.
- Continuous + flow matching in a VLM: JanusFlow, InternVL-U. Newest.

Pick by task: Show-o when you want T2I + inpainting + VQA in one open model with reasonable speed; Transfusion when quality is paramount and you can afford the two-loss plumbing.

## Use It

`code/main.py` simulates Show-o sampling:

- A toy grid of 16 VQ tokens.
- A mock "transformer" that predicts logits based on a prompt and the currently-unmasked tokens.
- Parallel masked sampling over 8 steps with cosine schedule.
- Prints the intermediate states (mask pattern evolution) and the final tokens.

Run it, watch the mask dissolve step by step.

## Ship It

This lesson produces `outputs/skill-unified-gen-model-picker.md`. Given a product that needs both understanding (VQA, captioning) and generation (T2I, inpainting) with an open-weights constraint, picks between Show-o family, Transfusion/MMDiT family, and Emu3 / Chameleon family with concrete trade-offs.

## Exercises

1. Masked discrete diffusion samples in ~16 steps. Why not 1? What breaks if you unmask everything at step 0?

2. Inpainting is free with masked diffusion. Propose a product use case (real or hypothetical) where Show-o's inpainting beats a specialist model.

3. Cosine schedule vs linear schedule: trace the number of unmasked tokens per step for T=8. Which is more balanced?

4. A 512×512 Show-o image is 1024 tokens. At vocab K=16384, the model emits 1024 * log2(16384) = 14,336 bits (~1.75 KiB) of data. Stable Diffusion outputs 512*512*24 bits = 6,291,456 bits (~768 KiB) of raw pixels. What is the compression ratio and what quality does it buy?

5. Read LlamaGen (arXiv:2406.06525). How is LlamaGen's class-conditional autoregressive image model different from Show-o's masked approach?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | Training to predict masked tokens; at inference, iteratively unmask the most-confident predictions |
| Cosine schedule | "Unmask schedule" | Decay of mask ratio over inference steps; concentrates confidence growth at mid-range |
| Parallel decoding | "All tokens at once" | Every step predicts the full sequence of masked tokens in one forward pass, then commits top-K |
| Hybrid attention | "Causal + bidirectional" | Mask that is causal over text tokens and bidirectional within image blocks |
| Inpainting | "Fill-in generation" | Condition on an image with some tokens masked, predict the missing ones; free from the training objective |
| Commitment rate | "Top-K per step" | How many tokens are declared "done" per iteration; controls inference vs quality trade-off |

## Further Reading

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified)

---

## Part 3 (ch236): Janus-Pro: Decoupled Encoders for Unified Multimodal Models

> Unified multimodal models have an unavoidable tension. Understanding wants semantic features — SigLIP or DINOv2 output vectors rich with concept-level information. Generation wants reconstruction-friendly codes — VQ tokens that compose back into crisp pixels. The two goals are not compatible in a single encoder. Janus (DeepSeek, October 2024) and Janus-Pro (DeepSeek, January 2025) argue the fix is to stop trying: decouple the two encoders. Share the transformer body between tasks, but route understanding through SigLIP and generation through a VQ tokenizer. At 7B, Janus-Pro beats DALL-E 3 on GenEval while matching LLaVA on MMMU. This lesson reads why two encoders work where one fails.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## Learning Objectives

- Explain why a single shared encoder compromises either understanding or generation quality.
- Describe Janus-Pro's routing: SigLIP features on the input side for understanding, VQ tokens on both input and output for generation.
- Trace the data-mix scaling that makes Janus-Pro succeed where Janus did not.
- Compare decoupled (Janus-Pro), coupled-continuous (Transfusion), and coupled-discrete (Show-o) architectures.

## The Problem

Unified models share a transformer body across understanding and generation. Previous attempts (Chameleon, Show-o, Transfusion) all use one visual tokenizer for both directions. The tokenizer is a compromise:

- Optimized for reconstruction (generation): VQ-VAE captures fine-grained pixel detail but produces tokens with weak semantic coherence.
- Optimized for semantics (understanding): SigLIP embeddings group "cat" images near "cat" tokens but do not permit good reconstruction.

Show-o and Transfusion pay for this with a visible quality tax on one direction. Janus-Pro asks: why require one tokenizer when the tasks have different needs?

## The Concept

### Decoupled visual encoding

Janus-Pro's architecture separates the two encoders:

- Understanding path. Input image → SigLIP-SO400m → 2-layer MLP → transformer body.
- Generation path. Input image (if conditioning on an existing image) → VQ tokenizer → token IDs → transformer body.
- Output generation. Image tokens predicted by the transformer → VQ decoder → pixels.

The transformer body is shared. Everything upstream and downstream of the body is task-specific.

Inputs are disambiguated by prompt format: a `<understand>` tag routes through SigLIP; `<generate>` routes through VQ. Or the routing is implicit from task.

### Why this works

Understanding loss gets SigLIP features, which CLIP-style pretraining has tuned for semantic similarity. The model's perception benchmarks improve over Show-o / Transfusion because the input features are better for the task.

Generation loss gets VQ tokens, which a tokenizer has tuned for reconstruction. Image quality improves over Show-o because VQ codes compose back to pixels cleanly.

The shared transformer body sees two input distributions (SigLIP and VQ) and learns to work with both. The claim: enough data + enough parameters, the body absorbs the switching.

### Data scaling — Janus vs Janus-Pro

Janus (original, arXiv 2410.13848) introduced the decoupling but at small scale (1.3B params, limited data). Janus-Pro (arXiv 2501.17811) scaled:

- 7B params (vs 1.3B).
- 90M image-text pairs for stage 1 (alignment) up from 72M.
- 72M for stage 2 (unified) up from 26M.
- Added 200k image-gen instruction samples for stage 3.

The upshot: Janus-Pro-7B matches LLaVA on MMMU (60.3 vs ~58) and beats DALL-E 3 on GenEval (0.80 vs 0.67). One open model, competitive on both sides of the unified spectrum.

### JanusFlow — the rectified flow variant

JanusFlow (arXiv 2411.07975) swaps the VQ generation path for a rectified-flow generation path (continuous). The split becomes SigLIP-for-understanding + rectified-flow-for-generation. Quality ceilings lift further. The architecture remains decoupled-encoders-shared-body.

### The shared body's job

The transformer body processes a unified sequence but with two input distributions. Its job is to:

- For understanding: consume SigLIP features + text tokens → emit text autoregressively.
- For generation: consume text tokens + (optional image VQ tokens) → emit image VQ tokens autoregressively.

The body has no modality-specific weights per block. It is the text-style transformer you'd expect to find inside Qwen or Llama, plus the two input adapters.

Interestingly, this means Janus-Pro's body could be initialized from a pretrained LLM. Janus-Pro does initialize from DeepSeek-MoE-7B. That choice matters: the LLM contributes reasoning ability that pure-from-scratch unified models struggle to reach.

### Compared to InternVL-U

InternVL-U (Lesson 12.10) is the 2026 follow-up. It combines:

- Native multimodal pretraining (InternVL3 backbone).
- Decoupled-encoder routing (SigLIP in, VQ + diffusion heads out).
- Unified understanding + generation + editing.

InternVL-U subsumes Janus-Pro's architectural choice into a larger framework. The decoupled-encoder idea is now the default for unified models at scale.

### Limitations

Decoupled encoders add architectural complexity. Two tokenizers to train, two input paths to maintain, two sets of fail modes. For products that do not need generation, Janus-Pro is over-engineered — pick a LLaVA-family understanding model.

For products that do not need understanding, Janus-Pro is overqualified — pick a Stable Diffusion 3 / Flux model.

For products that need both, Janus-Pro is now the reference open architecture.

## Use It

`code/main.py` simulates Janus-Pro routing:

- Two mock encoders: SigLIP-like (produces 256-dim semantic vectors) and VQ-like (produces integer codes).
- A prompt router that picks the encoder based on a task tag.
- A shared body (stand-in) that processes token sequences regardless of which encoder produced them.
- A switch from stage 1 (alignment) to stage 3 (instruction tune) weighted-sample schedule.

Print the routed paths for 3 examples: image QA, T2I, image editing.

## Ship It

This lesson produces `outputs/skill-decoupled-encoder-picker.md`. Given a product that wants unified generation + understanding at frontier-ish quality, it picks Janus-Pro, JanusFlow, or InternVL-U with a concrete data-scale recommendation.

## Exercises

1. Janus-Pro-7B beats DALL-E 3 on GenEval. Explain why a 7B open model can match a frontier proprietary model on generation but not on understanding.

2. Implement a router function: given prompt text, classify as `understand` or `generate`. How do you handle ambiguous prompts like "describe and then sketch"?

3. JanusFlow replaces the VQ path with rectified flow. What does the transformer body now output, and what changes in the loss?

4. Propose a fourth task the Janus-Pro architecture could handle with one more decoupled encoder. Examples: image segmentation (DINO-style), depth (MiDaS-style).

5. Read Janus-Pro Section 4.2 on data scaling. Which data stage contributes most to the T2I quality gain vs Janus?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Decoupled encoding | "Two visual encoders" | Separate tokenizer or encoder per direction: semantic for understanding, reconstruction for generation |
| Shared body | "One transformer" | Single transformer processes either encoder's output; no modality-specific weights |
| SigLIP for understanding | "Semantic features" | CLIP-family vision tower providing rich conceptual features but poor reconstruction |
| VQ for generation | "Reconstruction codes" | Vector-quantized tokens that decode cleanly back to pixels |
| JanusFlow | "Rectified-flow variant" | Janus-Pro with a continuous flow-matching generation head instead of VQ |
| Routing tag | "Task tag" | Prompt marker (`<understand>` / `<generate>`) that picks the input encoder |

## Further Reading

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/15-janus-pro-decoupled-encoders)

---

## Part 4 (ch237): MIO and Any-to-Any Streaming Multimodal Models

> GPT-4o ships a product most open models cannot replicate: an agent that hears voice, sees video, and speaks back in real time. The open-ecosystem answer by late 2024 was MIO (Wang et al., September 2024). MIO tokenizes text, image, speech, and music, trains one causal transformer over the interleaved sequences, and generates any modality to any modality. AnyGPT (Zhan et al., February 2024) was the proof of concept; MIO is the scale-up; Unified-IO 2 (Allen AI, December 2023) is the cousin with vision + action grounding. This lesson reads the any-to-any pattern — four tokenizers, one transformer, streaming-friendly decode.

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 minutes

## Learning Objectives

- Design a shared vocabulary that hosts text, image, speech, and music tokens without collisions.
- Compare SEED-Tokenizer (images) and SpeechTokenizer residual-VQ (speech) on compression + reconstruction trade-offs.
- Explain the four-stage curriculum that builds up any-to-any generation.
- Name the three open any-to-any recipes and their main trade-offs: MIO, AnyGPT, Unified-IO 2.

## The Problem

A unified multimodal model is easy to claim and hard to build at scale. Most "any-to-any" systems until 2024 were pipelined: vision model → text representation → speech model → audio. Each hop loses information, adds latency, and complicates training. GPT-4o's demo video showed a single-model alternative with subsecond response; open systems trailed by months.

The engineering challenges:

- Tokenizers must exist for every modality, compress losslessly-enough for reconstruction, and produce tokens at rates the transformer can consume.
- A single vocabulary must allocate space for text (32k+), image (16k+), speech (4k+), music (8k+). Forty-thousand-plus entries minimum.
- Training data must cover every input-output pair (text→image, image→speech, speech→image, etc.) or the model must compose.
- Inference must stream output tokens fast enough for conversational latency (<500ms time-to-first-audio-byte).

## The Concept

### Four tokenizers for four modalities

MIO's tokenizer stack:

- Text: standard BPE, vocab ~32000.
- Image: SEED-Tokenizer (2023) — quantized VAE with discrete codebook, 4096 entries, 32×32 tokens per image.
- Speech: SpeechTokenizer residual-VQ (2023) — encodes 16kHz waveform into 8 hierarchical codebooks; first level is coarse content, later levels add prosody and speaker identity.
- Music: similar residual-VQ (Meta's MusicGen / Encodec family), 4-8 codebooks.

Each modality produces integer tokens. The tokens get disjoint ID ranges in the shared vocabulary:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Total: ~48k vocabulary. The input embedding and output projection span all of it.

### Streaming decode

Speech generation uses residual-VQ. The transformer predicts the base (layer 0) speech tokens; a parallel-decoded residual quantizer predicts the subsequent layers. Each layer 0 token is roughly 50ms of audio at 16kHz.

The streaming pattern:

1. User speaks into mic; real-time audio tokenizer emits speech tokens every 50ms.
2. MIO consumes tokens as they arrive (prompt prefill + incremental forward).
3. Output tokens stream out as generated; a parallel speech decoder converts them to audio samples with ~50-150ms latency.
4. Time-to-first-audio-byte: ~300-500ms in MIO paper, approaching GPT-4o's ~250ms.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612), and Moshi (arXiv:2410.00037) are complementary streaming speech-LLM designs. Moshi in particular achieves 160ms round-trip on a single GPU.

### Four-stage curriculum

MIO's training curriculum:

1. Stage 1 — alignment. Large-scale modality-pair corpora: text-image, text-speech, text-music. Each pair uses its own token vocabulary segment. Trains the shared vocabulary.
2. Stage 2 — interleaved. Multi-modality interleaved documents (blogs with images + video, podcasts with transcripts, etc.). Trains cross-modality context.
3. Stage 3 — speech-enhanced. Extra audio data to lift speech quality without losing text capability.
4. Stage 4 — SFT. Instruction tuning across modalities: VQA, captioning, narration, speech-to-speech dialogue.

Missing a stage degrades specific capabilities: skip stage 2 and the model loses cross-modality context; skip stage 3 and speech is poor.

### Chain-of-visual-thought

MIO introduces chain-of-visual-thought: the model emits intermediate image tokens as a reasoning step. For "is the cat climbing a tree?" the model:

1. Emits `<image>` tokens rendering the scene (from the input image or a sketch).
2. Emits text analyzing the sketch.
3. Emits the final answer.

The rendered intermediate image serves as a scratchpad. Benchmarks improve on spatial-reasoning tasks. The idea mirrors chain-of-thought for text reasoning.

### Competitors in any-to-any

- AnyGPT (arXiv:2402.12226): 4 modalities (text, image, speech, music), similar design.
- Unified-IO 2 (arXiv:2312.17172): adds vision action outputs, depth, normals. More task diversity, smaller scale.
- NExT-GPT (arXiv:2309.05519): LLM + modality-specific diffusion decoders. Not a single-model approach.
- CoDi (arXiv:2305.11846): composable diffusion; any-to-any via shared latent.

MIO is the closest to pure-token any-to-any. AnyGPT is its conceptual ancestor.

### Latency budget

For a conversational product, every component's latency matters:

- Mic to audio tokens: ~50ms.
- Prefill (audio tokens + history): ~100ms on an 8B model.
- First output token: ~50ms.
- Parallel residual-VQ + speech decoder: ~100-150ms.

Total time-to-first-audio-byte: ~300ms minimum. GPT-4o claims ~250ms. Moshi claims 160ms. MIO/AnyGPT are in the 400-600ms range per public benchmarks.

### Why any-to-any stays hard

Even in 2026, open any-to-any models trail closed ones on two axes:

- Speech quality. The residual-VQ tokenizer is lossy; conversational speech sounds robotic compared to ElevenLabs-class voices.
- Cross-modality reasoning. Asking the model "sing about what you see" still fails more often than pure-vision tasks.

These are open research problems. Qwen3-Omni (Lesson 12.20) is the most advanced open attempt in 2025.

## Use It

`code/main.py`:

- Defines the four-modality vocabulary allocation and prints it.
- Routes a list of multimodal inputs (text, image, audio-clip, music) through the tokenizer router.
- Simulates streaming decode for a text-to-speech response with latency counting.
- Computes the expected time-to-first-audio-byte given encoder, prefill, and decoder latencies.

## Ship It

This lesson produces `outputs/skill-any-to-any-pipeline-auditor.md`. Given a conversational product spec (modalities in, modalities out, latency target), it audits the MIO-family design choices and computes the latency budget.

## Exercises

1. Your product accepts speech input and returns speech output. What's the end-to-end latency budget target? List the components that spend time.

2. SpeechTokenizer residual-VQ uses 8 codebooks. Propose why parallel-decoding the residual levels is necessary (vs sequential) and what latency savings it brings.

3. Your vocabulary has 32k text + 4k image + 4k speech. Add 8k music and ~10 separators. What is the embedding-matrix parameter cost at hidden dim 4096?

4. Chain-of-visual-thought emits an intermediate image. What kinds of questions benefit? What kinds are hurt by the extra tokens?

5. Read Moshi (arXiv:2410.00037). Describe its "inner monologue" technique and compare to MIO's chain-of-visual-thought.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Any-to-any | "Multimodal in/out" | A single model that accepts and emits text, image, speech, and music in any direction |
| Residual-VQ | "Speech tokenizer stack" | Multi-codebook tokenization where each layer adds information; base layer is content, later layers are prosody |
| SEED-Tokenizer | "Image codes" | Discrete image tokenizer with 4096-entry codebook used by MIO |
| Chain-of-visual-thought | "Visual scratchpad" | The model generates an intermediate image as a reasoning step before its final answer |
| Time-to-first-audio-byte | "TTFAB" | Latency from user voice to first audio output; <500ms for conversational feel |
| Four-stage curriculum | "Training recipe" | Alignment -> interleaved -> speech-enhanced -> SFT, in that order |

## Further Reading

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/16-mio-any-to-any-streaming)
