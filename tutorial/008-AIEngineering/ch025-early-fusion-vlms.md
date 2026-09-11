# Qwen-VL, InternVL, Chameleon & Emu3

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch230): Qwen-VL Family and Dynamic-FPS Video

> The Qwen-VL family — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — is the most influential open vision-language model lineage in 2026. Each generation made a single decisive architectural bet that the rest of the open ecosystem copied within twelve months: native dynamic resolution via M-RoPE, dynamic-FPS sampling with absolute time alignment, window attention in the ViT, and structured agent output formats. By Qwen3-VL, the recipe had stabilized: a 2D-RoPE-ViT encoder with native-aspect-ratio inputs, an MLP projector into a large Qwen3 language base, and training stages that emphasized OCR, grounding, and agent behavior as first-class targets. This lesson reads the family chronologically so you understand why every knob is where it is.

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## Learning Objectives

- Compute M-RoPE's three-axis rotations (temporal, height, width) and explain why all three are needed.
- Pick a dynamic-FPS sampling strategy for a video and reason about tokens-per-second vs event-detection accuracy.
- Name the four Qwen-VL generational upgrades in order and what each enabled.
- Wire a Qwen2.5-VL-style JSON agent output format and parse structured tool calls from a VLM response.

## The Problem

Qwen-VL shipped in August 2023 as a direct response to LLaVA-1.5 and BLIP-2. The gap the Qwen team targeted was threefold: resolution, video, and structured output.

Resolution: LLaVA-1.5 ran at 336×336. Fine for photos, useless for a Chinese-language invoice or a dense spreadsheet screenshot. Qwen-VL's first innovation was 448×448 and grounded bounding-box output, letting the model point at things.

Video: Video-LLaMA stacked per-frame encoders and fed them to the LLM. It worked for short clips, not for multi-minute videos where the temporal axis is the signal. The Qwen team wanted a single encoder that understood time.

Structured output: LLaVA emitted free-form text. An agent needs JSON. Qwen-VL trained on explicit JSON output formats including bounding-box coordinates as text.

Every Qwen-VL generation extends one of these three axes.

## The Concept

### Qwen-VL (August 2023)

The first generation: OpenCLIP ViT-bigG/14 as encoder (2.5B params), LLama-compatible Q-Former (1-step with 256 queries), Qwen-7B base. Contributions:

- 448×448 resolution (then SOTA for an open VLM).
- Grounding: trained on image-text pairs with explicit coordinate-token output. "The cat is at <box>(112, 204), (280, 344)</box>".
- Chinese + English multilingual training from the start.

Benchmarks at the time: competitive with GPT-4V on English, dominant on Chinese. The grounding supervision was the real headline.

### Qwen2-VL (September 2024) — M-RoPE and native resolution

Qwen2-VL replaced the fixed-resolution + Q-Former stack with a natively dynamic-resolution ViT encoder. Key changes:

- Native dynamic resolution. The ViT accepts any H×W divisible by 28 (patch 14 with 2× spatial merge). An image at 1120×672 (40×24 merged patches) produces 960 visual tokens. No resize, no tiling, no thumbnail.
- M-RoPE (Multimodal RoPE). Each token carries a 3D position (t, h, w) instead of 1D. For images t=0, for video t = frame_index. RoPE rotates query/key vectors by a frequency per axis. No positional embedding table.
- MLP projector. Drop the Q-Former; use a 2-layer MLP on the merged patch tokens.
- Video with dynamic FPS. Video sampled at 1-2 FPS by default, but the model accepts arbitrary frame counts.

Result: Qwen2-VL-7B matched GPT-4o on several multimodal benchmarks and beat it on DocVQA (94.5 vs 88.4). The architecture change was the decisive move.

### Qwen2.5-VL (February 2025) — dynamic FPS + absolute time

Qwen2.5-VL's big shift was video. Dynamic FPS is not just "sample more frames when needed." The paper formalized:

- Absolute time tokens. Instead of positional indices (frame 0, 1, 2...), use actual timestamps. "At 0:04, the cat jumps." The model sees `<time>0.04</time>` tokens interleaved with frame tokens.
- Dynamic FPS. Sample at 1 FPS for slow footage, 4+ FPS for action. The user or trainer chooses; M-RoPE adapts.
- Window attention in ViT. Spatial attention is windowed (local within blocks) for throughput; global attention every few layers.
- Explicit JSON output format. Trained on tool-call data: "{\"tool\": \"click\", \"coords\": [380, 220]}". Agent-ready out of the box.
- MRoPE-v2 scaling. Positions scale with max input size so a 10-minute video does not run out of frequency range.

Benchmarks: Qwen2.5-VL-72B beats GPT-4o on most video benchmarks, matches Gemini 2.0 on documents, and sets the open-model SOTA for GUI grounding (ScreenSpot: 84% accuracy vs 38% for GPT-4o).

### Qwen3-VL (November 2025)

Qwen3-VL is an incremental upgrade that consolidates rather than reinvents: larger LLM backbone (Qwen3-72B), expanded training data, improved OCR, stronger reasoning via the Qwen3 "thinking mode." The ViT and M-RoPE stay. The paper focuses on data and training improvements over architecture.

The lineage takeaway: by 2025 the Qwen-VL architecture had stabilized. Additional generations scale compute and data, not primitives.

### M-RoPE mathematically

Classical RoPE rotates a query `q` of dimension `d` by position `m` using paired coordinates:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE splits the hidden dim into three bands. Say `d = 96`. Assign 32 dims to temporal, 32 to height, 32 to width. Each band rotates by its own axis position. A patch at (t=5, h=10, w=20) gets rotations `R_t(5)`, `R_h(10)`, `R_w(20)` applied to its three bands.

Text tokens use `t = text_index, h = 0, w = 0` (or a normalized choice), keeping compatibility. Video frames use `t = frame_time, h = row, w = col`. Single images use `t = 0`.

The benefit: one position encoding handles text, image, and video without branching code or different position tables.

### Dynamic-FPS sampling logic

Given a video of duration `T` seconds and a target-tokens budget `B`:

1. Compute the maximum FPS you can afford: `fps_max = B / (T * tokens_per_frame)`.
2. Pick a target FPS from `{1, 2, 4, 8}` that satisfies `fps <= fps_max`.
3. If motion is high (optical-flow heuristic or explicit user request), pick higher FPS. If motion is low, pick lower.
4. Sample uniformly at the chosen FPS; insert `<time>t</time>` tokens between frames.

Qwen2.5-VL trains this logic implicitly; at inference the user controls via `fps` parameter. A 60-second action sequence at 4 FPS with 81 tokens per frame = 19440 tokens, manageable in a 32k context.

### Structured agent output

Qwen2.5-VL's agent training explicitly targets structured tool calls:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Parsing is deterministic: JSON.parse over the model's output. Compare to free-form "click at (1024, 512)" which required regex and ambiguity handling. The shift is why Qwen2.5-VL's ScreenSpot scores jumped from Qwen2-VL's 55% to 84%.

## Use It

`code/main.py` implements:

- M-RoPE position computation for a packed sequence mixing text, image patches, and video frames.
- Dynamic-FPS sampler: given (duration, budget, motion_level), pick FPS and emit frame timestamps.
- A toy Qwen2.5-VL JSON-output parser that handles tool-call responses with coordinate fields.

Run it, then feel the difference when you swap fixed-FPS for dynamic-FPS on a 5-minute video.

## Ship It

This lesson produces `outputs/skill-qwen-vl-pipeline-designer.md`. Given a video task (monitoring, agent, action recognition, accessibility), it emits the Qwen2.5-VL configuration (frame budget, FPS strategy, window-attention flag, agent-output mode) and a latency estimate. Use this whenever you deploy a Qwen-VL-family model for a video product.

## Exercises

1. Compute M-RoPE rotations for a patch at (t=3, h=5, w=7) with hidden 48 (16 per band, base theta 10000). Show the rotation angles for the first three pairs in each band.

2. A 10-minute security-camera recording at 1 FPS produces how many frames? At 384 resolution with 3× pool, how many total tokens? Does Qwen2.5-VL's default 32k context handle it?

3. Pick FPS for a 30-second tennis rally vs a 30-second recipe demo vs a 30-second UI-agent recording. Justify each with the dynamic-FPS logic.

4. Qwen2.5-VL drops the Q-Former entirely. Why does a simple MLP work in 2025 but not in 2023? (Hint: data scale and encoder quality.)

5. Parse three Qwen2.5-VL JSON tool-call outputs into Python dicts. What fails for malformed JSON and what recovery strategy does the Qwen cookbook recommend?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | 3D rotary position embedding with temporal, height, and width bands in the hidden dim |
| Dynamic FPS | "Smart sampling" | Frame sampling rate chosen per video based on motion, duration, and token budget |
| Absolute time token | "Timestamp token" | `<time>t</time>` interleaved in the sequence so the model sees actual seconds not frame index |
| Window attention | "Local attention" | Spatial self-attention restricted to small windows for speed; global attention added periodically |
| Structured agent output | "JSON mode" | Training data supervision teaching the VLM to emit parseable JSON with coords and tool names |
| min_pixels / max_pixels | "Resolution bounds" | Per-request Qwen2.5-VL controls bounding total pixel count and therefore token count |
| Grounding | "Point-at-it" | Outputting bounding-box coordinates as text tokens; used since Qwen-VL v1 |

## Further Reading

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps)

---

## Part 2 (ch231): InternVL3: Native Multimodal Pretraining

> Every open VLM before InternVL3 followed the same three-step recipe: take a text LLM trained on trillions of text tokens, bolt on a vision encoder, then fine-tune the seams. This works but has alignment debt — the text LLM has spent its full pretraining budget on pure text and does not natively understand visual tokens. When you add vision post-hoc, the LLM has to re-learn how to relate visual input to its text reasoning without forgetting the text. InternVL3 (Zhu et al., April 2025) rejects the post-hoc approach: one pretraining run, text and multimodal interleaved from step one. The result matches Gemini 2.5 Pro on MMMU-Pro at 78B params open. This lesson reads the case for native pretraining and what changes when you make it.

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## Learning Objectives

- Explain why post-hoc VLM training accumulates alignment debt, citing the three measurable symptoms (catastrophic forgetting, answer drift, visual-text inconsistency).
- Describe InternVL3's native pretraining corpus mix and why the ratio of text : interleaved : caption matters.
- Compare V2PE (variable visual position encoding) to Qwen2-VL's M-RoPE.
- Name the Visual Resolution Router (ViR) and Decoupled Vision-Language (DvD) deployment optimizations.

## The Problem

Post-hoc VLM training is the default. LLaVA, BLIP-2, Qwen-VL, Idefics — all take an already-pretrained LLM (Llama, Vicuna, Qwen, Mistral) and add vision. The training stages typically look like:

1. Frozen LLM + frozen vision encoder + trainable projector, trained on caption pairs to align embeddings.
2. Unfreeze LLM, train on instruction data (LLaVA-Instruct, ShareGPT4V).
3. Optional task-specific fine-tune.

Three symptoms of alignment debt show up:

- Catastrophic forgetting. The post-hoc VLM forgets text-only skills. GSM8K scores drop 5-10 points. Hellaswag scores drop. Pure-text agents regress.
- Answer drift. Small phrasings of the same visual question get different answers. The vision encoder connects to the LLM with weaker bindings than the LLM's own tokens.
- Visual-text inconsistency. The VLM can describe an image correctly and then answer a question contradicting its own description. Visual tokens do not participate in the LLM's internal consistency checks the same way text does.

These symptoms are well-documented. MM1.5 Section 4 quantifies them. LLaVA-OneVision's ablations hint at them. Native pretraining is the answer.

## The Concept

### Native multimodal pretraining

InternVL3 trains from scratch on a corpus that is native multimodal from step one. The mix is:

- 40% text-only data (FineWeb, Proof-Pile-2, etc.)
- 35% interleaved image-text data (OBELICS, MMC4-style)
- 20% paired image-caption data
- 5% video-text data

Vision tokens, text tokens, and cross-modal interactions all participate in the same loss from the first gradient step. No alignment pretraining, no projector freezing stage, no catastrophic forgetting to recover from.

Training is a single stage for the base model. Instruction tuning follows, but the base model already understands visual tokens as first-class citizens.

### V2PE (variable visual position encoding)

Qwen2-VL uses M-RoPE with fixed axis allocation. InternVL3 introduces V2PE: the position encoding varies per modality type (text, image, video) with learnable scaling. In practice:

- Text tokens get 1D position (text index).
- Image patches get 2D position (row, col).
- Video frames get 3D position (time, row, col).

The three share the same RoPE frequency base, but the hidden-dim allocation per band is a learned parameter rather than a fixed split. Freedom to trade off temporal vs spatial frequency resolution during pretraining.

V2PE's ablation claim: 1-2 points on video benchmarks over M-RoPE at the same compute. Not a revolution, but cleaner.

### Visual Resolution Router (ViR)

Deployment optimization. Not all images need full-resolution encoding. A photo with one object at low detail wastes tokens when encoded at 1280px native. ViR is a small classifier that predicts the minimum resolution needed to answer the question, before encoding.

The routing has three tiers: low-res (256 tokens), medium (576), high (2048+). For 60% of queries in production traffic, low or medium is sufficient. Net effect: 2-3× throughput at equal quality.

### Decoupled Vision-Language deployment (DvD)

When you serve a large VLM, the vision encoder runs once per image but the LLM runs autoregressively for every output token. The two components have different bottlenecks (vision = GPU memory bandwidth for conv + attention; LLM = KV cache). DvD splits them onto separate GPUs with streaming between.

For an 8B + 400M encoder model, DvD roughly doubles per-node throughput vs co-located.

### Single-stage vs multi-stage quality

InternVL3's primary benchmark claim: at 78B params, match Gemini 2.5 Pro's MMMU-Pro. At 38B, match GPT-4o. At 8B, lead the open-8B leaderboard. All on a single-stage pretrain + instruction-tune recipe.

The alignment-debt hypothesis is measurable: InternVL3-8B loses fewer text-benchmark points (MMLU, GSM8K) than Qwen2.5-VL-7B per unit of vision-benchmark gain. The model is more of a generalist because training was one piece, not two.

### InternVL3.5 and InternVL-U

InternVL3.5 (August 2025) scales the recipe. Same native-pretrain approach, more data, more params. MMMU improvements are incremental.

InternVL-U (2026) adds unified generation — image output via MMDiT heads on top of the same backbone. The "U" stands for "Understanding + generation," chasing Transfusion-style unified models (Lesson 12.13). The same native-pretrain backbone supports both understanding and generation heads.

### Trade-offs of native pretraining

Native pretraining is not free:

- Compute. Training a new VLM from scratch costs the same as training a text LLM — millions of GPU-hours. Post-hoc adaptation reuses existing LLM weights, saves most of the cost.
- Data. Interleaved image-text corpora at scale are rare. OBELICS is 141M documents; MMC4 is 571M. Text alone ships at 15T tokens. Multimodal pretraining data scarcity is a hard constraint.
- Base-LLM reuse. Native pretraining gives up the option to drop in a new LLM later. Post-hoc lets you swap Llama-3.1 for Llama-4 by retraining only the adapter.

The bet InternVL3 makes: the alignment debt is worse than the reuse loss. The benchmarks back the claim. The cost-to-produce bars future labs from cheaply replicating. Post-hoc VLMs will keep existing because they remain cheaper for most projects.

## Use It

`code/main.py` is a training-corpus mixer and ViR router simulator. It:

- Takes a target corpus mix (%text, %interleaved, %caption, %video) and computes expected steps per modality.
- Simulates ViR routing on a batch of queries (distribution: 50% low-detail, 30% medium, 20% high-detail) and reports average token count.
- Reports DvD throughput estimates given encoder vs LLM FLOPs.
- Prints a side-by-side of post-hoc vs native pretraining in params, compute, data, and expected alignment-debt symptoms.

## Ship It

This lesson produces `outputs/skill-native-vs-posthoc-auditor.md`. Given a proposed VLM training plan, it audits whether to go native or post-hoc, flags alignment-debt risk, and recommends a corpus mix. Use it when you are sizing a new open-VLM project and need to pick the training strategy.

## Exercises

1. Estimate the compute delta between InternVL3-8B (native pretrain) and LLaVA-OneVision-7B (post-hoc). Ratio of GPU-hours approximately? What explains the gap?

2. InternVL3 reports 40% text / 35% interleaved / 20% caption / 5% video. If your target task is video-heavy, propose a new ratio and argue why the base model still needs substantial text and caption data.

3. Read MM1.5 Section 4 on forgetting. Name the exact benchmark where post-hoc training showed the largest regression. How much did the regression cost?

4. ViR routes 60% of traffic to low-resolution encoding. What kinds of queries does it misroute (sends to low-res when high-res was needed)? Propose three router-failure modes.

5. DvD splits vision and LLM onto separate GPUs. Under what traffic pattern does DvD hurt throughput instead of helping?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Native multimodal pretraining | "From scratch together" | Text + image + video tokens participate in the loss from step 1, not bolted on later |
| Alignment debt | "Post-hoc penalty" | Measurable regression in text skills and answer consistency that comes from bolting vision onto a frozen LLM |
| V2PE | "Variable visual pos encoding" | Per-modality learnable position encoding allocation; InternVL3's M-RoPE successor |
| ViR | "Resolution router" | Small classifier that picks minimum resolution needed per query before encoding, saving inference tokens |
| DvD | "Decoupled deployment" | Vision encoder on one GPU, LLM on another, with stream handoff; doubles throughput for large VLMs |
| InternVL-U | "Unified understanding + generation" | 2026 follow-up that adds image-generation heads to the native-pretrain backbone |
| Interleaved corpus | "OBELICS / MMC4" | Documents with text and images in natural reading order; the raw material for native pretraining |

## Further Reading

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/10-internvl3-native-multimodal)

---

## Part 3 (ch232): Chameleon and Early-Fusion Token-Only Multimodal Models

> Every VLM we have seen so far keeps images and text separate. Visual tokens come from a vision encoder, flow into a projector, then meet text inside the LLM. The vision and text vocabularies never overlap. Chameleon (Meta, May 2024) asked: what if they did? Train a VQ-VAE that turns an image into a sequence of discrete tokens from a shared vocabulary. Every multimodal document is now one sequence — text tokens and image tokens interleaved, a single autoregressive loss. Side effect: the model can generate mixed-modality outputs — alternating text and image tokens in a single inference call. This lesson reads the early-fusion thesis and builds a toy version end to end.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- Explain why a shared vocabulary + single loss changes what the model can do.
- Describe how a VQ-VAE tokenizes an image into a discrete sequence compatible with a transformer's next-token objective.
- Name Chameleon's training-stability tricks: QK-Norm, dropout placement, LayerNorm ordering.
- Compare Chameleon vs BLIP-2's Q-Former approach and describe when each is the right choice.

## The Problem

Adapter-based VLMs (LLaVA, BLIP-2, Qwen-VL) treat text and image as two different things. A text token goes through `embed(text_token)`; an image goes through `visual_encoder(image) → projector → ... pseudo_tokens`. The model has two input paths that merge partway in.

Three consequences:

1. The LLM can only consume images, not emit them. Output is text only.
2. Mixed-modality documents (alternating paragraphs and images, as in an article) are awkward — you either parse the multimodal input outside the model or chain generations.
3. Distributional mismatch. Visual tokens and text tokens live in different regions of the hidden space, creating subtle alignment issues.

Chameleon rejects the premise: images are just sequences of discrete tokens from a shared vocabulary. Train the model on interleaved documents, one loss, one autoregressive decoder, and you unlock mixed-modality generation for free.

## The Concept

### VQ-VAE as image tokenizer

The tokenizer is a vector-quantized variational autoencoder. The architecture:

- Encoder: CNN + ViT that maps image to a spatial feature map, say 32×32 features of dim 256.
- Codebook: a learned vocabulary of K vectors (Chameleon uses 8192), also dim 256.
- Quantization: for each spatial feature, look up the nearest codebook entry by L2 distance. Replace the continuous feature with the integer index.
- Decoder: CNN that takes quantized features back to pixels.

Training: VAE reconstruction loss + commitment loss + codebook loss. The codebook indices form a discrete alphabet for images.

For Chameleon: one image becomes 32*32 = 1024 tokens drawn from a vocabulary of 8192. Concatenate with text tokens (from the LLM's BPE vocabulary, say 32000). Final vocabulary: 40192. The transformer sees one sequence, one loss.

### The shared vocabulary

Chameleon's vocabulary combines text tokens, image tokens, and modality separators. Each token has a single ID. The input embedding layer maps every ID to a D-dim hidden vector. The output projection maps hidden back to vocab logits. Softmax picks the next token, whatever modality.

Separators matter: `<image>` and `</image>` tags bracket the image-token sequence. At generation time, if the model emits `<image>`, downstream software knows the next 1024 tokens are VQ indices to send to the decoder for pixel rendering.

### Mixed-modality generation

Inference is next-token prediction in the shared vocabulary. Example prompt: "Draw a cat and describe it." Chameleon emits:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

The model picks the order autonomously — it may produce image then text, text then image, or interleave. Same decoder, same loss.

Compare to adapter VLMs where generation is text-only. Chameleon reopens the question of model output modalities.

### Training stability — QK-Norm, dropout, LayerNorm ordering

Early-fusion training is unstable at scale. Chameleon's paper documents three tricks:

- QK-Norm. Apply LayerNorm to the query and key projections inside attention, before the dot product. Prevents logit magnitude explosion at depth. Used by multiple post-2024 large models.
- Dropout placement. Dropout after every residual-add, not just after attention and MLP. More regularization required when gradients from image tokens can dominate.
- LayerNorm ordering. Pre-LN on the residual branch (standard), plus an extra LN on the skip connection of the last block. Stabilizes final-layer gradient flow.

Without these tricks, 34B-param Chameleon training diverged at multiple checkpoints. With them, it converges. The training recipe is as much of the contribution as the architecture.

### The tokenizer's reconstruction ceiling

VQ-VAE is lossy. At 8192 codebook entries and 1024 tokens per 512×512 image, reconstruction PSNR caps around 26-28 dB. This is enough for recognizable image gen but visibly worse than continuous-space diffusion (Stable Diffusion 3 achieves 32+ dB).

The tokenizer is the bottleneck. Better tokenizers (MAGVIT-v2, IBQ, SBER-MoVQGAN) lift the ceiling. Emu3 (Lesson 12.12) achieves SDXL-quality generation via a better tokenizer alone.

### Chameleon vs BLIP-2 / LLaVA

Chameleon (early fusion, shared vocab):
- One loss, one decoder.
- Generates mixed-modality output.
- Tokenizer is the quality ceiling.
- Expensive: VQ-VAE decoder per generated image on inference path.

BLIP-2 / LLaVA (late fusion, separate towers):
- Vision in, text out only.
- Reuses pretrained LLM.
- No tokenizer bottleneck for understanding.
- Cheap: single forward pass.

Pick by task. If you need image generation, Chameleon family. If you only need understanding, adapter-VLM is simpler and reuses more pretrained compute.

### Fuyu and AnyGPT

Fuyu (Adept, 2023) is a related approach: skip the separate vision encoder entirely, feed raw image patches through the LLM's input projection as if they were tokens, no tokenizer. Simpler than Chameleon, loses the shared-vocab output generation.

AnyGPT (Zhan et al., 2024) extends Chameleon to four modalities: text, image, speech, music. Same VQ-VAE trick for each, shared transformer. Any-to-any generation. Covered more in Lesson 12.16.

## Use It

`code/main.py` builds a toy end-to-end early-fusion model:

- A tiny VQ-VAE-style quantizer that maps 8×8 patches to codebook indices (K=16).
- A shared vocabulary of (text ids 0..31) + (image ids 32..47) + (separators 48, 49).
- A toy autoregressive decoder (bigram table) trained on synthetic captions + image-token sequences.
- Sampling loop that emits alternating text + image tokens given a prompt.

The code intentionally keeps the transformer tiny (bigrams) so you can trace the signal flow end to end.

## Ship It

This lesson produces `outputs/skill-tokenizer-vs-adapter-picker.md`. Given a product spec (understand only vs understand + generate, required image quality, cost budget), it picks between Chameleon-family (early fusion) and LLaVA-family (late fusion) and justifies with quantitative rules of thumb.

## Exercises

1. Chameleon uses K=8192 codebook entries and 1024 tokens per 512×512 image. Estimate the compression ratio vs a 24-bit RGB image. Is it lossy? How lossy?

2. A 4K image (3840×2160) at the same VQ-VAE density produces how many image tokens? Can a Chameleon-style model generate a 4K image in one inference call? What breaks first — context, tokenizer quality, or KV cache?

3. Implement QK-Norm in pure Python. Given a 64-dim query and key, show the dot product before and after LayerNorm. Why is magnitude control important at depth?

4. Read Chameleon Section 2.3 on training stability. Describe the exact failure mode the paper observed at 34B without QK-Norm. What was the "norm explosion" signature?

5. Extend the toy decoder to emit a mixed-modality response given a text-only prompt. Measure how often the model picks image-first vs text-first given training-data distribution 60% text-first / 40% image-first.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Early fusion | "Unified tokens" | Images converted to discrete tokens sharing the transformer's vocabulary from step one |
| VQ-VAE | "Image tokenizer" | CNN + ViT + codebook that maps images to integer indices the transformer can predict |
| Shared vocabulary | "One dictionary" | A single token ID space covering text + image + modality separators |
| QK-Norm | "Attention stabilizer" | LayerNorm applied to query and key before their dot product, prevents norm blowup |
| Mixed-modality generation | "Text + image output" | Inference that autonomously produces interleaved text and image tokens in one pass |
| Codebook size | "K entries" | Number of discrete vectors the VQ-VAE can quantize to; trades compression for fidelity |
| Tokenizer ceiling | "Reconstruction limit" | Best PSNR achievable by decoding VQ tokens; bounds the model's image quality |

## Further Reading

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/11-chameleon-early-fusion-tokens)

---

## Part 4 (ch233): Emu3: Next-Token Prediction for Image and Video Generation

> BAAI's Emu3 (Wang et al., September 2024) is the 2024 result that should have ended the diffusion-versus-autoregressive debate. A single Llama-style decoder-only transformer, trained only on the next-token-prediction objective, across a unified vocabulary of text + VQ image tokens + 3D VQ video tokens, beats SDXL on image generation and LLaVA-1.6 on perception. No CLIP loss. No diffusion schedule. Classifier-free guidance is used at inference for quality, but the core training objective is next-token prediction with teacher forcing. Published in Nature. This lesson reads the Emu3 thesis — why a better tokenizer plus scale is all you need — and contrasts with diffusion approaches.

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## Learning Objectives

- Explain why Emu3's single-loss next-token objective works despite the long-held assumption that diffusion is required for image quality.
- Describe the 3D video tokenizer: what a spatiotemporal VQ codebook looks like, why patches span time.
- Compare Emu3 vs Stable Diffusion XL on (training compute, inference cost, quality ceiling).
- Name the three roles the same Emu3 model plays: Emu3-Gen (image gen), Emu3-Chat (perception), Emu3-Stage2 (video gen).

## The Problem

The conventional wisdom through 2024: image generation needs diffusion. The argument: discrete image tokens lose too much information to reconstruct detail, and autoregressive sampling accumulates error across thousands of tokens. Stable Diffusion, DALL-E 3, Imagen, Midjourney all use some form of diffusion. Chameleon (Lesson 12.11) partially disproved this at small scale but did not match SDXL on quality.

Emu3 attacked the argument head-on. The claim: better visual tokenizer + enough scale + next-token loss = diffusion-beating image generation in the same model that also does perception.

The bet was controversial when it published. Two years on, the open-source unified-generation family (Emu3, Show-o, Janus-Pro, Transfusion) is the default path for research; production frontier models appear to use some variant.

## The Concept

### The Emu3 tokenizer

The key ingredient is the visual tokenizer. Emu3 trains a custom IBQ-class tokenizer (Inverse Bottleneck Quantizer, SBER-MoVQGAN family) at 8×8 resolution-reduction per token. A 512×512 image becomes 64×64 = 4096 tokens at codebook size 32768.

This is larger than Chameleon's 1024 tokens per 512×512 at K=8192 but cheaper per token (smaller codebook lookups, simpler codec). The key metric: reconstruction PSNR at 30.5 dB, competitive with Stable Diffusion's continuous latent space at 32 dB.

For video: a 3D VQ tokenizer encodes a spatiotemporal patch (4×4×4 pixels) to one integer. A 4s clip at 8 FPS has 32 frames; at 256×256 with 4× spatial and 4× temporal reduction, the token count is (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokens.

Tokenizer quality is the ceiling. Emu3's contribution is partly "we trained a very good tokenizer."

### Single-loss training

Emu3 uses one objective: next-token prediction on a shared vocabulary across text tokens, 2D image tokens, and 3D video tokens. Weights are multiplied by modality-specific factors during training to balance contribution, but the loss function is identical.

Train on a mix of:
- Image gen: `<text caption> <image> image_tokens </image>`
- Image perception: `<image> image_tokens </image> <question> text_tokens`
- Video gen: `<text caption> <video> video_tokens </video>`
- Video perception: analogous.
- Text only: standard NTP.

The model learns when to emit image tokens vs text tokens from the data distribution. Generation emerges from the model predicting image tokens after the `<image>` tag.

### Classifier-free guidance and temperature

Autoregressive image generation gets much better with classifier-free guidance (CFG) at inference. Emu3 uses it: generate twice, once with the full caption, once with an empty caption, mix the logits with a guidance weight (typical 3.0-7.0). This is the same CFG trick diffusion uses, borrowed to the autoregressive setting.

Temperature matters: too high, artifacts; too low, mode collapse. Emu3's recommended temperature is 1.0 for perception, 0.8 for image generation.

### Three roles, one model

Emu3 ships as three functionally distinct APIs but one underlying weight set:

- Emu3-Gen. Image generation. Input text, output image tokens.
- Emu3-Chat. VQA and captioning. Input image (tokens), output text.
- Emu3-Stage2. Video generation and video VQA. Input text or video, output text or video.

No task-specific heads. Just different prompt templates. Same checkpoint.

### Benchmarks

From Emu3 paper (September 2024):

- Image generation: beats SDXL on MJHQ-30K FID (5.4 vs 5.6), GenEval overall (0.54 vs 0.55 — statistical tie), and Deep-Eval's composite on-par.
- Image perception: beats LLaVA-1.6 on VQAv2 (75.1 vs 72.4) and roughly matches on MMMU.
- Video generation: 4-second-clip quality at competitive FVD with Sora-era publicly benchmarked models.

The numbers are not always winning — Emu3 trades a point here for a point there — but the claim "next-token prediction is all you need" is defensible across modalities.

### Compute cost

Emu3 was trained on ~300 billion multimodal tokens with a 7B-parameter model. GPU-hours roughly comparable to Llama-2-7B pretraining (2k-4k GPU-years on A100-class silicon). Diffusion models like Stable Diffusion 3 train in similar budgets but need separate text encoders and more complex pipelines.

At inference, Emu3 is slower than SDXL per image: 4096 image tokens at 30 tok/s is ~2 minutes per 512×512 image, vs 2-5 seconds for SDXL. Speculative decoding and KV-cache optimization narrow the gap but do not close it. Autoregressive image gen is compute-heavy; this is the standing trade-off.

### Why it matters

Emu3's deep contribution is conceptual. If next-token prediction scales to match diffusion on image generation, the unified-model path (one loss, one backbone, any modality) is viable. Future models do not need separate text encoders, separate diffusion schedulers, separate VAEs. One transformer, one tokenizer per modality, scale.

Show-o, Janus-Pro, and InternVL-U all build on or challenge this thesis. Chinese labs (BAAI, DeepSeek) publish more aggressively in this direction than US labs through 2025.

## Use It

`code/main.py` builds two toy pieces:

- A 2D vs 3D VQ tokenizer count calculator: given (resolution, patch, clip_length, FPS), compute token counts for image vs video.
- An autoregressive image-token sampler with classifier-free guidance at temperature.

The CFG implementation matches Emu3's recipe — mix conditional and unconditional logits with a guidance weight.

## Ship It

This lesson produces `outputs/skill-token-gen-cost-analyzer.md`. Given a generation product spec (image or video, target resolution, quality tier, latency budget), it computes token counts, inference cost, and picks Emu3-family vs diffusion.

## Exercises

1. Emu3 produces 4096 tokens per 512×512 image at 8×8 reduction. Compute the equivalent for 1024×1024 and 2048×2048. What happens to inference latency?

2. Read Emu3 Section 3.3 on the video tokenizer. Describe the 3D VQ patch shape and why it is 4×4×4 not 8×8×1.

3. Classifier-free guidance weight 5.0 vs 3.0: what visual effect? Trace the math in `code/main.py`.

4. Compute training FLOPs for Emu3-7B at 300B tokens and compare to Stable Diffusion 3. Which was more expensive to train?

5. Emu3 beats SDXL on FID but not on VQAv2 vs specialized VLMs. Explain why the unified-loss approach shows different strengths vs specialists on different benchmarks.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | Standard autoregressive loss: predict token[i+1] given token[0..i]; works for every modality when tokenized |
| IBQ tokenizer | "Inverse bottleneck quantizer" | A class of VQ-VAE with larger codebooks (32768+) and better reconstruction than Chameleon's |
| 3D VQ | "Spatiotemporal quantizer" | Codebook indexed by (time, row, col); one token covers a 4×4×4 pixel cube |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional logits with weight gamma; boosts image quality at inference |
| Unified vocabulary | "Shared tokens" | Text + image + video all draw from the same integer space; model predicts whichever modality comes next |
| MJHQ-30K | "Image gen benchmark" | Midjourney-quality benchmark with 30k prompts; Emu3 reports FID here |

## Further Reading

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/12-emu3-next-token-for-generation)
