# Embodied VLAs, Documents & Multimodal RAG/Agents

> Combined lessons (6 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch242): Embodied VLAs: RT-2, OpenVLA, π0, GR00T

> The first time a model read a recipe off a website and executed it in a kitchen robot was RT-2 (Google DeepMind, July 2023). RT-2 discretized actions as text tokens, co-fine-tuned a VLM on web data plus robot-action data, and proved that web-scale vision-language knowledge transfers to robotic control. OpenVLA (June 2024) shipped the open 7B reference. Physical Intelligence's π0 series (2024-2025) added flow-matching action experts. NVIDIA's GR00T N1 (March 2025) delivered dual-system (System 1 / System 2) control for humanoid robots at scale. The VLA primitive — vision-language-action, a single model that sees, reads, and acts — is the bridge between this phase's understanding models and the autonomous systems in Phase 15.

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## Learning Objectives

- Describe action tokenization: discrete bin encoding (RT-2), FAST efficient action tokens, continuous flow-matching actions (π0).
- Explain why co-fine-tuning on web + robot data preserves general-knowledge transfer to novel tasks.
- Compare OpenVLA (open 7B Llama+VLM), π0 (flow-matching), and GR00T N1 (dual-system) on the same robot task.
- Name the Open X-Embodiment dataset and its role as the RT-X training corpus.

## The Problem

A robot that does chores from natural language instructions has been a research target since the 1970s. The 2020s answer: a vision-language-action (VLA) model. Same VLM architecture used for VQA, but output is actions (joint torques, end-effector poses, discrete commands) instead of text.

Challenges specific to VLAs:

1. Action spaces are continuous (joint angles, forces) and high-dimensional (7-DOF arm + 3-DOF gripper = 10 dims at 30 Hz).
2. Robot-specific training data is scarce. Open X-Embodiment has ~1M trajectories; web text-image is 5B+.
3. Control frequency matters. 30 Hz control loop means 33ms budget per action.
4. Safety. A wrong action damages hardware, humans, or property.

## The Concept

### Action tokenization (RT-2)

RT-2's trick: represent each joint target as a quantized text token. Discretize the normalized [-1, 1] range into 256 bins, map each bin to a vocabulary ID. A 10-DOF action becomes 10 tokens at each control step.

Co-fine-tune a PaLM-X VLM on a mixture:

- Web image-text pairs (captioning, VQA).
- Robot demonstrations, action as tokens.

The model sees "pick up the red cube" (language) → image (vision) → 10-token action sequence (discretized joint targets). Web pretraining preserves general-knowledge transfer: RT-2 can follow "move towards the fast-moving object" even though "fast-moving" isn't in training data.

Inference at 3-5 Hz in the RT-2 paper, limited by VLM autoregressive decode.

### OpenVLA — the open 7B reference

OpenVLA (Kim et al., June 2024) is the open-weights RT-2 equivalent. 7B Llama backbone, DINOv2 + SigLIP dual vision encoder, action tokenization over 256 bins.

Trained on Open X-Embodiment (970k trajectories across 22 robots). Ships with LoRA fine-tuning support for adapting to new robots.

Inference: 4-5 Hz on an A100 with quantization. Fast enough for slow manipulation, not for high-frequency control.

### FAST tokenizer — faster action decode

Pertsch et al. (2024) showed that discrete-bin tokenization is inefficient — most actions cluster in a small region of bin-space. FAST (Frequency-domain Action Sequence Tokenizer) compresses action sequences via DCT and quantizes the coefficients.

A 30-step action trajectory becomes ~10 FAST tokens instead of 300 discrete-bin tokens. Inference speeds up 3-5× without quality loss.

### π0 and flow-matching actions

Physical Intelligence's π0 (Black et al., October 2024) replaces discrete action tokens with a flow-matching action expert:

- A small action transformer reads the VLM's hidden states and outputs a continuous 50-step action sequence via rectified flow.
- The action head trains with flow-matching loss; VLM pretraining stays unchanged.
- Inference: full action sequence emitted in ~5 denoising steps, effectively 50 Hz control.

π0's claim: beats OpenVLA and Octo on a wide suite of manipulation tasks. The continuous-action formulation preserves smoothness that discretization destroys.

π0.5 and π0-FAST are incremental upgrades. π0-FAST combines FAST tokenization with flow matching.

### GR00T N1 — dual-system for humanoids

NVIDIA's GR00T N1 (March 2025) is built for humanoid robots (>30 DOF, full-body):

- System 2: a large VLM reading scene + instruction, producing high-level subgoals at ~1 Hz.
- System 1: a small action-head transformer producing low-level 50-100 Hz joint commands conditioned on the subgoals.

The split maps to Kahneman's fast-and-slow thinking: System 2 plans, System 1 acts. Benefits: slow VLM-sized planning does not block fast control; System 1 stays small for latency.

GR00T N1.7 (late 2025) improves data scaling. GR00T fine-tunes with sim-to-real data from Omniverse.

### Open X-Embodiment

The training data. RT-X (October 2023) assembled 22 datasets covering 1M trajectories across 22 robots. Open X-Embodiment is the corpus everyone uses:

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table.
- Each sample: (robot state, camera views, instruction, action sequence).
- Training hygiene: unify action space, normalize joint ranges, resize cameras.

OpenVLA and π0 train on Open X-Embodiment. Domain gap to any specific robot is closed by LoRA fine-tuning on 100-1000 task-specific demos.

### Co-fine-tuning vs robot-only

Co-fine-tuning mixes web VQA data with robot trajectories. The ratio matters: too much VQA and the model forgets actions; too much robot data and the model loses general knowledge.

RT-2's ratio: ~1:1. OpenVLA: ~0.5:1 web-to-robot. π0: similar. The precise ratio is a hyperparameter to tune per dataset size.

Robot-only training produces task-specific models that fail on out-of-distribution instructions. Co-fine-tuning is the difference between "pick up the red cube (in demo)" and "pick up the third largest object from the left (novel phrasing)."

### Safety and action limits

Every production VLA ships with:

- Hard joint limits (can't torque past spec).
- Velocity limits (soft clipping).
- Workspace bounds (end-effector cannot leave the table).
- Human-in-the-loop approval for novel tasks.

These sit outside the VLA as control-layer checks. The VLA's output is a suggestion, not a command.

## Use It

`code/main.py`:

- Implements 256-bin action tokenization and de-tokenization.
- Sketches a FAST tokenizer based on DCT + quantization.
- Compares token-count per action step across (discrete-bin, FAST, continuous-flow).
- Prints a lineage summary of RT-2 → OpenVLA → π0 → GR00T.

## Ship It

This lesson produces `outputs/skill-vla-action-format-picker.md`. Given a robot task (manipulation, navigation, humanoid whole-body), picks between discrete-bin + RT-2, FAST + OpenVLA, flow-matching + π0, or dual-system + GR00T.

## Exercises

1. A 10-DOF arm at 30 Hz control rate. Discrete-bin tokenization at 256 bins emits how many tokens per second? Can a 7B VLM keep up?

2. FAST tokenization compresses 30-step trajectories to ~10 tokens. What does the user lose if the trajectory has high-frequency motion (e.g., drumming)?

3. π0's flow-matching head denoises in ~5 steps. Compare throughput to OpenVLA's autoregressive decode at 4-5 Hz.

4. GR00T's System 1 / System 2 split maps to Kahneman. Propose a different split (System 3?) that might help bipedal walking.

5. Read Open X-Embodiment Section 4 on dataset curation. Name the three curation rules that prevent domain leakage.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | Model that takes image + instruction and outputs action commands |
| Action tokenization | "Discrete bins" | Quantize continuous joint targets into 256 bins per dim, each a vocab ID |
| FAST tokenizer | "Frequency action tokens" | DCT + quantize to compress 30-step trajectories to ~10 tokens |
| Co-fine-tune | "Mix web + robot" | Train on web VQA data alongside robot demos to preserve general knowledge |
| Flow-matching action head | "π0 continuous output" | Small transformer that outputs a 50-step action sequence via rectified flow |
| System 1 / System 2 | "Dual-system control" | Large VLM plans slowly, small action head acts quickly; GR00T pattern |
| Open X-Embodiment | "RT-X dataset" | 1M-trajectory cross-robot dataset; the training corpus |

## Further Reading

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot)

---

## Part 2 (ch243): Document and Diagram Understanding

> Documents are not photos. A PDF, scientific paper, invoice, or handwritten form has layout, tables, diagrams, footnotes, headers, and semantic structure that plain image understanding cannot capture. The pre-VLM stack was a pipeline: Tesseract OCR + LayoutLMv3 + table-extraction heuristics. The VLM wave replaced that with OCR-free models — Donut (2022), Nougat (2023), DocLLM (2023) — that emit structured markup directly. By 2026 the frontier is just "feed the page image to Claude Opus 4.7 at 2576px native," and the structured-markup output comes for free. This lesson reads the three-era arc of document AI.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## Learning Objectives

- Explain the three eras of document AI: OCR pipeline, OCR-free, VLM-native.
- Describe LayoutLMv3's three input streams: text, layout (bbox), image patches, with unified masking.
- Compare Donut (OCR-free, image → markup), Nougat (scientific paper → LaTeX), DocLLM (layout-aware generative), PaliGemma 2 (VLM-native).
- Pick a document model for a new task (invoices, scientific papers, handwritten forms, Chinese receipts).

## The Problem

"Understand this PDF" is deceptively hard. The information sits in:

- Text content (90% of the signal).
- Layout (headers, footnotes, sidebars, two-column format).
- Tables (rows, columns, merged cells).
- Figures and diagrams.
- Handwritten annotations.
- Fonts and typography (title vs body).

Raw OCR dumps the text and loses the rest. A system that cares about invoices needs to know "Total: $1,245" came from the bottom-right, not from a footnote.

## The Concept

### Era 1 — OCR pipeline (pre-2021)

The classic stack:

1. PDF → image per page.
2. Tesseract (or commercial OCR) extracts text with per-word bounding boxes.
3. Layout analyzer identifies blocks (header, table, paragraph).
4. Table structure recognizer parses tables.
5. Domain rules + regex extract fields.

Works for clean printed text. Breaks on handwriting, skewed scans, complex tables, non-English scripts. Every failure mode requires a custom exception path.

### TrOCR (2021)

TrOCR (Li et al., arXiv:2109.10282) replaced Tesseract's classic CNN-CTC with a transformer encoder-decoder trained on synthetic + real text images. Clean win on handwritten and multilingual text. Still a pipeline (detector then TrOCR then layout), but the OCR step improved dramatically.

### Era 2 — OCR-free (2022-2023)

The first OCR-free models said: skip detection entirely, map image pixels to structured output directly.

Donut (Kim et al., arXiv:2111.15664):
- Encoder-decoder transformer, encoder is Swin-B.
- Output is JSON for form understanding, markdown for summarization, or any task-specific schema.
- No OCR, no layout, no detection.

Nougat (Blecher et al., arXiv:2308.13418):
- Trained specifically on scientific papers.
- Output is LaTeX / markdown.
- Handles equations, multi-column layout, figures.
- The model every arXiv-parser calls.

These are specialists, not generalists. Donut on a scientific paper fails; Nougat on an invoice fails.

### LayoutLMv3 (2022)

A different track. LayoutLMv3 (Huang et al., arXiv:2204.08387) keeps OCR but adds layout understanding:

- Three input streams: OCR text tokens, per-token 2D bounding boxes, image patches.
- Masked training objective across all three modalities (masked text, masked patches, masked layout).
- Downstream: classification, entity extraction, table QA.

LayoutLMv3 is the peak of OCR-based document understanding. Strong on forms and invoices. Requires OCR upstream. Best pre-VLM accuracy on standardized document benchmarks.

### DocLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908) is LayoutLM's generative sibling. Generates free-form answers conditioned on layout tokens. Better for QA on documents; still depends on OCR input.

### Era 3 — VLM-native (2024+)

2024 VLMs became good enough to replace the pipeline entirely. Feed the full page image at high resolution to a VLM, ask the question, get an answer.

- LLaVA-NeXT 336-tile AnyRes works for small documents.
- Qwen2.5-VL dynamic-resolution handles 2048+ pixels natively.
- Claude Opus 4.7 supports 2576px documents.
- PaliGemma 2 (April 2025) trains specifically for documents + handwriting.

The gap between VLM-native and OCR-pipeline closed rapidly. By 2026, VLM-native wins on:

- Scene text (hand-written + printed, mixed scripts).
- Complex tables with merged cells.
- Math equations embedded in text.
- Figures with text annotations.

OCR pipelines still win on:

- Pure-scan workloads at massive scale where per-page latency matters.
- Pipeline reliability (deterministic failures vs VLM hallucinations).
- Regulated environments requiring auditable OCR output.

### The Claude 4.7 / GPT-5 frontier

At 2576-pixel native input, frontier VLMs do document understanding at near-human accuracy. The benchmark numbers from early 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, pipelined LayoutLMv3 ~83.
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

The closed-model gap is mostly resolution and base-LLM scale. Open models at 7B are a few points behind but catching up.

### Math equations and LaTeX output

Scientific papers need exact LaTeX output for equations. Nougat was trained on this. VLMs trained with LaTeX targets (Qwen2.5-VL-Math, Nougat derivatives) produce usable LaTeX. Without explicit LaTeX training, VLMs produce readable but imprecise transcriptions.

For scientific-paper pipelines in 2026: chain Nougat on the PDF, then a VLM on tricky pages.

### Handwriting

Still the hardest sub-task. Mixed printed + handwritten (doctors' notes, filled forms) is where OCR pipelines still beat VLMs for cost. Handwritten-only VLMs are improving (Claude 4.7, PaliGemma 2).

### 2026 recipe

For a new document-AI project:

- Pure-printed invoices at scale: LayoutLMv3 + rules, cost-efficient.
- Mixed documents (scientific + handwritten + forms): VLM-native (PaliGemma 2 or Qwen2.5-VL).
- Full arXiv ingestion: Nougat for math, VLM for figures.
- Regulatory: OCR pipeline + VLM validator for cross-check.

## Use It

`code/main.py`:

- A toy layout-aware tokenizer: given (text, bbox) pairs, produces the LayoutLMv3-style input.
- A Donut-style task schema generator: JSON template for forms.
- A comparison of token budgets per page across OCR-pipeline, Donut, Nougat, and VLM-native.

## Ship It

This lesson produces `outputs/skill-document-ai-stack-picker.md`. Given a document-AI project (domain, scale, quality, regulatory), picks between OCR pipeline, OCR-free specialist, and VLM-native.

## Exercises

1. Your project is 10M invoices per day. Which stack minimizes cost-per-page without losing accuracy?

2. Why does LayoutLMv3 outperform pure-CLIP-VLMs on form QA but underperform at scene-text? What does the bbox stream give up?

3. Nougat generates LaTeX. Propose a test case where VLM-native output beats Nougat on LaTeX fidelity, and a case where Nougat wins.

4. Read PaliGemma 2 paper (Google, 2024). What was the key training-data addition that lifted document accuracy vs PaliGemma 1?

5. Design a regulatory-safe hybrid: OCR pipeline as primary, VLM as secondary cross-check. How do you resolve disagreement?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | Stage-wise stack: detect -> OCR -> layout -> rules; deterministic, fragile |
| OCR-free | "Donut-style" | Image-to-output transformer that skips explicit OCR; single model |
| Layout-aware | "LayoutLM" | Input includes per-token bbox coordinates; unified masking across modalities |
| VLM-native | "Frontier VLM" | Feed page image directly to Claude/GPT/Qwen VLM at high resolution; no pipeline |
| DocVQA | "Doc benchmark" | Document VQA standard; most-cited score |
| Markup output | "LaTeX / MD" | Structured output format instead of free-form text; enables downstream automation |

## Further Reading

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/22-document-diagram-understanding)

---

## Part 3 (ch244): ColPali and Vision-Native Document RAG

> Traditional RAG parses PDFs into text, splits into chunks, embeds chunks, stores vectors. Every step loses signal: OCR drops chart data, chunking breaks table rows, text embeddings ignore figures. ColPali (Faysse et al., July 2024) asked the simpler question: why extract text at all? Embed the page image directly via PaliGemma, use ColBERT-style late interaction for retrieval, and keep all the layout, figures, fonts, and formatting signal the document carries. Published benchmarks: 20-40% better end-to-end accuracy than text-RAG on visually-rich documents. ColQwen2, ColSmol, and VisRAG extended the pattern. This lesson reads the vision-native RAG thesis and builds a tiny ColPali-like indexer.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## Learning Objectives

- Explain the difference between bi-encoder retrieval (one vector per document) and late-interaction retrieval (many vectors per document).
- Describe ColBERT's MaxSim operation and how ColPali generalizes it from text tokens to image patches.
- Build a tiny ColPali-like indexer: page → patch embeddings → MaxSim over query-term embeddings → top-k pages.
- Compare ColPali + Qwen2.5-VL generator vs text-RAG + GPT-4 on an invoices / financial reports use case.

## The Problem

Text-RAG on PDFs throws away most of the document. A financial report's Q3 revenue growth is usually in a chart; a medical report's findings are in annotated images; a legal contract's signature block is a layout fact, not a text fact.

The text-RAG pipeline:

1. PDF → text via OCR / pdftotext.
2. Text → 300-500 token chunks.
3. Chunk → bi-encoder embedding (one vector).
4. User query → embedding → cosine similarity → top-k chunks.
5. Chunks + query → LLM.

Five lossy steps. Charts not captured. Tables broken across chunks. Multi-column layout flattens. Figure annotations disappear.

ColPali's fix: skip OCR, embed the page image directly. Use ColBERT-style late interaction for retrieval so the model can attend to fine-grained patches at query time.

## The Concept

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) is a text retrieval method. Instead of one vector per document, it produces one vector per token. At query time:

- Query tokens get their own embeddings (N_q vectors).
- Document tokens get embeddings (N_d vectors, typically cached).
- Score = sum over query tokens of max over document tokens of cosine similarity: Σ_i max_j cos(q_i, d_j).

This is the MaxSim operation. Each query token "picks" its best-matching document token. The final score is the sum.

Pros: strong recall, handles term-level semantics. Cons: N_d vectors per document, storage expensive.

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) applies the ColBERT pattern to images.

- Each page is encoded by PaliGemma (ViT + language) into patch embeddings: N_p vectors per page.
- Each user query (text) is encoded into query-token embeddings: N_q vectors.
- Score = Σ_i max_j cos(q_i, p_j), i.e., MaxSim over query-text-tokens and page-image-patches.
- Retrieve top-k pages by total score.

At document-ingestion time: embed every page with PaliGemma, store all patch embeddings. At query time: embed the query tokens, compute MaxSim against all stored page embeddings, return top-k pages.

Pros: end-to-end beats text-RAG by 20-40% on visually rich documents. Each patch-vector captures local layout and content.

Cons: N_p patches × 4-byte floats × D-dim vectors per page = storage grows fast. Mitigated by PQ / OPQ quantization.

### ColQwen2 and ColSmol

ColQwen2 (illuin-tech, 2024-2025) swaps PaliGemma for Qwen2-VL. Better base encoder, better retrieval.

ColSmol is the smaller-scale variant for local / edge use. A ColSmol retriever at ~1B params runs on consumer GPU.

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) is a different variant: instead of MaxSim on patches, pool each page into a single vector with a VLM then bi-encoder retrieve. Faster indexing + smaller storage, weaker recall.

The quality-vs-cost trade-off: ColPali for quality, VisRAG for scale.

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) extends multi-modal retrieval to multi-page multi-document reasoning. Retrieves pages across documents, composes a multi-page context for the VLM.

### ViDoRe — the benchmark

ColPali's companion benchmark. Visual Document Retrieval Evaluation. Tasks include financial reports, scientific papers, administrative documents, medical records, manuals. Metric: nDCG@5.

ColPali-v1 scores ~80% nDCG@5 on ViDoRe; text-RAG on the same documents scores ~50-60%.

### The end-to-end RAG pipeline

For a vision-native RAG:

1. Ingest: PDF → page images → PaliGemma encoding → store all patch embeddings.
2. Query: user text → query-token embeddings → MaxSim against all indexed pages → top-k pages.
3. Generate: top-k page images + query → VLM (Qwen2.5-VL or Claude) → answer.

No OCR anywhere. Figures, charts, fonts, layout all flow into the answer.

### Storage math

A 50-page financial report with 729 patches per page and 128-dim embeddings:

- ColPali: 50 * 729 * 128 * 4 bytes = ~18 MB raw, ~4 MB after PQ.
- Text-RAG: 50 chunks * 768-dim * 4 bytes = ~150 kB.

ColPali is ~30× more storage per document. At scale, OPQ / PQ brings it down to ~5-10×, usually tolerable.

### When text-RAG still wins

- Pure-text documents with no layout signal (wiki articles, chat logs). Text-RAG is simpler and storage-cheaper.
- Multi-million-page archives where storage dominates cost.
- Strict regulatory requirements demanding extractable OCR text alongside the retrieval.

For everything else in 2026 — financial reports, scientific papers, legal contracts, medical records, UX documentation — vision-native RAG wins.

## Use It

`code/main.py`:

- Toy patch encoder: maps a "page" (small grid of feature vectors) to an array of patch embeddings.
- MaxSim scorer: computes the ColBERT-style score between a query token embedding set and a page patch set.
- Indexes 5 toy pages, runs 3 queries, returns top-k with scores.

## Ship It

This lesson produces `outputs/skill-vision-rag-designer.md`. Given a document-RAG project, picks ColPali / ColQwen2 / VisRAG / text-RAG and sizes the storage.

## Exercises

1. A 200-page annual report at 729 patches per page, 128-dim emb, 4-byte floats. Compute raw storage and PQ-compressed (8×) storage.

2. MaxSim is Σ_i max_j cos(q_i, p_j). What does this sum capture that a simple mean similarity does not?

3. ColPali indexes pages as patch sets. What changes if we instead index at the word level (as ColBERT does)? Trade-offs?

4. Design the end-to-end pipeline for a 1M-page corpus with a latency budget of 500ms per query. Pick ColQwen2 / VisRAG and justify.

5. Read M3DocRAG (arXiv:2411.04952). Describe the multi-page attention pattern and how it differs from single-page ColPali retrieval.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | Retrieval using per-token or per-patch embeddings + MaxSim, not a single doc vector |
| MaxSim | "Max-over-patches" | For each query token, pick the highest-similarity document token; sum across query |
| Bi-encoder | "Single-vector" | One vector per document; faster but loses granularity |
| Multi-vector | "Many-vectors-per-doc" | Store N_p vectors per document / page; storage cost grows but recall improves |
| Patch embedding | "Page feature" | One vector per image patch from a VLM encoder, cached per page |
| ViDoRe | "Vision doc bench" | ColPali's benchmark suite for visual document retrieval |
| PQ quantization | "Product quantization" | Compression that maintains vector similarity while shrinking storage ~8× |

## Further Reading

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/23-colpali-vision-native-rag)

---

## Part 4 (ch245): Multimodal RAG and Cross-Modal Retrieval

> Vision-native document RAG is one slice. Production multimodal RAG goes wider — retrieving across text, images, audio, and video for workflows like trip planning ("find me a quiet vegan brunch with natural light"), medical triage ("what injury matches this photo + these notes"), e-commerce ("outfits similar to this selfie, in my size"), and field service ("diagnose this engine sound plus photo of the part"). Three 2025 surveys — Abootorabi et al., Mei et al., Zhao et al. — codified the sub-problems: cross-modal retrieval, retrieval fusion, generation grounding, multimodal evaluation. This lesson reads the surveys and designs a production pipeline.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 minutes

## Learning Objectives

- Design cross-modal retrieval: text → image, image → text, audio → video, etc.
- Compare three fusion strategies: score fusion, attention-based fusion, MoE fusion.
- Explain generation grounding: what "cite your sources" looks like when sources are a mix of modalities.
- Name the three canonical multimodal RAG surveys of 2025 and their sub-problem taxonomy.

## The Problem

Single-modality RAG is a solved pattern: embed query, embed chunks, retrieve, stuff into LLM. Multimodal RAG requires:

1. Multiple retrieval heads (each modality needs embeddings in a compatible space).
2. Fusion of retrieval results across modalities.
3. Generation grounding that cites sources across modalities.
4. Evaluation metrics that cover cross-modal signal.

The 2025 surveys all arrive at the same taxonomy.

## The Concept

### Cross-modal retrieval

Retrieve documents of modality B given a query of modality A. Three patterns:

1. Shared embedding space. CLIP and CLAP produce text + image / text + audio embeddings in a shared space. Cosine similarity across modalities works directly. Limited to CLIP-trained pairs.

2. Per-modality encoder + translation. Text encoder + image encoder + a small translator module mapping between spaces. Sen2Sen by Gupta et al. and other 2024 designs. Flexible but adds complexity.

3. VLM as encoder. Use a VLM's hidden states as the retrieval representation. Any modality the VLM supports works. Higher quality, more expensive.

Choice: CLIP / SigLIP 2 for text+image; CLAP for text+audio; VLM-hidden-states for cross-modal at frontier quality.

### Fusion strategies

You retrieved 10 results: 5 images, 3 text passages, 2 audio clips. How do you merge?

Score fusion (cheapest). Each modality has its own retriever, each returns scores. Normalize scores within-modality then sum. Simple, often works.

Attention-based fusion. Concatenate all retrieved items, let a small attention network weight them. Needs training.

MoE fusion. Gating network routes to modality-specific experts. Different query types route differently — a visual question weights images higher.

Production default: score fusion with a slight bias toward the query's dominant modality. Upgrade to MoE if A/B shows clear wins on your domain.

### Generation grounding

The LLM should cite which retrieved item drove each claim. For multi-modal:

- Text source: standard citation `[1]`.
- Image source: `[img 3]` with a short caption.
- Audio: `[audio 2 at 0:34]`.

Train the generator with grounding-aware data: each claim in the training target is tagged with the source index. At inference, the model naturally emits citations.

### The 2025 surveys

Abootorabi et al. (arXiv:2502.08826, "Ask in Any Modality"): taxonomy for multimodal RAG. Covers retrieval, fusion, generation. Broadest coverage.

Mei et al. (arXiv:2504.08748, "A Survey of Multimodal RAG"): focuses on sub-task benchmarks and failure modes. Useful for evaluation design.

Zhao et al. (arXiv:2503.18016): vision-focused survey. Strong on ColPali-family work.

Reading all three gives you the state of the art as of spring 2025. Most of the sub-problems are still open.

### MuRAG — the foundational paper

MuRAG (Chen et al., 2022) was the first multimodal RAG. Retrieved image + text from a multimodal KB, generated answers. Showed feasibility before the VLM wave. Modern systems (REACT, VisRAG, M3DocRAG) build on it.

### A production trip-planner example

Query: "find me a quiet vegan brunch with natural light."

Pipeline:

1. Decompose query. "quiet" → audio/review keyword; "vegan brunch" → menu item; "natural light" → image feature.
2. Retrieve per modality:
   - Text retrieval on reviews: "vegan brunch, quiet ambiance."
   - Image retrieval on restaurant photos: "natural light, airy."
   - Audio retrieval on ambient-sound clips: "low decibel, no music."
3. Fuse scores. Each restaurant has a composite score.
4. Top-k restaurants → VLM generator with all evidence → answer with citations.

This is well beyond text-RAG. Each modality adds signal that text alone misses.

### Agentic multimodal RAG

Multi-hop: if the first retrieval does not return high-confidence answers, the LLM reformulates and retrieves again. Agentic RAG patterns from Phase 14 apply here. Examples:

- Retrieve initial top-10 → LLM asks "too noisy, filter for <40 dB" → re-retrieve.
- Retrieve images → LLM sees one has a menu → retrieve the menu text → answer.

Adds complexity but handles queries that single-shot retrieval cannot.

### Evaluation

Cross-modal evaluation is still immature. Common proxies:

- Recall@k per modality.
- Fused top-k accuracy.
- Human-judged end-to-end satisfaction.
- Task-specific (bookings completed, purchases made).

No standard benchmark spans all modalities. Most papers evaluate on domain-specific tasks.

## Use It

`code/main.py`:

- Three mock retrievers (text, image, audio) operating on a shared corpus of restaurants.
- Score fusion that combines modality scores with configurable weights.
- A generator stub that emits a final answer with citations.
- A simple agentic loop that reformulates the query if confidence is low.

## Ship It

This lesson produces `outputs/skill-multimodal-rag-designer.md`. Given a product spec with a multimodal query flow, designs retrievers, fusion, generator, and evaluation.

## Exercises

1. Propose a medical-triage multimodal RAG: query = photo of injury + text symptoms. What modalities retrieve from what KB?

2. Score fusion is a simple weighted sum. What failure mode does it have that MoE fusion avoids?

3. Read Abootorabi et al.'s taxonomy (Section 3). What are the three canonical sub-problems and how do they map to your chosen product?

4. Design an eval spec for a trip-planner multimodal RAG. What metrics cover image recall, audio recall, and composite correctness?

5. Agentic multi-hop RAG has a latency tax per round-trip. At what query difficulty does the accuracy gain justify the latency?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Query one modality, retrieve another" | Text query retrieves images; image query retrieves text; requires a shared space or translator |
| Score fusion | "Combine scores" | Weighted sum of per-modality retrieval scores; simplest fusion |
| MoE fusion | "Modality-routed experts" | Gating network picks which modality's scores to trust per query |
| Grounded generation | "Cite your sources" | Each claim in the answer tagged with the source index |
| MuRAG | "First multimodal RAG" | 2022 paper that established the multimodal RAG pattern |
| Agentic multi-hop | "Reformulate and retry" | LLM re-queries retrievers when first-pass confidence is low |

## Further Reading

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/24-multimodal-rag-cross-modal)

---

## Part 5 (ch246): Multimodal Agents and Computer-Use (Capstone)

> The 2026 frontier product is a multimodal agent that reads screenshots, clicks buttons, navigates web UIs, fills forms, and completes workflows end-to-end. SeeClick and CogAgent (2024) proved the GUI-grounding primitive. Ferret-UI added mobile. ChartAgent introduced visual tool-use for charts. VisualWebArena and AgentVista (2026) are the benchmarks the frontier chases — and even Gemini 3 Pro and Claude Opus 4.7 score ~30% on AgentVista's hard tasks. This capstone pulls together every thread of Phase 12: perception (high-res VLM), reasoning (LLM with tool use), grounding (coordinate output), long-horizon memory, and evaluation.

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## Learning Objectives

- Design a multimodal agent loop: perceive → reason → act → observe → repeat.
- Build a GUI grounding output schema (click coordinates, type text, scroll, drag) the VLM can emit as JSON.
- Compare screenshot-only agents vs accessibility-tree agents vs hybrid agents.
- Set up a multimodal agent benchmark evaluation on a small VisualWebArena slice.

## The Problem

A booking-site workflow: "find me a flight to Tokyo for April 15, aisle seat under $800, book it."

A multimodal agent needs to:

1. Take a screenshot of the browser.
2. Parse the screenshot + URL + goal into a plan.
3. Emit a structured action: click (at x,y), type "Tokyo" (at element E), scroll down, select (radio button).
4. Apply the action to the browser.
5. Observe the new state (next screenshot).
6. Repeat until the task is done.

Each step is a multimodal VLM call. The VLM output must be parseable JSON. Errors compound across steps, so recovery matters.

## The Concept

### GUI grounding — the primitive

GUI grounding is: given a screenshot and a natural language instruction, output the (x, y) coordinate to click (or other action).

SeeClick (arXiv:2401.10935) was the first open result at scale: fine-tune a VLM on synthetic + real GUI data, output coordinates as plain text tokens. Works.

CogAgent (arXiv:2312.08914) added 1120×1120 high-resolution encoding for dense UIs. Score: ~84% on web navigation.

Ferret-UI (arXiv:2404.05719) focuses on mobile UIs, integrates with iOS accessibility data.

Output format is usually JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

The `element_desc` helps recovery: if coordinates drift between screenshots, the semantic hint lets the system re-ground.

### Action schemas

A typical action schema has 6-10 action types:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

The agent emits one action per step. The browser wrapper executes and returns the new state.

### Screenshot-only vs accessibility-tree

Two input modes:

- Screenshot-only: full image, no structural info. Most general; works on any app.
- Accessibility tree: structured DOM / iOS accessibility info. Much more reliable for grounding; works where the tree is available.
- Hybrid: both, with the tree as a reliable grounder for atomic actions and the screenshot for semantic context.

Production agents use hybrid when possible. Browser automation (Selenium + accessibility) always has the tree; desktop apps sometimes do.

### Long-horizon memory

A 20-step workflow generates 20 screenshots. The VLM's context fills up fast. Three compression strategies:

- Summary-chain: after every 5 steps, summarize what has happened, drop old screenshots.
- Skip-frame: keep the first, last, and every 3rd screenshot.
- Tool-recorded log: execute actions, keep a text log of what was done; don't re-look at old screenshots.

Claude's computer-use API uses the log pattern. Simpler, more reliable.

### Visual tool use

ChartAgent (arXiv:2510.04514) introduces visual tool use for chart understanding: crop, zoom, OCR, call external detection. The agent can output "crop to region (100, 200, 300, 400) then call OCR" as a tool call. The tool returns text; the VLM continues reasoning.

This pattern generalizes: set-of-mark prompting, region annotation, and external detection tools all fit the same "output a tool call, receive a structured response" schema.

### The 2026 benchmarks

- ScreenSpot-Pro. GUI grounding on ~1k web screenshots. Open SOTA Qwen2.5-VL-72B ~85%. Frontier ~90%.
- VisualWebArena. End-to-end web tasks (shop, forum, classifieds). Open SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). The hardest 2026 benchmark. Realistic workflows across 12 domains. Frontier models score 27-40%; open models 10-20%.
- WebArena / WebShop. Older benchmarks; saturated by frontier.

### Why it's still hard

Agent performance bottlenecks:

1. Visual grounding at fine scale. "Click the small X" fails often at mobile resolution.
2. Long-horizon planning. After 10 actions, the agent drifts from the goal.
3. Error recovery. When a click fails (wrong button), detecting + recovering is rarely trained data.
4. Cross-page context. Jumping between tabs or long forms loses state.

Research directions: memory architectures, explicit replanning, multimodal verification (screenshot match for action success).

### The capstone build-it

The capstone task: build a computer-use agent that:

1. Reads the HTML + screenshot of a booking-site mock page.
2. Plans a multi-step sequence: search → select → fill form → submit.
3. Emits JSON actions matching the action schema.
4. Evaluates on a fixed 10-task slice.

The lesson provides scaffold code that is easy to extend into a real browser.

## Use It

`code/main.py` is the capstone scaffold:

- Action schema JSON definition (10 actions).
- Mock browser state as dict.
- Agent loop skeleton: receive state, emit action, apply, loop.
- 10-task mini-benchmark (synthetic pages) to measure end-to-end success rate.
- Error-recovery hook for when an action fails.

## Ship It

This lesson produces `outputs/skill-multimodal-agent-designer.md`. Given a computer-use product (domain, action set, evaluation target), designs the full agent loop, memory strategy, grounding mode, and expected benchmark score.

## Exercises

1. Extend the action schema with a `screenshot_region` tool (crop + zoom). What tasks benefit?

2. Read AgentVista (arXiv:2602.23166). Describe the hardest task category and why frontier models still fail.

3. Long-horizon memory compression: design a summary-chain with ≤4 screenshots kept live, any number logged.

4. Build an error-recovery hook: on action failure (button not found), what does the agent do next?

5. Compare screenshot-only Claude 4.7 to hybrid screenshot + accessibility-tree Qwen2.5-VL on 10 web tasks. Which wins on which tasks?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | Model outputs (x,y) for the target of an instruction on a screenshot |
| Action schema | "Tool definitions" | JSON description of valid actions (click, type, scroll, drag) |
| Accessibility tree | "Structured DOM" | Machine-readable UI hierarchy from browser/iOS APIs |
| Hybrid agent | "Screenshot + tree" | Uses both image and structured info; more reliable than either alone |
| Visual tool use | "Zoom/crop/detect" | Agent calls external vision tools (OCR, detection) mid-plan |
| Summary-chain | "Memory compression" | Periodic text summaries replace long screenshot history |
| VisualWebArena | "E2E web bench" | 2024 benchmark for end-to-end web tasks |
| AgentVista | "2026 hard bench" | 12-domain realistic workflows; even Gemini 3 Pro scores ~30% |

## Further Reading

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/25-multimodal-agents-computer-use)

---

## Part 6 (ch420): Multimodal Document QA (Vision-First PDF, Tables, Charts)

> The 2026 document-QA frontier moved away from OCR-then-text and toward vision-first late interaction. ColPali, ColQwen2.5, and ColQwen3-omni treat each PDF page as an image, embed it with multi-vector late interaction, and let the query attend to patches directly. On financial 10-Ks, scientific papers, and handwritten notes this pattern beats OCR-first by a large margin. Build the pipeline end to end on 10k pages and publish the side-by-side against OCR-then-text.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Time:** 30 hours

## Problem

Enterprises sit on PDFs that OCR pipelines mangle: scanned 10-Ks with rotated tables, scientific papers dense with equations, charts that only make sense as images, handwritten annotations. Treating these as text-first means losing half the signal. The 2026 answer is late-interaction multi-vector retrieval on raw page images. ColPali (Illuin Tech) introduced it; ColQwen2.5-v0.2 and ColQwen3-omni pushed accuracy. On ViDoRe v3, vision-first retrieval scores above OCR-then-text by meaningful margins — and the gap widens on charts, tables, and handwriting.

The trade-off is storage and latency. A ColQwen embedding is ~2048 patch vectors per page, not a single 1024-dim vector. Raw storage balloons. DocPruner (2026) brings 50% pruning without measurable accuracy loss. You will index 10k pages, measure ViDoRe v3 nDCG@5, serve answers under 2s, and compare directly against an OCR-then-text baseline.

## Concept

Late interaction means every query token scores against every patch token, and the maximum score per query token is summed. You get fine-grained matching without needing a single pooled vector. A multi-vector index (Vespa, Qdrant multi-vector, or AstraDB) stores the per-patch embeddings and runs MaxSim at retrieval time.

The answerer is a vision-language model that takes the query plus the top-k retrieved pages as images and writes an answer with evidence regions (bounding boxes or page references). Qwen3-VL-30B, Gemini 2.5 Pro, and InternVL3 are the 2026 frontier choices. For equations and scientific notation, an OCR fallback (Nougat, dots.ocr) is spliced in as an optional text channel.

Evaluation is a two-dimensional matrix. One axis: content type (plain text paragraphs, dense tables, bar/line charts, handwritten notes, equations). Other axis: retrieval approach (vision-first late interaction vs OCR-then-text vs hybrid). Each cell gets nDCG@5 and answer accuracy. The report is the deliverable.

## Architecture

```mermaid
graph TD
    pdf[PDFs] --> render[page renderer (PyMuPDF, 180 DPI)]
    render --> embed[ColQwen2.5-v0.2 embed]
    embed --> prune[DocPruner 50% compression]
    prune --> index[multi-vector index (Vespa / Qdrant)]
    query[query] --> index
    index --> retrieve[retrieve top-k pages (MaxSim)]
    retrieve --> vlm[VLM answerer: Qwen3-VL-30B / Gemini 2.5 Pro]
    ocr[OCR fallback] --> vlm
    vlm --> answer[answer with cited page numbers + evidence regions]
    answer --> ui[Streamlit / Next.js viewer]
```

## Stack

- Page rendering: PyMuPDF (fitz) at 180 DPI, portrait-normalized
- Late-interaction model: ColQwen2.5-v0.2 or ColQwen3-omni (vidore team on Hugging Face)
- Index: Vespa with multi-vector field, or Qdrant multi-vector, or AstraDB with MaxSim
- Pruning: DocPruner 2026 policy (keep high-variance patches, 50% compression at < 0.5% accuracy loss)
- OCR fallback (equations / dense tables): dots.ocr or Nougat
- VLM answerer: Qwen3-VL-30B self-hosted or Gemini 2.5 Pro hosted; InternVL3 as fallback
- Evaluation: ViDoRe v3 benchmark, M3DocVQA for multi-page reasoning
- Viewer UI: Next.js 15 with canvas overlay for evidence regions

## Build It

1. **Ingest.** Walk a corpus of 10k PDF pages across 10-Ks, scientific papers, and scanned documents. Render each page to a 1536x2048 PNG. Persist `{doc_id, page_num, image_path}`.

2. **Embed.** Run ColQwen2.5-v0.2 on each page image. Output shape ~2048 patch embeddings of dim 128. Apply DocPruner to keep the highest-signal half. Write to Vespa multi-vector field or Qdrant multi-vector.

3. **Query.** For each incoming query, embed with the query tower (token-level embeddings). Run MaxSim against the index: for every query token, take the max dot-product over page patch embeddings, sum. Return top-k pages.

4. **Synthesize.** Call Qwen3-VL-30B with the query and the top-5 page images. Prompt: "Answer using only the supplied pages. Cite each claim by (doc_id, page) and name the region (figure, table, paragraph)."

5. **Evidence regions.** Post-process the answer to extract cited regions. If the VLM emits bounding boxes (Qwen3-VL does), render them as overlays in the viewer.

6. **OCR fallback.** For pages identified as equation-dense (heuristic on image variance), run Nougat or dots.ocr and pass the OCR text as an extra channel alongside the image.

7. **Eval.** Run ViDoRe v3 (retrieval nDCG@5) and M3DocVQA (multi-page QA accuracy). Also run OCR-then-text pipeline on the same corpus with the same synthesizer. Produce a content-type × approach matrix.

8. **UI.** Streamlit prototype first; Next.js 15 production viewer with page-by-page evidence-region overlay.

## Use It

```console
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Ship It

`outputs/skill-doc-qa.md` describes the deliverable: a vision-first multimodal document QA system tuned to a specific corpus and evaluated against an OCR-then-text baseline on ViDoRe v3.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline and published leaderboard |
| 20 | Evidence-region grounding | Fraction of cited regions that actually contain the answer span |
| 20 | Storage and latency engineering | DocPruner compression ratio, index p95, answer p95 |
| 20 | Multi-page reasoning | Accuracy on a hand-labeled 100-question multi-page set |
| 15 | Source-inspection UX | Viewer clarity, overlay fidelity, side-by-side comparison tools |
| **100** | | |

## Exercises

1. Measure ColQwen2.5-v0.2 vs ColQwen3-omni on the same corpus. Which pages does one get right and the other miss? Add a "content class" tag to the index to route by type.

2. Prune embeddings aggressively (75%, 90%). Find the compression cliff: the point where ViDoRe nDCG@5 drops below the OCR baseline.

3. Build a hybrid: run OCR-then-text and ColQwen in parallel, fuse with RRF, rerank with a cross-encoder. Does the hybrid beat either alone? Where does it help most?

4. Swap Qwen3-VL-30B for a smaller VLM (Qwen2.5-VL-7B). Measure the accuracy-per-dollar curve.

5. Add handwritten-note support. Render the handwriting corpus, embed with ColQwen, measure retrieval. Compare against a handwriting OCR pipeline.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColPali-style retrieval" | Query tokens score against page patches independently; MaxSim aggregates |
| Multi-vector | "Per-patch embedding" | Each document has many vectors, not one pooled vector |
| MaxSim | "Late-interaction scoring" | For every query token, take max similarity over document vectors; sum |
| DocPruner | "Patch compression" | 2026 pruning that keeps 50% of patches with negligible accuracy loss |
| ViDoRe v3 | "Document-retrieval benchmark" | The 2026 standard for measuring visual-document retrieval |
| Evidence region | "Cited bounding box" | A bbox on the source page that localizes the answer span |
| OCR fallback | "Equation channel" | Text pipeline used alongside vision for equation- or table-heavy pages |

## Further Reading

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali)
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [ColQwen family on Hugging Face](https://huggingface.co/vidore)
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952)
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html)
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors)
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html)
- [Nougat OCR](https://github.com/facebookresearch/nougat)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/04-multimodal-document-qa)
