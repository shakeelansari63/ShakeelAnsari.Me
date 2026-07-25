# Long-Context Evaluation — NIAH, RULER, LongBench, MRCR

Gemini 3 Pro advertises 10M tokens of context. At 1M tokens, 8-needle MRCR drops to 26.3%. Advertised ≠ usable. Long-context evaluation tells you the actual capacity of the model you are shipping on.

## The Problem

You have a 200-page contract. The model claims a 1M-token context. You ask: "What is the termination clause?" The model answers from the cover page because the termination clause sits at 120k tokens deep, past where the model actually attends.

The 2026 context-capacity gap: Spec sheets say 1M or 10M. Reality says 60-70% of that is usable.

- **Retrieval (single needle):** near-perfect up to advertised max on frontier models.
- **Multi-hop / aggregation:** degrades sharply past ~128k on most models.
- **Reasoning over dispersed facts:** the first task to fail.

## The Concept

**Needle-in-a-Haystack (NIAH, 2023).** Place a fact at a controlled depth. Ask the model to retrieve it. Sweep depth × length. Frontier models saturate this now.

**RULER (Nvidia, 2024).** 13 task types across retrieval, multi-hop tracing, aggregation, QA. Reveals models that pass NIAH but fail multi-hop.

**LongBench v2 (2024).** 503 multiple-choice questions, 8k-2M word contexts. The production benchmark.

**MRCR (Multi-Round Coreference Resolution).** Multi-turn coreference at scale. 8, 24, 100-needle variants.

**NoLiMa.** Non-lexical needle. Needle and query share no literal overlap.

**What to actually report:**
- Advertised context window.
- Effective retrieval length (NIAH pass at 90%).
- Effective reasoning length (multi-hop pass at threshold).
- Degradation curve (accuracy vs context length per task).

Usually the reasoning-effective length is 25-50% of the advertised window.

## Build It

### Step 1: A Custom NIAH for Your Domain

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)

def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Sweep `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Plot the heatmap.

### Step 2: A Multi-Needle Variant

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Single-needle success does not predict multi-needle success.

### Step 3: Multi-Hop Variable Tracing (RULER-style)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Requires chaining three assignments. Frontier models at 128k often drop to 50-70% accuracy.

### Step 4: LongBench v2 on Your Stack

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

## Pitfalls

- NIAH-only evaluation says nothing about multi-hop.
- Uniform depth sampling misses "lost in the middle."
- Lexical overlap with filler makes retrieval trivial.
- Ignoring latency: 1M-token prompts take 30-120 seconds to prefill.
- Vendor self-reported numbers always need independent verification.

## Use It

| Situation | Benchmark |
|-----------|-----------|
| Quick sanity check | Custom NIAH at 3 depths × 3 lengths |
| Model selection for production | RULER at your target length |
| Real-world QA quality | LongBench v2 single-doc-QA subset |
| Model upgrade regression | Fixed in-house NIAH + RULER harness |

## Exercises

1. **Easy.** Build NIAH with 3 depths × 3 lengths. Run on any model. Plot 3×3 heatmap.
2. **Medium.** Add a 3-needle variant. Compare against single-needle pass rate.
3. **Hard.** Construct a variable-tracing task in 64k filler. Measure accuracy across 3 frontier models.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| NIAH | Plant a fact in filler, ask the model to retrieve it. |
| RULER | 13 task types across retrieval / multi-hop / aggregation / QA. |
| Effective context | Length where accuracy still holds above threshold. |
| Lost in the middle | Models under-attend to content in the middle of inputs. |
| Multi-needle | Multiple plants; tests attention juggling. |
| NoLiMa | Needle and query share no literal tokens. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/28-long-context-evaluation)
