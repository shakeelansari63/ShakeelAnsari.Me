# Chunking Strategies, Compared

> Chunking decides what your retriever can ever surface. Get the boundaries wrong and no model can repair the damage downstream.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04, 06, 07; Phase 19 Track B foundations
**Time:** ~90 minutes

## Learning Objectives

- Implement five chunking strategies: fixed-window, sentence, recursive-split, semantic clustering, structural markdown.
- Measure recall@k on a fixture corpus with gold-labeled answer spans.
- Read a chunk-length distribution and recognize failure modes.
- Pick a default for a new corpus by inspecting three properties.

## The Concept

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

## Failure modes

| Failure | Description |
|---------|-------------|
| Orphan sentences | Sentence packing misses the topic sentence |
| Mid-symbol cuts | Fixed-window splits identifiers in half |
| Header-only chunks | Structural splitter emits empty headings |
| Semantic drift | Uniform-topic corpus under-cuts with semantic clustering |

## Build It

`code/main.py` implements five chunkers, `mock_embed`, `DenseIndex`, `eval_recall`.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Recall@k | Fraction of queries where any top-k chunk overlaps gold answer span |
| Chunk overlap | Re-include last N chars of previous chunk |
| Structural splitter | Cut at H1/H2/H3 boundaries |
| Semantic chunker | Embed sentences, cluster by centroid similarity |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/64-chunking-strategies-advanced)
