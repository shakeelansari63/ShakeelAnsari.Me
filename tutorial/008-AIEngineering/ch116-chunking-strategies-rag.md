# Chunking Strategies for RAG

Chunking configuration influences retrieval quality as much as the choice of embedding model (Vectara NAACL 2025). Get chunking wrong and no amount of reranking saves you.

## The Concept

Six chunking strategies:

**Fixed chunking.** Split every N characters/tokens. Simplest baseline. Breaks mid-sentence.

**Recursive.** LangChain's `RecursiveCharacterTextSplitter`. Try `\n\n`, then `\n`, then `.`, then space. Falls back cleanly. The 2026 default.

**Semantic.** Embed each sentence. Compute cosine similarity between adjacent sentences. Split where similarity drops below a threshold.

**Sentence.** Split on sentence boundaries. One sentence per chunk or window of N sentences.

**Parent-document.** Store small child chunks for retrieval and larger parent chunk for context. Retrieve by child, return parent.

**Late chunking (2024).** Embed the whole document at the token level first, then pool token embeddings into chunk embeddings.

**Contextual retrieval (Anthropic, 2024).** Prepend each chunk with an LLM-generated summary of its position in the document. 35-50% improvement.

### Match Chunk Size to Query Type

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

## Build It

### Step 1: Fixed and Recursive Chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Step 2: Semantic Chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

### Step 3: Parent-Document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping
```

### Step 4: Contextual Retrieval

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

## Pitfalls

- Chunking evaluated only on factoid queries misses multi-hop differences.
- Semantic chunking without minimum size produces 40-token fragments.
- Overlap as cargo cult: 2026 studies find often zero benefit.
- No min/max enforcement breaks retrieval.
- Never let a chunk span two documents.

## Use It

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Heavy cross-reference | Late chunking or contextual retrieval |
| Short utterances | One document = one chunk |

Start with recursive 512. Measure recall@5 on a 50-query eval set. Tune from there.

## Exercises

1. **Easy.** Chunk one 20-page document with fixed(512,0), recursive(512,0), and recursive(512,100).
2. **Medium.** Build a 30-query eval set. Measure recall@5 for recursive, semantic, and parent-document.
3. **Hard.** Implement contextual retrieval. Measure MRR improvement over baseline recursive.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Chunk | Sub-document unit that gets embedded, indexed, retrieved. |
| Overlap | N tokens shared between adjacent chunks. |
| Semantic chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Retrieve small children, return larger parents. |
| Late chunking | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | LLM-generated summary prepended to each chunk. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag)
