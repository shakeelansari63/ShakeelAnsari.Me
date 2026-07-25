# Entity Linking & Disambiguation

NER found "Paris." Entity linking decides: Paris, France? Paris Hilton? Paris, Texas? Without linking, your knowledge graph stays ambiguous.

## The Concept

Entity linking (EL) resolves each mention to a unique entry in a knowledge base. Two subtasks:

1. **Candidate generation.** Given "Jordan," which KB entries are plausible?
2. **Disambiguation.** Given the context, which candidate is the right one?

**Disambiguation approaches:**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Fast, no training.
2. **Embedding-based (BLINK).** Encode mention + context and each candidate's description. Pick max cosine. The 2020-2024 default.
3. **Generative (GENRE, LLM-based).** Decode the entity's canonical name token-by-token with constrained decoding.

## Build It

### Step 1: Build an Alias Index

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data: ~18M (alias, entity) pairs.

### Step 2: Embedding-Based (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

At index time, embed every KB entity once. At query time, embed mention + context, dot-product against candidates, pick max.

### Step 3: Generative Entity Linking

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Combined with a whitelist (Outlines `choice`), this is the simplest EL pipeline to ship in 2026.

## Pitfalls

- **NIL handling.** Some mentions are not in the KB. Must predict NIL.
- **Mention boundary errors.** Upstream NER misses partial spans.
- **Popularity bias.** Trained systems over-predict frequent entities.
- **KB staleness.** New companies/events are not in last year's Wikipedia dump.

## Use It

| Situation | Pick |
|-----------|------|
| General-purpose English + Wikipedia | BLINK or REL |
| LLM-friendly, few mentions/day | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| Domain-specific KB | Custom BERT with KB-aware retrieval + fine-tune |

Production pattern: NER → coref → EL on each mention → collapse clusters to one canonical entity per cluster.

## Exercises

1. **Easy.** Implement the prior+context disambiguator on 10 ambiguous mentions.
2. **Medium.** Encode 50 ambiguous mentions with a sentence transformer. Compare embedding-based to Jaccard overlap.
3. **Hard.** Build a 1k-entity domain KB. Implement NER + EL end-to-end.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Entity linking | Map a mention to a unique KB entry. |
| Candidate generation | Return a shortlist of plausible KB entries. |
| Disambiguation | Score candidates using context, pick the winner. |
| Alias index | Map from surface form → candidate entities. |
| NIL | Explicit prediction that no KB entry matches. |
| AIDA-CoNLL | 1,393 Reuters articles with gold entity links. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/25-entity-linking)
