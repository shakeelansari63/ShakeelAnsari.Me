# Relation Extraction & Knowledge Graph Construction

NER found the entities. Entity linking anchored them. Relation extraction finds the edges between them. A knowledge graph is the sum of nodes, edges, and their provenance.

## The Concept

**Triple form.** `(subject_entity, relation_type, object_entity)`. Relations come from a closed ontology or an open set.

**Three extraction approaches:**

1. **Rule/pattern-based.** Hearst patterns: "X such as Y" → `(Y, isA, X)`. Precise, brittle.
2. **Supervised classifier.** Given two entity mentions, predict the relation. Trained on TACRED, ACE.
3. **Generative LLM.** Prompt the model to emit triples. Works out of the box. Needs provenance.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Anchor every entity span with exact positions. Extract triples linked to anchors. Verify each triple element against source text. Supplement with a coverage pass.

## Build It

### Step 1: Pattern-Based Extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

### Step 2: Supervised Relation Classification

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL is a seq2seq relation extractor: text in, triples out, already in Wikidata property ids.

### Step 3: LLM-Prompted Extraction with Anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text.
"""
```

Verify every returned span against the source. Reject anything where `text[start:end] != triple_entity`.

### Step 4: Canonicalize onto a Closed Ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",
    "was born in":   "P19",
    "founded":        "P112",
    "works at":       "P108",
}

def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None
```

Canonicalization is often 60-80% of the engineering work.

## Pitfalls

- Coreference before RE. "He founded Apple" needs coreference resolution first.
- Entity canonicalization. "Apple Inc" and "Apple" must resolve to the same node.
- Hallucinated triples. Enforce span verification.
- Relation canonicalization drift. Collapse to canonical ids.
- Temporal errors. Many relations are time-bounded. Use qualifiers.

## Use It

| Situation | Pick |
|-----------|------|
| Fast production, general domain | REBEL with Wikidata canonicalization |
| Domain-specific | SciREX-style domain fine-tune |
| LLM-prompted, audited output | AEVS pipeline |

Integration pattern: NER → coref → entity linking → RE → ontology mapping → graph load.

## Exercises

1. **Easy.** Run the pattern extractor on 5 news-article sentences. Hand-check precision.
2. **Medium.** Use REBEL on the same sentences. Compare triples precision/recall.
3. **Hard.** Build the AEVS pipeline. Measure hallucination rate before vs after verify step.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Triple | `(s, r, o)` tuple. Atomic unit of a KG. |
| Open IE | Open-vocabulary relation phrases; high recall, low precision. |
| Closed ontology | Bounded set of relation types (Wikidata, UMLS). |
| Canonicalization | Map surface names/relations to canonical ids. |
| AEVS | Anchor-Extraction-Verification-Supplement pipeline (2026). |
| Provenance | Every triple carries a doc id + char-span to its source. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg)
