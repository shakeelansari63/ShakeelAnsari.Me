# Coreference Resolution

"She called him. He did not answer. The doctor was at lunch." Three references to two people and nobody is named. Coreference resolution figures out who is who.

## The Concept

Coreference resolution links every expression that refers to the same real-world entity into one cluster. It is the glue between surface-level NLP (NER, parsing) and downstream semantics (IE, QA, summarization, KG).

**Mention types:**
- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Appositive.** "Tim Cook, Apple's CEO,"

**Architectures (in order of sophistication):**

1. **Rule-based (Hobbs, 1978).** Syntactic-tree-based pronoun resolution using grammar rules.
2. **Mention-pair classifier.** Predict whether each pair of mentions corefer.
3. **Mention-ranking.** For each mention, rank candidate antecedents.
4. **Span-based end-to-end (Lee et al., 2017).** Enumerate all candidate spans, predict mention scores and antecedent probabilities. The modern default.
5. **Generative (2024+).** Prompt an LLM to list pronouns and antecedents.

**Evaluation.** Five standard metrics (MUC, B³, CEAF, BLANC, LEA). Report the average of the first three as CoNLL F1. State-of-the-art on CoNLL-2012: ~83 F1.

## Build It

### Step 1: Pretrained Neural Coreference

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Cluster 1: [Apple, The company, they], Cluster 2: [new products]

### Step 2: Using LLMs for Coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Two failure modes: LLMs over-merge ("him" and "her" referring to distinct people), and silently drop mentions in long documents.

## Pitfalls

- **Singleton explosion.** Some systems report every mention as its own cluster.
- **Pronouns in long context.** Performance drops ~15 F1 on documents over 2000 tokens.
- **Gender assumptions.** Hard-coded gender rules break on non-binary referents.
- **LLM drift on long docs.** Use sliding-window + merge.

## Use It

| Situation | Pick |
|-----------|------|
| English, single document | `en_coreference_web_trf` or AllenNLP neural coref |
| Multilingual | SpanBERT / XLM-R on OntoNotes or Multilingual CoNLL |
| Quick LLM baseline | GPT-4o / Claude with structured-output coref prompt |

Integration pattern: run NER first, run coref, merge coref clusters into NER entities. Downstream tasks see one entity per cluster.

## Exercises

1. **Easy.** Run the rule-based resolver on 5 hand-crafted paragraphs. Measure mention-link accuracy.
2. **Medium.** Use a pretrained neural coref model on a news article. Compare clusters against your manual annotation.
3. **Hard.** Build a coref-enhanced NER pipeline. Measure entity-coverage improvement vs NER-only on 100 articles.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Mention | A span of text that refers to an entity. |
| Antecedent | The earlier mention a later one corefers with. |
| Cluster | Set of mentions that all refer to the same real-world entity. |
| Anaphora | Later mention refers to earlier ("he" → "John"). |
| Cataphora | Earlier mention refers to later ("When he arrived, John..."). |
| Bridging | Implicit reference ("The wheels" → a previously mentioned car). |
| CoNLL F1 | Average of MUC, B³, CEAF-φ4 F1 scores. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/24-coreference-resolution)
