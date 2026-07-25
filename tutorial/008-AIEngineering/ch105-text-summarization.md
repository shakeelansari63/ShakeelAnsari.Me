# Text Summarization

Extractive systems tell you what the document said. Abstractive systems tell you what the author meant. Different tasks, different pitfalls.

## The Concept

**Extractive summarization** is a ranking problem. Score every sentence, return the top-k. The output is always grammatical. The risk is missing content distributed across the article.

**Abstractive summarization** is a generation problem. A transformer produces new text conditioned on the input. Fluent and compressive but may hallucinate.

**Evaluation with ROUGE.** ROUGE-1 and ROUGE-2 score unigram and bigram overlap. ROUGE-L scores longest common subsequence. 40 ROUGE-L is "good", 50 is "exceptional."

## Build It

### Step 1: TextRank (Extractive)

```python
import math
import re
from collections import Counter

def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())

def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom

def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

### Step 2: Abstractive with BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""
summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

### Step 3: ROUGE Evaluation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Always use stemming. Without it, "running" and "run" count as different words.

### Beyond ROUGE (2026)

Production recommendation: report ROUGE-L for legacy comparison, BERTScore for semantic overlap, G-Eval for coherence and factuality. Calibrate against 50-100 human-labeled summaries.

### The Factuality Problem

Abstractive summaries hallucinate. Types: entity swap, number drift, polarity flip, fact invention.

For anything user-facing where factuality matters (news, medical, legal, financial), extractive is the safer default. Abstractive needs a factuality check in the loop.

## Use It

| Use case | Recommended |
|---------|-------------|
| News, 3-5 sentence summary | `facebook/bart-large-cnn` |
| Scientific papers | `google/pegasus-pubmed` |
| Multi-document, long-form | Any LLM with 32k+ context |
| Dialog summarization | `philschmid/bart-large-cnn-samsum` |
| Extractive, low hallucination risk | TextRank or `sumy`'s LexRank |

## Exercises

1. **Easy.** Run TextRank on 5 news articles. Measure ROUGE-L.
2. **Medium.** Implement entity-level factuality: extract NEs from source and summary, compute recall/precision.
3. **Hard.** Compare BART-large-CNN against an LLM on 50 articles. Report ROUGE-L, factuality, and cost.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Extractive | Return sentences verbatim from the source. Never hallucinates. |
| Abstractive | Generate new text conditioned on source. Can hallucinate. |
| ROUGE | N-gram/LCS overlap between system output and reference. |
| TextRank | PageRank over sentence similarity graph. |
| Factuality | Whether summary claims are supported by the source. |
| Hallucination | Content in summary the source does not support. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/12-text-summarization)
