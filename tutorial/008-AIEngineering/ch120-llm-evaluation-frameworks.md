# LLM Evaluation — RAGAS, DeepEval, G-Eval

Exact-match and F1 miss semantic equivalence. Human review does not scale. LLM-as-judge is the production answer — with enough calibration to trust the number.

## The Problem

Your RAG system answers "June 29th, 2007." The gold reference is "June 29, 2007." Exact Match scores 0. F1 scores ~75%. A human would score 100%. You need an evaluator that understands meaning, runs cheaply at scale, and surfaces the right failure modes.

## The Concept

**LLM-as-judge.** Replace a static metric with an LLM that scores outputs given a rubric. GPT-4o-mini at ~$0.003 per scored case enables 1000-sample regression evals for under $5.

**The RAG four (RAGAS):**

| Metric | Question | Backend |
|--------|----------|---------|
| Faithfulness | Do claims come from context? | NLI-based entailment |
| Answer relevance | Does answer address question? | Generate hypothetical questions |
| Context precision | What fraction of chunks were relevant? | LLM-judge |
| Context recall | Did retrieval return everything? | LLM-judge against gold answer |

**G-Eval.** Define a custom criterion. The framework auto-expands into chain-of-thought evaluation steps, then scores 0-1.

**Calibration.** Never trust raw judge score until correlated against human labels. Compute Spearman rho. If < 0.7, your rubric needs work.

## Build It

### Step 1: Faithfulness with NLI (RAGAS-style)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

LLM = Callable[[str], str]

def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()

def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

### Step 2: Answer Relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

### Step 3: G-Eval Custom Metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

### Step 4: CI Gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric

def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

## Pitfalls

- No calibration. Judge with 0.3 correlation to human labels is noise.
- Self-evaluation. Using the same LLM to generate and judge inflates scores 10-20%.
- Positional bias in pairwise judging. Judges prefer the first option.
- Raw aggregate hides failures. Always inspect the bottom quantile.
- Golden dataset rot. Version your eval sets.

## Use It

| Use case | Framework |
|---------|-----------|
| RAG quality monitoring | RAGAS (4 metrics) |
| CI/CD regression gates | DeepEval + pytest |
| Custom domain criteria | G-Eval within DeepEval |
| Online live-traffic monitoring | RAGAS reference-free mode |

## Exercises

1. **Easy.** Use RAGAS on 10 RAG examples with known hallucinations. Verify faithfulness catches each.
2. **Medium.** Hand-label 50 QA answers 0-1. Score with G-Eval. Measure Spearman rho.
3. **Hard.** Build a pytest CI gate with DeepEval. Intentionally regress the retriever and verify the gate fails.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| LLM-as-judge | Prompt a judge model to score outputs 0-1 given a rubric. |
| RAGAS | Open-source eval framework with 4 reference-free RAG metrics. |
| Faithfulness | Fraction of answer claims entailed by retrieved context. |
| Context precision | Fraction of top-K chunks that actually mattered. |
| G-Eval | Rubric + chain-of-thought eval steps + 0-1 score. |
| Calibration | Spearman correlation between judge and human score. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks)
