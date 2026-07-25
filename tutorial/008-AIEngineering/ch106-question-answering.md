# Question Answering Systems

Three systems shaped modern QA. Extractive found spans. Retrieval-augmented grounded them in documents. Generative produced answers. Every modern AI assistant is a mix of the three.

## The Concept

**Extractive QA.** Given a question and a passage containing the answer, find the start and end indices of the answer span. Never hallucinates, never handles questions the passage cannot answer.

**Retrieval-augmented (RAG).** Two stages. A retriever finds top-k passages. A reader (extractive or generative) produces the answer. This is the bedrock of every RAG pipeline.

**Generative / Closed-book QA.** A large language model answers from its parametric memory. Fastest at inference, least reliable on facts.

## Build It

### Step 1: Extractive QA with a Pretrained Model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

### Step 2: A Retrieval-Augmented Pipeline

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)

def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]

def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)

print(answer("When was the first iPhone released?"))
```

### Step 3: Generative with RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

The prompt pattern matters. Explicit grounding cuts hallucination rates by 40-60%.

### Step 4: Evaluation

SQuAD uses **Exact Match (EM)** and **token-level F1**. EM is strict. F1 gives partial credit.

For production QA: answer accuracy (LLM-judged), citation accuracy, refusal calibration, retrieval recall.

### RAGAS: The 2026 Production Eval Framework

`RAGAS` scores four dimensions without gold references:

- **Faithfulness.** Does each claim come from retrieved context? (NLI-based)
- **Answer relevance.** Does the answer address the question?
- **Context precision.** Of retrieved chunks, what fraction were relevant?
- **Context recall.** Did retrieval contain all needed information?

## Use It

| Use case | Recommended |
|---------|-------------|
| Given passage, find answer span | `deepset/roberta-base-squad2` |
| Over a fixed corpus, closed-book not acceptable | RAG: dense retriever + LLM reader |
| Highly factual, regulated domains | Extractive over an authoritative corpus |

## Exercises

1. **Easy.** Set up SQuAD extractive pipeline on 10 Wikipedia passages with 10 questions.
2. **Medium.** Add a refusal classifier. Return "I don't know" when top retrieval score is below a threshold.
3. **Hard.** Build RAG over a 10,000-document corpus. Implement hybrid retrieval (BM25 + dense) with RRF fusion.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Extractive QA | Predict start and end indices of the answer within a given passage. |
| Open-domain QA | No given passage; must retrieve then answer. |
| RAG | Retriever + reader pipeline. |
| SQuAD | Stanford Question Answering Dataset. EM + F1 metrics. |
| Hallucination | Reader output not supported by retrieved context. |
| Refusal calibration | System correctly says "I don't know" when unable to answer. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/13-question-answering)
