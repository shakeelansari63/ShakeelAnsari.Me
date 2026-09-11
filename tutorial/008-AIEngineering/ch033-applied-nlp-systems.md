# Translation, Summarization, QA & Search

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch104): Machine Translation

Translation is the task that paid for NLP research for thirty years and keeps paying now.

A model reads a sentence in one language and produces a sentence in another. Length varies. Word order varies. "I miss you" in French is "tu me manques" — literally "you are lacking to me." No word-level alignment survives that.

## The Concept

Modern MT is a transformer encoder-decoder trained on parallel text. Three operational choices drive quality:

- **Tokenizer.** SentencePiece BPE trained on a mixed-language corpus. Shared vocabulary enables zero-shot pairs.
- **Model size.** NLLB-200 distilled 600M (laptop) to 3.3B (production default) to 54.5B (research ceiling).
- **Decoding.** Beam width 4-5. Length penalty. Constrained decoding for terminology consistency.

## Build It

### Step 1: A Pretrained MT Call

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```
Les chats courent.
```

Three things: `src_lang` tells the tokenizer which script/segmentation, `forced_bos_token_id` tells the decoder which language to generate, and both are NLLB-specific.

### Step 2: BLEU and chrF

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Always use `sacrebleu`. It normalizes tokenization so scores are comparable across papers.

### The Three-Tier Evaluation Hierarchy (2026)

- **Heuristic** (BLEU, chrF). Fast, reference-based, interpretable.
- **Learned** (COMET, BLEURT, BERTScore). Neural models trained on human judgment.
- **LLM-as-judge** (reference-free). GPT-4-as-judge matches human agreement ~80% with a good rubric.

### What Breaks in Production

- **Hallucination.** Model invents content not in the source.
- **Off-target generation.** Model translates into the wrong language.
- **Terminology drift.** Inconsistent translation of the same term across docs.
- **Formality mismatch.** "tu" vs "vous" — model picks whichever was more common in training.
- **Length explosion on short input.** Overlong translations from very short inputs.

### Fine-tuning for a Domain

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)

def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )

ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

A few thousand high-quality parallel examples beats hundreds of thousands of noisy web-scraped ones.

## Use It

| Use case | Recommended starting point |
|---------|---------------------------|
| Any-to-any, 200 languages | `facebook/nllb-200-distilled-600M` or `nllb-200-3.3B` |
| English-centric, 50 languages | `facebook/mbart-large-50-many-to-many-mmt` |
| Short runs, cheap inference | Helsinki-NLP / Marian models |
| Maximum quality | GPT-4 / Claude / Gemini with translation prompts |

## Exercises

1. **Easy.** Translate a 5-sentence paragraph to French and back. Measure round-trip preservation.
2. **Medium.** Implement a language-ID check on translation outputs.
3. **Hard.** Fine-tune NLLB-200 on a 5,000-pair domain corpus. Measure BLEU before and after.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| BLEU | N-gram precision with brevity penalty. [0, 100]. |
| chrF | Character-level F-score. Sensitive for morphologically rich languages. |
| NMT | Transformer encoder-decoder trained on parallel text. |
| NLLB | Meta's 200-language MT model family. |
| Constrained decoding | Force specific tokens to appear/not appear in output. |
| Hallucination | Model output not supported by the source. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/11-machine-translation)

---

## Part 2 (ch105): Text Summarization

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

---

## Part 3 (ch106): Question Answering Systems

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

---

## Part 4 (ch107): Information Retrieval and Search

BM25 is precise but brittle. Dense casts a wide net but misses keywords. Hybrid is the 2026 default. Everything else is tuning.

## The Concept

Four layers of retrieval:

1. **Sparse retrieval (BM25).** Fast, precise on exact matches, terrible on semantics. Runs over an inverted index in sub-10ms per query.
2. **Dense retrieval.** Encode query and documents into vectors. Nearest neighbor search. Captures paraphrases and semantic similarity.
3. **Fusion.** Merge ranked lists from sparse and dense. Reciprocal Rank Fusion (RRF) ignores raw scores and only uses rank positions.
4. **Cross-encoder rerank.** Take top-30 from fusion, run a cross-encoder, keep top-5. Slower per pair but far more accurate.

## Build It

### Step 1: BM25 from Scratch

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

`k1=1.5` controls term-frequency saturation. `b=0.75` controls length normalization.

### Step 2: Dense Retrieval with a Bi-Encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings

def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

### Step 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

### Step 4: Hybrid Search + Rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

### Step 5: Evaluation

| Metric | Meaning |
|--------|---------|
| Recall@k | How often is the correct doc in top-k? |
| MRR | Average of 1/rank of first relevant document. |
| nDCG@k | Accounts for relevance gradations. |

## Hard-won Lessons from 2026 Production RAG

- 80% of RAG failures trace to ingestion and chunking, not the model.
- Chunking strategy matters more than chunk size.
- Parent-doc pattern: retrieve small children, return larger parents.
- k_rerank=3 is usually optimal.
- HyDE / query expansion bridges the phrasing gap.
- Three-way retrieval (BM25 + dense + learned-sparse) outperforms two-way.

## Use It

| Scale | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + MiniLM embeddings + RRF |
| 100k-10M docs | FAISS or pgvector + Elasticsearch |
| 10M+ docs | Qdrant / Weaviate / Vespa with hybrid support |

## Exercises

1. **Easy.** Implement `hybrid_search` on a 500-document corpus. Compare recall@5 for BM25-only, dense-only, and hybrid.
2. **Medium.** Add MRR calculation. Report MRR for each retrieval method.
3. **Hard.** Fine-tune a dense encoder on your domain using MultipleNegativesRankingLoss.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| BM25 | Okapi BM25. Scores by term frequency, IDF, and length. |
| Dense retrieval | Encode query + doc into vectors, find nearest neighbors. |
| Bi-encoder | Encodes query and doc independently. Fast. |
| Cross-encoder | Encodes query + doc together. Slow but accurate. |
| RRF | Combine rankings by summing `1/(k + rank)`. |
| Recall@k | Fraction of queries where a relevant doc is in top-k. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/14-information-retrieval-search)
