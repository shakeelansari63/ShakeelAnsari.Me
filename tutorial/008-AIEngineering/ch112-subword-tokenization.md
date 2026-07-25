# Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece

Word tokenizers choke on unseen words. Character tokenizers blow up sequence length. Subword tokenizers split the difference. Every modern LLM ships on one.

## The Concept

**BPE (Byte-Pair Encoding).** Start with character-level vocabulary. Count every adjacent pair. Merge the most frequent pair into a new token. Repeat until target vocab size. Used by GPT-2/3/4, Llama, Mistral.

**Byte-level BPE.** Same algorithm over raw bytes (256 base tokens). Guarantees zero [UNK] tokens. GPT-2 uses 50,257 tokens.

**Unigram.** Start with a huge vocabulary. Iteratively prune tokens whose removal least increases corpus log-likelihood. Used by T5, mBART, Gemma.

**WordPiece.** Merge pairs that maximize likelihood rather than raw frequency. Used by BERT, DistilBERT.

**SentencePiece vs tiktoken.** SentencePiece trains vocabularies (BPE or Unigram) directly on raw Unicode text. tiktoken is OpenAI's fast encoder against pre-built vocabularies.

## Build It

### Step 1: BPE from Scratch

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Three facts: `</w>` marks word end, frequency weighting makes high-frequency pairs win early, merge list is ordered.

### Step 2: Encode with the Learned Merges

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

### Step 3: SentencePiece in Practice

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995,
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

No pre-tokenization required. Space encoded as `▁`. `character_coverage` controls preservation of rare characters.

### Step 4: tiktoken for OpenAI-Compatible Vocabs

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Encoding-only. Fast (Rust backend). Exact match with GPT-4/5 tokenization.

## Pitfalls

- **Tokenizer drift.** Training on vocab A, deploying against vocab B. Check `tokenizer.json` hash in CI.
- **Whitespace ambiguity.** "hello" vs " hello" produce different tokens.
- **Multilingual undertraining.** English-heavy corpora produce vocabs that split non-Latin scripts into 5-10x more tokens.
- **Emoji splits.** A single emoji can take 5 tokens.

## Use It

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram) |
| Serving an OpenAI-compatible API | tiktoken (o200k_base) |
| Domain-specific vocab | Train custom BPE on domain corpus |

Vocabulary size heuristic: 32k for <1B params, 50-100k for 1-10B, 200k+ for multilingual/frontier.

## Exercises

1. **Easy.** Train a 500-merge BPE on a tiny corpus. Encode three held-out words.
2. **Medium.** Compare token counts on 100 English Wikipedia sentences between cl100k_base, o200k_base, and a SentencePiece BPE.
3. **Hard.** Train the same corpus with BPE, Unigram, and WordPiece. Measure downstream accuracy on a small sentiment classifier.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| BPE | Greedy merge of most-frequent character pairs until target vocab size. |
| Byte-level BPE | BPE over raw 256 bytes. No unknown tokens ever. |
| Unigram | Prunes from large candidate set using log-likelihood. |
| SentencePiece | Library training BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | OpenAI's Rust-backed BPE encoder for pre-built vocabs. |
| Merge list | Ordered list of `(a, b) → ab` merges. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/19-subword-tokenization)
