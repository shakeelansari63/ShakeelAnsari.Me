# Machine Translation

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
