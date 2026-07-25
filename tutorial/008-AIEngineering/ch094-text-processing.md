# Text Processing — Tokenization, Stemming, Lemmatization

Language is continuous. Models are discrete. Preprocessing is the bridge.

A model cannot read "The cats were running." It reads integers. Every NLP system opens with three questions: where does a word start, what is the root of the word, and how do we treat "run", "running", "ran" as the same thing when it helps but different when it doesn't.

## The Three Operations

**Tokenization** splits a string into tokens. Word-level for classical NLP, subword for transformers, character for languages without whitespace.

**Stemming** chops suffixes with rules. Fast, aggressive, dumb. `running` → `run`. `organization` → `organ`. Hard failure mode on the second example.

**Lemmatization** reduces a word to its dictionary form using grammar knowledge. Slower, accurate, needs a lookup table. `ran` → `run` (needs to know "ran" is past tense). `better` → `good` (needs to know comparative forms).

Rule of thumb: stem when speed matters and you can tolerate noise (search indexing, rough classification). Lemmatize when meaning matters (question answering, semantic search, anything the user will read).

## Build It

### Regex Word Tokenizer

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Three patterns: words with optional apostrophe (`don't`, `it's`), pure numbers, single non-whitespace non-alphanumeric characters (punctuation).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Failure modes: `3pm` splits to `['3', 'pm']`. URLs, emails, hashtags all break. For production, add patterns before the general ones.

### Porter Stemmer (Step 1a)

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Rules compete. Earlier rules win. Order matters more than any single rule.

### Lookup-based Lemmatizer

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

`watched` is not in our table. Real lemmatization covers `ed`, irregular verbs, comparative adjectives. This is why production uses WordNet, spaCy, or a full morphological analyzer.

### Pipe Them Together

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

## Use It

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)

def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"): return "v"
    if tag.startswith("J"): return "a"
    if tag.startswith("R"): return "r"
    return "n"

lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` handles contractions and unicode. `PorterStemmer` runs all five phases. `WordNetLemmatizer` needs the POS tag translated from Penn Treebank to WordNet abbreviations.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy hides the whole pipeline behind `nlp(text)`. Tokenization, POS tagging, and lemmatization all run. Faster than NLTK at scale.

### When to Pick Which

| Situation | Pick |
|-----------|------|
| Teaching, research, swapping components | NLTK |
| Production, multi-language, speed matters | spaCy |
| Transformer pipeline | Use `tokenizers` / `transformers` and skip classical preprocessing |

## Two Failure Modes Nobody Warns About

**Reproducibility drift.** NLTK and spaCy change behavior between versions. Pin library versions. Write a preprocessing regression test.

**Training / inference mismatch.** Train with aggressive preprocessing, deploy on raw user input, watch performance crater. Ship preprocessing as a function inside the model package, not as a notebook cell.

## Exercises

1. **Easy.** Extend `tokenize` to keep URLs as single tokens.
2. **Medium.** Implement Porter step 1b with the double-consonant rule.
3. **Hard.** Build a lemmatizer using WordNet with Porter fallback. Measure accuracy against plain WordNet and plain Porter.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Token | Whatever unit the model consumes. Word, subword, character, or byte. |
| Stem | Result of rule-based suffix stripping. Not always a real word. |
| Lemma | Dictionary form. Requires grammatical context. |
| POS tag | Grammatical category like NOUN, VERB, ADJ. Needed for lemmatization. |
| Morphology | How a word changes form based on tense, number, case. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/01-text-processing)
