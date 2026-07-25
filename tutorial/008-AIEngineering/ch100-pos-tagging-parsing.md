# POS Tagging and Syntactic Parsing

Grammar was unfashionable for a while. Then every LLM pipeline needed to validate structured extraction, and it came back.

Lesson 01 promised that lemmatization needs a POS tag. Without knowing `running` is a verb, a lemmatizer cannot reduce it to `run`. This lesson introduces the tagsets, the baselines, and the point where you stop implementing from scratch and call spaCy.

## The Concept

**POS tagging** labels each token with a grammatical category. The **Penn Treebank (PTB)** tagset: 36 tags (`NN` singular noun, `NNS` plural noun, `VBD` verb past tense, etc.). The **Universal Dependencies (UD)** tagset is coarser (17 tags) and language-agnostic.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing** produces a tree. Two styles:

- **Constituency parsing.** Noun phrases, verb phrases nest inside each other.
- **Dependency parsing.** Each word has one head word, labeled with a grammatical relation.

Dependency parsing won because it generalizes cleanly across languages.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It

### Step 1: Most-Frequent-Tag Baseline

```python
from collections import Counter, defaultdict

def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag

def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

On the Brown corpus, this hits ~85% accuracy.

### Step 2: Bigram HMM Tagger

Model the joint probability: `P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)`

```python
import math

def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab

def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)

def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        V[0][j] = log_prob(transitions, "<BOS>", tag, tr_denom, alpha) + log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM on Brown hits ~93% accuracy. The jump from 85% to 93% is mostly transition probabilities.

## Use It

Every production NLP library ships POS and dependency parsers.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

### Where This Matters in 2026

Lemmatization (needs POS), structured extraction from LLM outputs, aspect-based sentiment (dependency tells which adjective modifies which noun), query understanding, cross-lingual transfer, low-compute pipelines.

## Exercises

1. **Easy.** Measure most-frequent-tag baseline accuracy on a small tagged corpus (~85%).
2. **Medium.** Train the bigram HMM and report per-tag precision/recall.
3. **Hard.** Use spaCy's dependency parse to extract subject-verb-object triples from 1000 sentences.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| POS tag | Grammatical category. PTB has 36; UD has 17. |
| Penn Treebank | English-specific fine-grained tagset. |
| Universal Dependencies | Multilingual, coarser than PTB. |
| Dependency parse | Each word has one head, each edge a grammatical relation. |
| Viterbi | Dynamic programming for highest-probability tag sequence. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing)
