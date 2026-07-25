# CNNs and RNNs for Text

Convolutions learn n-grams. Recurrences remember. Both are superseded by attention. Both still matter on constrained hardware.

TF-IDF and Word2Vec produce flat vectors that ignore word order. Two families of architectures filled that gap before transformers arrived.

## The Concept

**TextCNN** (Kim, 2014). Apply 1D convolutions over sequences of word embeddings. A filter of width 3 is a learnable trigram detector. Max-pool to a fixed-size representation. Flat, parallel, fast.

**RNN.** Process tokens one at a time, maintaining a hidden state. Sequential, memory-bearing. Dominated sequence modeling from 2014 to 2017.

**LSTM** adds gates (input, forget, output) and a cell state that stabilizes gradients through long sequences. **GRU** simplifies LSTM to two gates with similar accuracy and fewer parameters.

**Bidirectional RNNs** run one RNN forward and another backward, concatenating hidden states. Essential for tagging tasks.

## Build It

### Step 1: TextCNN in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

The `transpose(1, 2)` reshapes `[batch, seq_len, embed_dim]` to `[batch, embed_dim, seq_len]` for `nn.Conv1d`.

### Step 2: LSTM Classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool over the sequence usually beats taking the last hidden state for classification.

### Step 3: The Vanishing Gradient Demo

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)

# At weight=0.9 over 100 steps: 0.9^100 ≈ 2.7e-5
```

LSTMs fix this with a **cell state** that runs through the network with only additive interactions.

### Why This Still Was Not Enough

1. Sequential bottleneck: RNNs on 1000-length sequences require 1000 serial steps.
2. Fixed-size context vector in encoder-decoder setups compresses all input into one vector.
3. Distant-dependency accuracy ceiling.

Attention solved all three. Transformers dropped recurrence entirely.

## Use It

When it fits the constraint:

- **Edge/on-device inference.** TextCNN + GloVe is 10-100x smaller than a transformer.
- **Streaming/online classification.** RNN processes one token at a time.
- **Tiny models for baselines.** Train a TextCNN in 5 minutes on a CPU.
- **Sequence labeling with limited data.** BiLSTM-CRF is production-grade NER for 1k-10k sentences.

Everything else goes to a transformer.

## Exercises

1. **Easy.** Train a TextCNN on a 3-class toy dataset. Verify multi-filter-width outperforms single width.
2. **Medium.** Compare max-pool, mean-pool, and last-state pooling for LSTM classifier.
3. **Hard.** Build a BiLSTM-CRF NER tagger. Train on CoNLL-2003.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| TextCNN | Stack of 1D convolutions over word embeddings with global max-pool. |
| RNN | Hidden state updated at each timestep: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN with input/forget/output gates and a cell state. |
| GRU | Two gates instead of three. Similar accuracy, fewer params. |
| Bidirectional | Forward + backward concatenated. Both left and right context. |
| Vanishing gradient | Repeated <1 weight multiplication makes early gradients effectively zero. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text)
