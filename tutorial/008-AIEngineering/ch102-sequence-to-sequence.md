# Sequence-to-Sequence Models

Two RNNs pretending to be a translator. The bottleneck they hit is the reason attention exists.

Classification maps a variable-length sequence to a single label. Translation maps a variable-length sequence to another of different length, with no guarantee of length parity. The seq2seq architecture (Sutskever, Vinyals, Le, 2014) cracked this with two RNNs: one reads the source, the other generates the target token by token.

## The Concept

**Encoder.** An RNN that reads the source sentence. Its final hidden state is the **context vector** — a fixed-size summary of the entire input.

**Decoder.** Another RNN initialized from the context vector. At each step it takes the previously generated token and produces a distribution over the target vocabulary. Repeat until `<EOS>`.

**Teacher forcing.** During training, the decoder's input at step `t` is the ground-truth token, not its own previous prediction. This stabilizes training. The gap between training and inference is called **exposure bias**.

**The bottleneck.** Everything the encoder learned must be squeezed into one context vector. Long sentences lose detail. Attention (lesson 10) fixes this by letting the decoder look at every encoder hidden state.

## Build It

### Step 1: An Encoder

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

### Step 2: A Decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

### Step 3: Training Loop with Teacher Forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

### Step 4: Inference Loop (Greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

**Beam search** keeps the top-k partial sequences alive and picks the highest-scoring complete one at the end. Beam width 3-5 is standard.

### Step 5: The Bottleneck, Demonstrated

Train on a copy task. Results show the bottleneck:

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

A single GRU hidden state cannot losslessly memorize a 40-token input.

## Use It

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

### Exposure Bias Mitigations

Scheduled sampling (anneal teacher forcing), minimum risk training (train on BLEU), reinforcement learning fine-tuning (RLHF for LLMs).

## Exercises

1. **Easy.** Implement the toy copy task. Measure accuracy at lengths 5, 10, 20.
2. **Medium.** Add beam search with width 3. Measure BLEU against greedy.
3. **Hard.** Fine-tune `facebook/bart-base` on a 10k-pair paraphrase dataset.

## Key Terms

| Term | What it actually means |
|------|-----------------------|
| Encoder | Reads source. Produces per-step hidden states and final context vector. |
| Decoder | Generates target tokens one at a time from context vector. |
| Context vector | Final encoder hidden state. Fixed-size bottleneck that attention solves. |
| Teacher forcing | Feed ground-truth previous token at training time. |
| Exposure bias | Model trained on true tokens never practiced recovering from own mistakes. |
| Beam search | Keep top-k partial sequences alive instead of greedy commitment. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence)
