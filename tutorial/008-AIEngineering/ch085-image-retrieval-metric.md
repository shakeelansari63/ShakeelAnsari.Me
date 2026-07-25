# Image Retrieval & Metric Learning

> A retrieval system ranks candidates by a distance in embedding space. Metric learning is the discipline of shaping that space so the distances mean what you want.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP)
**Time:** ~45 minutes

## Learning Objectives

- Explain triplet, contrastive, and proxy-based losses; pick the right one
- Implement L2-normalisation and cosine similarity correctly
- Build a FAISS index; query by text and image; report recall@K
- Use DINOv2/CLIP as off-the-shelf embedding backbones

## The Problem

Retrieval is everywhere: duplicate detection, visual search, face re-ID. The hard part is defining what counts as similar.

## The Concept

### The four loss families

| Loss | Pros | Cons |
|------|------|------|
| Contrastive | Simple | Needs many negatives |
| Triplet | Intuitive margin control | Hard mining expensive |
| NT-Xent/InfoNCE | Scales to large batches | Needs big batch |
| Proxy-based (ProxyNCA) | Fast, stable, no mining | Proxy overfitting on small data |

### Triplet loss

```
L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
```

### Cosine similarity vs L2

`||a - b||^2 = 2 - 2 cos(a, b)` when L2-normalised. Pick one convention and stick to it.

### Recall@K

Fraction of queries with at least one correct match in top K.

### FAISS

- IndexFlatIP: brute force, up to ~1M vectors
- IndexIVFFlat: approximate, fast
- IndexHNSW: graph-based, fastest for many queries

## Build It

### Step 1: Triplet loss

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()
```

### Step 2: Semi-hard mining

```python
def semi_hard_negatives(emb, labels, margin=0.2):
    dist = torch.cdist(emb, emb)
    same_class = labels[:, None] == labels[None, :]
    diff_class = ~same_class
    N = emb.size(0)

    positives = dist.clone()
    positives[~same_class] = float("-inf")
    positives.fill_diagonal_(float("-inf"))
    pos_idx = positives.argmax(dim=1)

    semi_hard = dist.clone()
    semi_hard[same_class] = float("inf")
    d_ap = dist[torch.arange(N), pos_idx].unsqueeze(1)
    semi_hard[dist <= d_ap] = float("inf")
    neg_idx = semi_hard.argmin(dim=1)

    fallback_mask = semi_hard[torch.arange(N), neg_idx] == float("inf")
    if fallback_mask.any():
        hardest = dist.clone()
        hardest[same_class] = float("inf")
        neg_idx = torch.where(fallback_mask, hardest.argmin(dim=1), neg_idx)
    return pos_idx, neg_idx
```

### Step 3: Recall@K

```python
def recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=1):
    sim = query_emb @ gallery_emb.T
    _, top_k = sim.topk(k, dim=-1)
    matches = (gallery_labels[top_k] == query_labels[:, None]).any(dim=-1)
    return matches.float().mean().item()
```

### Step 4: Putting it together

```python
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, in_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

torch.manual_seed(0)
num_classes = 6
protos = F.normalize(torch.randn(num_classes, 128), dim=-1)

def sample_batch(bs=32):
    labels = torch.randint(0, num_classes, (bs,))
    x = protos[labels] + 0.15 * torch.randn(bs, 128)
    return x, labels

enc = Encoder()
opt = Adam(enc.parameters(), lr=3e-3)

for step in range(200):
    x, y = sample_batch(32)
    emb = enc(x)
    pos_idx, neg_idx = semi_hard_negatives(emb, y)
    loss = triplet_loss(emb, emb[pos_idx], emb[neg_idx])
    opt.zero_grad(); loss.backward(); opt.step()
```

## Use It

- DINOv2 + FAISS: general visual retrieval
- CLIP + FAISS: text queries
- Fine-tuned DINOv2 + FAISS: instance-level retrieval
- Managed: Milvus, Weaviate, Qdrant

## Ship It

- `outputs/prompt-retrieval-loss-picker.md`
- `outputs/skill-recall-at-k-runner.md`

## Exercises

1. **(Easy)** Plot PCA of embeddings before/after training for toy data.
2. **(Medium)** Add ProxyNCA loss; compare convergence vs triplet.
3. **(Hard)** Embed 1k ImageNet val with DINOv2, build FAISS index, report recall@{1,5,10}.

## Key Terms

## Further Reading

- [FaceNet (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832)
- [Triplet Loss for Person Re-ID (Hermans et al., 2017)](https://arxiv.org/abs/1703.07737)
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/20-image-retrieval-metric)
