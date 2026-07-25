# SGLang and RadixAttention for Prefix-Heavy Workloads

> SGLang treats the KV cache as a first-class, reusable resource stored in a radix tree. Where vLLM schedules requests FCFS, SGLang's cache-aware scheduler prioritizes requests with longer shared prefixes — effectively a depth-first radix traversal so hot branches stay resident in HBM. On Llama 3.1 8B with ShareGPT-like 1K prompts, SGLang hits ~16,200 tok/s to vLLM's ~12,500, a ~29% edge. On prefix-heavy RAG workloads the advantage reaches 6.4x.

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## Learning Objectives

- Diagram RadixAttention: how prefixes are stored in a radix tree and how KV blocks are shared across sequences rooted at the same branch.
- Explain cache-aware scheduling and why FCFS is wrong for prefix-heavy traffic.
- Compute expected speedup for a workload given prefix-cache hit rate and prompt length distribution.
- Name the prompt-ordering discipline that makes the 6.4x number real vs a lost upside.

## The Problem

Classic serving treats each request's prompt as opaque. Even when 5,000 RAG requests all start with the same 2,000-token system prompt plus same retrieval preamble, vLLM prefills that 2,000-token prefix 5,000 times. The GPU does the same work over and over.

RadixAttention does exactly this. Tokens are indexed in a radix tree; each node owns KV blocks for the token sequence on its path from root. A new request walks the tree: any node whose token matches re-uses that node's KV blocks. Prefill cost becomes proportional to the "new" suffix, not the full prompt.

## The Concept

### The radix tree as a KV index

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

A new request with system prompt + "Context: <doc A>" + "Question: Carol" walks: system prefix matches (124 blocks reused), doc-A branch matches (31 blocks reused), then allocates fresh blocks only for "Question: Carol" (4 blocks). ~40x savings on prefill.

### Cache-aware scheduling

1. **Depth-first dispatch** — prefer requests rooted at the same branch as the current running set.
2. **LRU at branch level, not block level** — evict whole branches rather than individual blocks.

### Benchmark numbers

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% edge).
- Prefix-heavy RAG: up to 6.4x on SGLang.
- Voice cloning workloads: 86.4% prefix-cache hit rate.
- Deployed on 400,000+ GPUs in 2026.

### The ordering gotcha

The 6.4x number relies on consistent prompt-template ordering. If your client constructs prompts with dynamic interleaved content, the tree cannot find the shared prefix. Fix the order: immutable first (system, tools), then retrieval context, then user question.

## Use It

`code/main.py` implements a toy radix-tree KV cache plus a scheduler with two policies.

```python
"""Toy RadixAttention scheduler — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import random

KV_BUDGET_BLOCKS = 160
BLOCK_TOKENS = 16

def token_count(seg: str) -> int:
    if seg == "SYSTEM": return 2000
    if seg.startswith("DOC_"): return 500
    if seg.startswith("Q_"): return 60
    if seg == "TOOLS": return 300
    return 100

@dataclass
class Request:
    rid: int
    segments: list[str]

class RadixCache:
    def __init__(self, budget_blocks: int = KV_BUDGET_BLOCKS):
        self.budget = budget_blocks
        self.used = 0
        self.time = 0
        self.nodes: dict[tuple[str, ...], list[int]] = {}

    def walk(self, segments: list[str]) -> int:
        reused = 0
        self.time += 1
        for i in range(1, len(segments) + 1):
            key = tuple(segments[:i])
            if key in self.nodes:
                reused += token_count(segments[i - 1])
                self.nodes[key][1] = self.time
            else:
                break
        return reused

    def insert(self, segments: list[str]) -> None:
        for i in range(1, len(segments) + 1):
            key = tuple(segments[:i])
            if key in self.nodes:
                continue
            blocks = (token_count(segments[i - 1]) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            while self.used + blocks > self.budget and self._evict_one():
                pass
            self.nodes[key] = [blocks, self.time]
            self.used += blocks

    def _evict_one(self) -> bool:
        leaves = [k for k in self.nodes if not any(
            other != k and other[:len(k)] == k for other in self.nodes)]
        if not leaves:
            return False
        victim = min(leaves, key=lambda k: self.nodes[k][1])
        self.used -= self.nodes.pop(victim)[0]
        return True

def simulate(requests: list[Request], scheduler: str) -> dict:
    cache = RadixCache()
    if scheduler == "CACHE_AWARE":
        branch_count: dict[tuple[str, ...], int] = defaultdict(int)
        for r in requests:
            for i in range(1, len(r.segments) + 1):
                branch_count[tuple(r.segments[:i])] += 1
        def score(r: Request) -> int:
            return max(branch_count[tuple(r.segments[:i])] * sum(
                token_count(s) for s in r.segments[:i]) for i in range(1, len(r.segments) + 1))
        order = sorted(requests, key=score, reverse=True)
    else:
        order = list(requests)
    saved = 0
    total = 0
    for r in order:
        prompt_tokens = sum(token_count(s) for s in r.segments)
        total += prompt_tokens
        reused = cache.walk(r.segments)
        saved += reused
        cache.insert(r.segments)
    return {"hit_rate": saved / total if total else 0, "saved": saved, "total": total}

def workload_rag(n: int = 80, docs: int = 4, seed: int = 1) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        doc = f"DOC_{rng.randrange(docs)}"
        q = f"Q_{i}"
        reqs.append(Request(i, ["SYSTEM", "TOOLS", doc, q]))
    rng.shuffle(reqs)
    return reqs

def workload_scrambled(n: int = 80, docs: int = 4, seed: int = 1) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        doc = f"DOC_{rng.randrange(docs)}"
        q = f"Q_{i}"
        prefix = ["SYSTEM", "TOOLS", doc]
        rng.shuffle(prefix)
        reqs.append(Request(i, prefix + [q]))
    rng.shuffle(reqs)
    return reqs

def main() -> None:
    print("=" * 88)
    print("TOY RADIX CACHE — cache hit rate across schedulers and orderings")
    print("=" * 88)
    report("RAG workload | FCFS", simulate(workload_rag(), "FCFS"))
    report("RAG workload | CACHE_AWARE", simulate(workload_rag(), "CACHE_AWARE"))
    report("RAG scrambled prefix | FCFS", simulate(workload_scrambled(), "FCFS"))
    report("RAG scrambled prefix | CACHE_AWARE", simulate(workload_scrambled(), "CACHE_AWARE"))

def report(label: str, res: dict) -> None:
    print(f"{label:44}  hit_rate={res['hit_rate']:6.1%}   saved={res['saved']:>6}/{res['total']:<6} tok")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-radix-scheduler-advisor.md`. Given workload description, produces a prompt-ordering prescription and go/no-go for SGLang adoption.

## Exercises

1. Run `code/main.py`. Compare FCFS and cache-aware — where does the delta come from?
2. Modify prompts to randomly permute [system, tools, context]. What happens to hit rate?
3. Compute HBM cost of a 2,000-token system prompt as one radix branch on Llama 3.1 8B.
4. Read the SGLang RadixAttention paper. Explain tree-shaped LRU vs block-shaped LRU.
5. Customer reports 8% cache hit rate. Name three likely causes.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| RadixAttention | "the SGLang thing" | KV cache indexed as a radix tree so shared prefixes reuse blocks |
| Radix tree | "compact trie" | Tree where each node owns a token range and its KV blocks |
| Cache-aware scheduler | "hot-branch-first" | Scheduler that prefers requests sharing the resident branch |
| Prefix-cache hit rate | "how much of your prompt was free" | Fraction of prompt tokens served from reused KV blocks |
| FCFS | "first-come first-served" | Default scheduling that breaks prefix locality |
| Prompt template ordering | "the cache key" | The prompt's component order determines what the tree can share |

## Further Reading

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang documentation](https://sgl-project.github.io/)
- [SGLang paper (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/06-sglang-radixattention)
