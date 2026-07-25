# vLLM Serving Internals: PagedAttention, Continuous Batching, Chunked Prefill

> vLLM's dominance in 2026 rests on three compounding defaults, not a single trick. PagedAttention is always on. Continuous batching injects new requests into the active batch between decode iterations. Chunked prefill slices long prompts so decode tokens never starve. Turn all three on and a Llama 3.3 70B FP8 on one H100 SXM5 pushes 2,200-2,400 tok/s at 128 concurrent — roughly 25% above vLLM's own default and 3-4x a naive PyTorch loop.

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## Learning Objectives

- Explain PagedAttention as a KV cache allocator: blocks, block tables, and why fragmentation stays under 4% at production load.
- Diagram continuous batching at the iteration level: how finished sequences leave the batch and new ones join without draining.
- Describe chunked prefill in one sentence and name which latency metric it protects (TTFT tail, not mean throughput).
- Name the 2026 vLLM v0.18.0 gotcha that bites teams enabling every optimization at once.

## The Problem

A naive PyTorch serve loop runs one request at a time: tokenize, prefill, decode until EOS, return. At one user this works. At one hundred, it is a queue. The obvious fix — static batching — pads every request to the longest prompt, pads every decode to the longest expected output, and stalls the whole batch on the slowest sequence.

vLLM solves three problems at once. PagedAttention stops KV cache fragmentation from eating 60-80% of GPU memory. Continuous batching lets requests join and leave the batch between each decode iteration. Chunked prefill breaks a 32k-token prompt into ~512-token slices that interleave with decode.

## The Concept

### PagedAttention as a virtual memory system

A KV cache is `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` per sequence. For Llama 3.3 70B at 8192 tokens, that is roughly 1.25 GB per sequence in BF16. If you pre-reserve 8192 slots for every request but the average request only uses 1500 tokens, you waste roughly 82% of the HBM.

PagedAttention borrows the idea from OS virtual memory. KV cache is not contiguous per sequence. It is allocated in fixed-size blocks (default 16 tokens). Each sequence has a block table that maps its logical token positions to physical block IDs. Fragmentation drops from 60-80% (classic) to under 4%.

### Continuous batching at the iteration level

Old "dynamic batching" waited for a window to fill a batch, then ran prefill + decode + decode + decode until every sequence finished. Continuous batching operates between each decode step:

1. Any sequence in `RUNNING` that just hit EOS or max_tokens is removed.
2. If there are free KV blocks, admit new sequences from the waiting queue.
3. The forward pass runs on whatever is now in `RUNNING`, emitting one new token per sequence.

### Chunked prefill protects TTFT tail

A 32k-token prompt on Llama 3.3 70B takes ~800 ms of pure prefill on one H100. While prefill runs, decode tokens for every other sequence wait. Chunked prefill splits prefill into fixed-size chunks (default 512 tokens) and schedules each chunk as a unit. P99 ITL under mixed load drops from ~50 ms to ~15 ms.

### The three defaults interact

All three features assume each other. PagedAttention gives the scheduler a fine-grained KV resource. Continuous batching needs that fine-grained resource. Chunked prefill is a decision the scheduler makes on the same `RUNNING` list.

### The 2026 v0.18.0 gotcha

In vLLM v0.18.0 you cannot combine `--enable-chunked-prefill` with draft-model speculative decoding (`--speculative-model`). The documented exception is N-gram GPU speculative decoding in the V1 scheduler.

### Numbers you should remember

- Llama 3.3 70B FP8, H100 SXM5, 128 concurrent, all three on: 2,200-2,400 tok/s.
- Default vLLM (no chunked prefill): ~1,800 tok/s.
- Naive PyTorch forward loop: ~600 tok/s.
- KV fragmentation under PagedAttention: <4%.
- P99 ITL with chunked prefill: ~15 ms; without: ~50 ms.

### What the scheduler looks like

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))
        else:
            batch.append(decode_one_token(s))

    run_forward(batch)
```

## Use It

`code/main.py` simulates a vLLM-style scheduler with toggleable features.

```python
"""Toy continuous-batching scheduler — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import random
import statistics

FORWARD_LATENCY_PER_TOKEN = 0.0005
PREFILL_LATENCY_PER_TOKEN = 0.00004
BATCH_OVERHEAD = 0.0002
CHUNK_SIZE = 512
KV_BLOCK_SIZE = 16
KV_BLOCKS_AVAILABLE = 1800

@dataclass
class Request:
    req_id: int
    prompt_len: int
    output_len: int
    arrived_at: float
    prefilled: int = 0
    generated: int = 0
    ttft: float | None = None
    last_token_at: float | None = None
    itl_samples: list[float] = field(default_factory=list)

    @property
    def in_prefill(self) -> bool:
        return self.prefilled < self.prompt_len

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len

    def blocks_needed(self) -> int:
        total = self.prompt_len + self.output_len
        return (total + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

def make_workload(n: int = 60, seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    now = 0.0
    for i in range(n):
        now += rng.expovariate(40.0)
        prompt_len = rng.choice([128, 256, 512, 2048, 8192])
        out_len = rng.randint(50, 300)
        reqs.append(Request(i, prompt_len, out_len, now))
    return reqs

def report(label: str, reqs: list[Request], sim_end: float) -> None:
    ttfts = [r.ttft - r.arrived_at for r in reqs if r.ttft is not None]
    itls = [dt for r in reqs for dt in r.itl_samples]
    total_out = sum(r.generated for r in reqs)
    throughput = total_out / sim_end if sim_end else 0
    mean_ttft = statistics.mean(ttfts) * 1000 if ttfts else 0
    p99_itl = sorted(itls)[int(0.99 * len(itls)) - 1] * 1000 if itls else 0
    print(f"{label:28}  throughput={throughput:6.0f} tok/s   mean_TTFT={mean_ttft:6.1f} ms   P99_ITL={p99_itl:5.1f} ms   finished={sum(r.done for r in reqs)}/{len(reqs)}")

def simulate_naive(reqs: list[Request]) -> float:
    now = 0.0
    for r in reqs:
        if now < r.arrived_at:
            now = r.arrived_at
        now += r.prompt_len * PREFILL_LATENCY_PER_TOKEN + BATCH_OVERHEAD
        r.prefilled = r.prompt_len
        r.ttft = now
        r.last_token_at = now
        for _ in range(r.output_len):
            prev = r.last_token_at
            now += FORWARD_LATENCY_PER_TOKEN + BATCH_OVERHEAD
            r.generated += 1
            r.itl_samples.append(now - prev)
            r.last_token_at = now
    return now

def simulate_static(reqs: list[Request], batch: int = 16) -> float:
    now = 0.0
    i = 0
    while i < len(reqs):
        window = reqs[i:i + batch]
        i += batch
        now = max(now, max(r.arrived_at for r in window))
        pad_prompt = max(r.prompt_len for r in window)
        pad_output = max(r.output_len for r in window)
        now += pad_prompt * PREFILL_LATENCY_PER_TOKEN + BATCH_OVERHEAD
        for r in window:
            r.prefilled = r.prompt_len
            r.ttft = now
            r.last_token_at = now
        for _ in range(pad_output):
            prev_now = now
            now += FORWARD_LATENCY_PER_TOKEN * len(window) / 16 + BATCH_OVERHEAD
            for r in window:
                if r.generated < r.output_len:
                    r.generated += 1
                    r.itl_samples.append(now - prev_now)
                    r.last_token_at = now
    return now

def simulate_continuous(reqs: list[Request], chunked: bool) -> float:
    waiting = deque(sorted(reqs, key=lambda r: r.arrived_at))
    running: list[Request] = []
    blocks_used = 0
    now = 0.0
    while waiting or running:
        while waiting and waiting[0].arrived_at <= now:
            r = waiting[0]
            if blocks_used + r.blocks_needed() > KV_BLOCKS_AVAILABLE:
                break
            blocks_used += r.blocks_needed()
            running.append(waiting.popleft())
        if not running:
            if not waiting:
                break
            now = waiting[0].arrived_at
            continue
        batch_tokens = 0
        prefill_work = 0
        decoded: list[Request] = []
        for r in running:
            if r.in_prefill:
                remaining = r.prompt_len - r.prefilled
                take = min(CHUNK_SIZE if chunked else remaining, remaining)
                r.prefilled += take
                prefill_work += take
                if r.prefilled >= r.prompt_len:
                    r.ttft = now + prefill_work * PREFILL_LATENCY_PER_TOKEN
            else:
                decoded.append(r)
                batch_tokens += 1
        dt = (prefill_work * PREFILL_LATENCY_PER_TOKEN + batch_tokens * FORWARD_LATENCY_PER_TOKEN + BATCH_OVERHEAD)
        now += dt
        for r in decoded:
            prev = r.last_token_at or r.ttft or now
            r.generated += 1
            r.itl_samples.append(now - prev)
            r.last_token_at = now
            if r.ttft is None:
                r.ttft = now
        finished = [r for r in running if r.done]
        for r in finished:
            blocks_used -= r.blocks_needed()
            running.remove(r)
    return now

def main() -> None:
    print("=" * 80)
    print("TOY vLLM SCHEDULER — four modes on the same 60-request workload")
    print("=" * 80)
    base = make_workload()
    w1 = [Request(r.req_id, r.prompt_len, r.output_len, r.arrived_at) for r in base]
    report("NAIVE", w1, simulate_naive(w1))
    w2 = [Request(r.req_id, r.prompt_len, r.output_len, r.arrived_at) for r in base]
    report("STATIC (batch=16, padded)", w2, simulate_static(w2))
    w3 = [Request(r.req_id, r.prompt_len, r.output_len, r.arrived_at) for r in base]
    report("CONTINUOUS (no chunk)", w3, simulate_continuous(w3, chunked=False))
    w4 = [Request(r.req_id, r.prompt_len, r.output_len, r.arrived_at) for r in base]
    report("CONTINUOUS + CHUNKED", w4, simulate_continuous(w4, chunked=True))

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-vllm-scheduler-reader.md`. Given a serving config, it produces a scheduler diagnosis that names which of the three defaults is bottlenecking.

## Exercises

1. Run `code/main.py`. Compare `STATIC` to `CONTINUOUS` on mixed short and long requests.
2. Modify the toy scheduler to add `--max-num-batched-tokens`.
3. Re-read the vLLM v0.18.0 release notes — which flag combinations are mutually exclusive?
4. Compute the KV cache fragmentation waste for 1,000 requests with mean 1,500 output tokens, std 600.
5. Explain in one paragraph why chunked prefill helps P99 ITL but not throughput in isolation.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PagedAttention | "the KV trick" | Fixed-size block allocator for KV cache; fragmentation <4% |
| Continuous batching | "dynamic batching, but right" | Admit/release decisions made every decode iteration |
| Chunked prefill | "prefill splitting" | Break long prefill into 512-token slices interleaved with decode |
| TTFT | "first token time" | Prefill + queue + network; dominated by prefill at long prompts |
| Goodput | "throughput that meets SLO" | Tokens/sec where every request hit TTFT and ITL targets |
| V1 scheduler | "the new scheduler" | vLLM's 2026 scheduler |
| `--gpu-memory-utilization` | "the memory knob" | Fraction of HBM reserved for KV blocks after weights and activations |

## Further Reading

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/04-vllm-serving-internals)
