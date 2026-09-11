# GPU Autoscaling, vLLM & Disaggregated Serving

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch361): GPU Autoscaling on Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> Three layers, not one. Karpenter provisions nodes dynamically (under one minute, 40% faster than Cluster Autoscaler). KAI Scheduler handles gang scheduling, topology awareness, and hierarchical queues — it prevents the 7-of-8 partial allocation trap where seven nodes wait and burn on one missing GPU. Application-level autoscalers (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) scale on inference-specific signals — queue depth, KV cache utilization — not CPU/DCGM duty cycle. The classic HPA trap is that `DCGM_FI_DEV_GPU_UTIL` is a duty-cycle measurement: 100% could be 10 requests or 100. vLLM pre-allocates KV cache memory, so memory never triggers scale-down.

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- Diagram the three autoscaling layers (node provisioning, gang scheduling, application-level) and name the tool used at each layer.
- Explain why `DCGM_FI_DEV_GPU_UTIL` is the wrong HPA signal for vLLM and name two replacements (queue depth, KV cache utilization).
- Describe gang scheduling and the partial-allocation failure mode KAI Scheduler prevents (7 of 8 GPUs idle).
- Name the Karpenter consolidation policy (`WhenEmptyOrUnderutilized`) that terminates running GPU jobs and state the 2026 safe alternative.

## The Problem

Your team ships an LLM-serving service on Kubernetes. You set up HPA with `DCGM_FI_DEV_GPU_UTIL` as the signal. The service pins at 100% utilization during business hours. HPA never scales up — it already thinks you're full. You add a replica manually; TTFT drops. HPA still doesn't scale. The signal is lying to you.

Separately, you use Cluster Autoscaler for nodes. A 1M-token prompt arrives at 2 a.m.; the cluster spends 3 minutes provisioning a node, and the request times out.

Separately again, you deploy a 70B model requiring 8 GPUs across 2 nodes. The cluster has 7 GPUs free and 1 spread across 3 nodes. Cluster Autoscaler provisions a node for the 1 missing GPU. Seven nodes wait 4 minutes burning money while Kubernetes gets the last GPU up.

Three layers, three different failure modes. GPU-aware autoscaling in 2026 is not "turn on HPA." It's composing node provisioning, gang scheduling, and application-signal autoscaling.

## The Concept

### Layer 1 — node provisioning (Karpenter)

Karpenter watches pending pods and provisions nodes within ~45-60 seconds (Cluster Autoscaler typically takes 90-120 seconds for GPU nodes). It picks instance types dynamically per the `NodePool` constraint — if your pod needs 8 H100s and the cluster has no matching node, Karpenter provisions one directly instead of scaling an existing group.

**The consolidation trap**: Karpenter's default `consolidationPolicy: WhenEmptyOrUnderutilized` is dangerous for GPU pools. It will terminate a running GPU node to migrate pods to a cheaper right-sized instance. For inference workloads that means evicting running requests and reloading a 70B model on the new node.

Safe setting for GPU pools:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

### Layer 2 — gang scheduling (KAI Scheduler)

**Gang scheduling** — schedule all-or-nothing. A distributed inference pod requiring 8 GPUs either all 8 start together or none do. Without this, you get the partial-allocation trap: 7 of 8 pods start, wait indefinitely, burn money.

**Topology awareness** — know which GPUs share NVLink, which sit on the same rack, which have InfiniBand between them. Place pods accordingly. A DeepSeek-V3 67B tensor-parallel workload must stay on one NVLink domain; KAI Scheduler respects that.

**Hierarchical queues** — multiple teams compete for the same GPU pool with priority and quota. Team A's production pinch gets preempted by Team B's training job only if priority rules allow.

### Layer 3 — application-level signals

**The HPA trap**: `DCGM_FI_DEV_GPU_UTIL` is a duty-cycle metric — it measures whether the GPU was doing work at each sampling interval. 100% utilization could mean 10 concurrent requests or 100; the GPU was busy either way.

**2026 replacement signals**:

- Queue depth (number of requests waiting for prefill).
- KV cache utilization (what fraction of blocks are allocated to active sequences).
- Per-replica P99 TTFT (your SLA signal).
- Goodput (requests meeting all SLOs per second).

NVIDIA Dynamo Planner and llm-d Workload Variant Autoscaler consume these signals and scale replicas.

### When to use what

| Scale decision | Tool |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA (or custom HPA on queue depth) |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Disaggregated prefill/decode complicates everything

If you run disaggregated prefill/decode, you have two pod classes with different scaling triggers: prefill pods scale on queue depth, decode pods scale on KV cache pressure.

### Cold start matters here too

Cold-start mitigation is where node provisioning time becomes user-visible. Karpenter's 45-60 second warm-up plus a 20GB model load plus engine init means a from-zero request takes 2-5 minutes.

### Numbers you should remember

- Karpenter node provisioning: ~45-60s vs Cluster Autoscaler ~90-120s (GPU nodes).
- KAI Scheduler prevents partial-allocation waste — 7-of-8 trap.
- `DCGM_FI_DEV_GPU_UTIL` as HPA signal: broken; use queue depth or KV utilization.
- Karpenter `WhenEmptyOrUnderutilized`: terminates running GPU jobs. Use `WhenEmpty + consolidateAfter: 1h` for inference.

## Use It

`code/main.py` simulates a three-layer autoscaler on a bursty GPU workload. Compares naive HPA (duty cycle), queue-depth HPA, and KAI-gang-scheduled scaling.

```python
"""Three-layer GPU autoscaling simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass
import random

NODE_PROVISION_SEC = 50
CLUSTER_AUTOSCALER_SEC = 110
MODEL_LOAD_SEC = 45
REQUEST_PREFILL_SEC = 0.6
REQUEST_DECODE_SEC = 1.8
MIN_WARM_REPLICAS = 1
MAX_REPLICAS = 16
GPU_PER_REPLICA = 1
HPA_TICK_SEC = 15
TARGET_GPU_UTIL = 70

@dataclass
class Request:
    arrived_at: float
    started_at: float | None = None
    completed_at: float | None = None
    dropped: bool = False

def make_workload(duration_sec: int = 3600, seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    for _ in range(int(duration_sec)):
        t = _
        if t < 600:
            rate = 0.5
        elif t < 1800:
            rate = 4.0
        else:
            rate = 1.2
        if rng.random() < rate / 10:
            reqs.append(Request(arrived_at=float(t)))
    return reqs

def simulate(strategy: str, reqs: list[Request]) -> dict:
    replicas_ready = MIN_WARM_REPLICAS
    replicas_target = MIN_WARM_REPLICAS
    replica_available_at = {i: 0.0 for i in range(MIN_WARM_REPLICAS)}
    queue: list[Request] = []
    reqs = sorted(reqs, key=lambda r: r.arrived_at)
    cursor = 0
    now = 0.0
    sim_end = max(r.arrived_at for r in reqs) + 60
    idle_gpu_sec = 0.0
    pending_replicas: list[tuple[float, int]] = []
    next_replica_id = MIN_WARM_REPLICAS
    peak_replicas = replicas_ready

    while now < sim_end:
        while cursor < len(reqs) and reqs[cursor].arrived_at <= now:
            queue.append(reqs[cursor])
            cursor += 1
        for ready_at, rid in list(pending_replicas):
            if ready_at <= now:
                replica_available_at[rid] = now
                replicas_ready += 1
                pending_replicas.remove((ready_at, rid))

        free_replicas = [rid for rid, t in replica_available_at.items() if t <= now]
        for rid in free_replicas:
            if queue:
                r = queue.pop(0)
                r.started_at = now
                service_time = REQUEST_PREFILL_SEC + REQUEST_DECODE_SEC
                r.completed_at = now + service_time
                replica_available_at[rid] = r.completed_at
            else:
                idle_gpu_sec += HPA_TICK_SEC

        if strategy == "DUTY_CYCLE":
            pending_ids = {rid for _, rid in pending_replicas}
            busy = sum(1 for rid, t in replica_available_at.items() if t > now and rid not in pending_ids)
            util = busy / max(replicas_ready, 1) * 100
            if util > TARGET_GPU_UTIL and replicas_target < MAX_REPLICAS:
                replicas_target += 1
            elif util < 20 and replicas_target > MIN_WARM_REPLICAS:
                replicas_target -= 1
        elif strategy == "QUEUE_DEPTH":
            qd = len(queue)
            if qd > 5 and replicas_target < MAX_REPLICAS:
                replicas_target = min(MAX_REPLICAS, replicas_target + max(1, qd // 5))
            elif qd == 0 and replicas_target > MIN_WARM_REPLICAS:
                replicas_target = max(MIN_WARM_REPLICAS, replicas_target - 1)
        elif strategy == "KAI_GANG":
            qd = len(queue)
            if qd > 3 and replicas_target < MAX_REPLICAS:
                replicas_target = min(MAX_REPLICAS, replicas_target + max(2, qd // 3))
            elif qd == 0 and replicas_target > MIN_WARM_REPLICAS:
                replicas_target = max(MIN_WARM_REPLICAS, replicas_target - 1)

        while replicas_ready + len(pending_replicas) < replicas_target:
            ready_at = now + NODE_PROVISION_SEC + MODEL_LOAD_SEC
            pending_replicas.append((ready_at, next_replica_id))
            replica_available_at[next_replica_id] = ready_at
            next_replica_id += 1
        peak_replicas = max(peak_replicas, replicas_ready + len(pending_replicas))

        if replicas_ready > replicas_target:
            idle = [rid for rid, t in replica_available_at.items() if t <= now]
            if idle:
                replica_available_at.pop(idle[0])
                replicas_ready -= 1

        for r in queue[:]:
            if now - r.arrived_at > 30:
                r.dropped = True
                queue.remove(r)

        now += HPA_TICK_SEC

    dropped = sum(1 for r in reqs if r.dropped)
    completed = sum(1 for r in reqs if r.completed_at is not None)
    started = [r for r in reqs if r.started_at is not None]
    mean_wait = sum(r.started_at - r.arrived_at for r in started) / len(started) if started else 0.0
    return {"strategy": strategy, "total": len(reqs), "completed": completed, "dropped": dropped,
            "mean_wait_s": mean_wait, "idle_gpu_min": idle_gpu_sec / 60, "peak_replicas": peak_replicas}

def report(row: dict) -> None:
    print(f"{row['strategy']:14}  reqs={row['total']:4}  done={row['completed']:4}  dropped={row['dropped']:3}  "
          f"mean_wait={row['mean_wait_s']:5.1f}s  idle_gpu={row['idle_gpu_min']:6.1f}min  peak={row['peak_replicas']:2}")

def main() -> None:
    print("=" * 80)
    print("GPU AUTOSCALING — three strategies on a bursty workload (1-hour sim)")
    print("=" * 80)
    base = make_workload()
    for strategy in ("DUTY_CYCLE", "QUEUE_DEPTH", "KAI_GANG"):
        reqs = [Request(arrived_at=r.arrived_at) for r in base]
        result = simulate(strategy, reqs)
        report(result)
    print("\nDUTY_CYCLE drops requests because DCGM_FI_DEV_GPU_UTIL is a duty-cycle metric.")
    print("QUEUE_DEPTH reacts to the actual backlog.")
    print("KAI_GANG scales more aggressively and avoids partial-alloc stalls.")

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-gpu-autoscaler-plan.md`. Given cluster topology, workload shape, and SLO, it designs a three-layer autoscaling plan.

## Exercises

1. Run `code/main.py`. Under a bursty workload, how many requests does naive duty-cycle HPA drop that queue-depth HPA catches?
2. Design a Karpenter NodePool for a cluster serving Llama 3.3 70B FP8 on H100 SXM5. Specify `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, and a taint that keeps non-GPU workloads off these nodes.
3. Your team reports deployments stuck in Pending. Diagnose — is this Karpenter, kube-scheduler, or KAI Scheduler?
4. Pick a signal to autoscale disaggregated prefill pods and a different signal for decode pods.
5. Compute the cost of the `WhenEmptyOrUnderutilized` consolidation trap on a 24x7 production service.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Karpenter | "the node provisioner" | Kubernetes node autoscaler; sub-minute provisioning |
| KAI Scheduler | "the GPU scheduler" | Secondary scheduler for gang + topology + queues |
| Gang scheduling | "all or nothing" | Schedule N pods atomically or defer all of them |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | Duty-cycle metric; NOT a scaling signal for LLMs |
| Queue depth | "waiting requests" | Correct HPA signal for prefill-bound scaling |
| KV cache utilization | "memory pressure" | Correct HPA signal for decode-bound scaling |
| `WhenEmpty + 1h` | "safe consolidation" | Policy that doesn't evict running GPU jobs |

## Further Reading

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [llm-d GitHub](https://github.com/llm-d/llm-d)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes)

---

## Part 2 (ch362): vLLM Serving Internals: PagedAttention, Continuous Batching, Chunked Prefill

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

---

## Part 3 (ch375): Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d

> Prefill is compute-bound; decode is memory-bound. Running both on the same GPU wastes one resource. Disaggregation splits them onto separate pools and transfers KV cache between them over NIXL (RDMA/InfiniBand or TCP fallback). NVIDIA Dynamo (GTC 2025 announce, 1.0 GA) sits above vLLM/SGLang/TRT-LLM — its Planner Profiler + SLA Planner auto-rate-match prefill:decode ratios to meet SLOs. NVIDIA publishes throughput gains in this ballpark — developer.nvidia.com (2025-06) shows a ~6x improvement for DeepSeek-R1 MoE on GB200 NVL72 + Dynamo in the medium-latency regime, and the Dynamo product page (developer.nvidia.com, undated) advertises up to 50x MoE throughput on GB300 NVL72 + Dynamo vs Hopper. The "30x" figure is a community aggregate across full-stack Blackwell + Dynamo + DeepSeek-R1 reports; we have not found a single primary source stating exactly 30x, so treat it as a directional claim. llm-d (Red Hat + AWS) is Kubernetes-native: prefill / decode / router as independent Services with per-role HPA. llm-d 0.5 adds hierarchical KV offloading, cache-aware LoRA routing, UCCL networking, scale-to-zero. Economics: internal rollup of multiple customer disclosures suggests 30–40% savings on $2M-class inference spend (i.e., $600-800K/year) when switching from colocated serving to disaggregated with Dynamo at constant SLA; the specific $2M→$600-800K figure is an internal composite, not a single published case study — use it as an order-of-magnitude anchor, not a reference citation. Short prompts (<512 tokens, short output) don't justify the transfer cost.

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## Learning Objectives

- Explain why prefill and decode have different optimal GPU allocations and quantify the waste under colocation.
- Diagram the disaggregated architecture: prefill pool, decode pool, KV transfer via NIXL, router.
- Name the condition when disaggregation does NOT pay off (short prompts, short outputs).
- Distinguish NVIDIA Dynamo (stack-above) from llm-d (Kubernetes-native) and match each to an operational context.

## The Problem

You run Llama 3.3 70B on 8 H100s. Under mixed workload (long prompts + short outputs), GPUs idle during decode because most of the compute was spent on prefill. Under different workload (short prompts + long outputs), the opposite happens. Colocated prefill + decode means you over-provision both.

Budget impact: 20-40% of GPU time is wasted on the wrong resource. You are buying H100 compute to run memory-bound decode, or buying H100 HBM bandwidth to run compute-bound prefill. Both are expensive waste.

Disaggregation splits prefill and decode onto separate pools sized for each's bottleneck. KV cache transfers from prefill pool to decode pool via high-bandwidth interconnect.

## The Concept

### Why the bottlenecks differ

**Prefill** — run the transformer over the full input prompt in one forward. Matrix multiplications dominate; compute-bound. H100 FP8 gives ~2000 TFLOPS of useful throughput. Batch efficiency is good — one forward processes many tokens.

**Decode** — generate one token at a time, reading the full weights each iteration. Memory-bandwidth-bound. HBM3 gives ~3 TB/s. Batch efficiency is good only at high concurrency — the weights read amortizes across the batch.

Colocating them: you buy GPUs optimized for both. H100 is good at both but costs the same either way. At scale, you want prefill pool on H100 / compute-heavy; decode pool on H200 / memory-heavy, or with aggressive quantization.

### The architecture

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL is NVIDIA's inter-node transport. Uses RDMA/InfiniBand when available, TCP fallback otherwise. Transfer latency is real — typically 20-80 ms for KV cache of a 4K-token prompt on 70B FP8. This is why short prompts don't justify disaggregation: the transfer tax exceeds the savings.

### Dynamo vs llm-d

**NVIDIA Dynamo** (GTC 2025 announce, 1.0 GA):
- Sits above vLLM, SGLang, TRT-LLM as an orchestrator.
- Planner Profiler measures workload, SLA Planner auto-configures prefill:decode ratios.
- Rust core, Python extensibility.
- Throughput gains: NVIDIA reports 6x for DeepSeek-R1 MoE on GB200 NVL72 + Dynamo in the medium-latency regime (developer.nvidia.com, 2025-06); community reports of "up to 30x" on full Blackwell + Dynamo + DeepSeek-R1 stacks lack a single primary source and should be treated as directional.
- GB300 NVL72 + Dynamo: up to 50x MoE throughput vs Hopper per the Dynamo product page (developer.nvidia.com, undated).

**llm-d** (Red Hat + AWS, Kubernetes-native):
- Prefill / decode / router as independent Kubernetes Services.
- Per-role HPA with queue depth (prefill) / KV utilization (decode) signals.
- `topologyConstraint packDomain: rack` packs prefill+decode cliques on the same rack for high-bandwidth KV transfer.
- llm-d 0.5 (2026): hierarchical KV offloading, cache-aware LoRA routing, UCCL networking, scale-to-zero.

Use Dynamo if you want a managed stack-above orchestrator. Use llm-d if you want Kubernetes-native primitives and are committed to the CNCF ecosystem.

### Economics

Internal composite (not a single published case study — order-of-magnitude anchor):

- $2M/year inference spend on colocated serving.
- Switched to disaggregated with Dynamo.
- Same request volume, same P99 latency SLA.
- Reported savings: $600K–$800K/year (30–40% reduction).
- No new hardware.

We synthesize this figure from multiple customer disclosures rather than a single citable case study; closest published data point is Baseten's 2x faster TTFT / 61% higher throughput with Dynamo KV routing (baseten.co, 2025-10), and VAST + CoreWeave's projection of 60–130% more tokens/$ at 40–60% KV hit rate (vastdata.com, 2025-12). The savings come from right-sizing each pool; prefill-heavy workloads (RAG with 8K+ prefixes) benefit more than balanced ones.

### When NOT to disaggregate

- Prompts < 512 tokens and outputs < 200 tokens: transfer tax dominates gain.
- Small cluster (< 4 GPUs): not enough pool diversity.
- Team cannot operate two GPU pools with per-role scaling: Dynamo helps but not trivially.
- No RDMA fabric: TCP transfer tax is heavier.

### The router integrates with Phase 17 · 11

Disaggregated routers are KV-cache-aware (Phase 17 · 11). A request lands on the decode pool holding its prefix — if no match, it flows prefill → decode. Hit rate and disaggregation compound — the cache-aware router determines whether a new prefill is even needed.

### MoE on Blackwell is where the real numbers are

GB300 NVL72 + Dynamo shows 50x MoE throughput over Hopper baselines. MoE expert routing is compute-heavy on prefill but memory-heavy on decode (expert caches), so disaggregation is a double win. 2026 frontier model serving is MoE-dominant (DeepSeek-V3, future GPT-5 variants).

### Numbers you should remember

Benchmark numbers drift — NVIDIA and the inference stack post updated results every quarter. Re-check before quoting.

- DeepSeek-R1 on GB200 NVL72 + Dynamo: ~6x throughput vs baseline in the medium-latency regime (developer.nvidia.com, 2025-06); community "up to 30x" claims on full Blackwell + Dynamo stacks are directional aggregates without a single primary source.
- GB300 NVL72 + Dynamo: up to 50x MoE throughput vs Hopper (developer.nvidia.com, undated).
- Savings anchor (internal composite, not a single case study): $600-800K/year off a $2M annual spend at constant SLA.
- Disaggregation threshold: prompts >512 tokens + outputs >200 tokens.
- KV transfer via NIXL: 20-80 ms for 4K-prompt KV on 70B FP8.

## Use It

`code/main.py` simulates colocated vs disaggregated serving. Reports throughput, cost per request, and the prompt-length crossover.

## Ship It

This lesson produces `outputs/skill-disaggregation-decider.md`. Given workload and cluster, decides whether to disaggregate.

## Exercises

1. Run `code/main.py`. At what prompt length does disaggregation beat colocation?
2. Design the prefill pool and decode pool for a RAG service with P99 prefix length 8K, output 300.
3. Dynamo vs llm-d: pick one for a pure-Kubernetes shop with no Python runtime preference.
4. Compute KV transfer cost: 4K prefill on 70B FP8 = ~500 MB KV. At RDMA 100 GB/s, transfer = 5 ms. At TCP 10 GB/s = 50 ms. Which matters for your SLA?
5. MoE expert routing changes KV access patterns. How does disaggregation behave with MoE that activates different experts per token?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | Separate GPU pools for each phase |
| NIXL | "NVIDIA transport" | Dynamo's inter-node KV transfer (RDMA/TCP) |
| NVIDIA Dynamo | "the orchestrator" | Stack-above coordinator for vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | "Dynamo auto-config" | Measures workload, configures pool ratios |
| SLA Planner | "Dynamo policy" | Auto-rate-matches prefill:decode to meet SLOs |
| `packDomain: rack` | "llm-d topology" | Pack prefill+decode on same rack for fast KV |
| UCCL | "unified collective" | llm-d 0.5 networking layer for scale-to-zero |
| MoE expert routing | "expert per token" | DeepSeek-V3 pattern; disaggregation helps |

## Further Reading

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/17-disaggregated-prefill-decode)

---

## Part 4 (ch376): vLLM Production Stack with LMCache KV Offloading

> vLLM's production-stack is the reference Kubernetes deployment — router, engines, and observability wired together. LMCache is the KV-offloading layer that extracts KV cache out of GPU memory and reuses it across queries and engines (CPU DRAM, then disk/Ceph). The vLLM 0.11.0 KV Offloading Connector (January 2026) makes this asynchronous and pluggable via the Connector API (v0.9.0+). Offload latency is not user-facing. LMCache is valuable even without shared prefixes — when a GPU runs out of KV slots, preempted requests can be restored from CPU instead of recomputing prefill. Published benchmarks on 16x H100 (80GB HBM) across 4 a3-highgpu-4g: when KV cache exceeds HBM, both native CPU offload and LMCache substantially improve throughput; at low KV footprint, all configs match baseline with small overhead.

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Diagram the vLLM production-stack layers: router, engines, KV offload, observability.
- Explain the KV Offloading Connector API (v0.9.0+) and how the 0.11.0 asynchronous path hides offload latency.
- Quantify when LMCache CPU-DRAM helps (KV > HBM) vs adds overhead (KV small enough to fit HBM).
- Pick between native vLLM CPU offload and LMCache connector given deployment constraints.

## The Problem

Your vLLM serving shows GPUs at 100% HBM with preemption events whenever concurrency climbs. Requests get evicted, requeued, and you re-prefill the same 2K-token prompt four times in a minute. GPU compute is spent on redundant prefills; goodput is well below raw throughput.

Adding more GPUs costs linearly. Adding more HBM is not possible. But CPU DRAM is cheap — one socket has 512 GB+ at latency orders of magnitude worse than HBM but fine for "temporarily warm" KV cache.

LMCache extracts KV cache to CPU DRAM so preempted requests recover fast, and repeated prefixes across engines share cache without each engine re-prefilling.

## The Concept

### vLLM production-stack

`github.com/vllm-project/production-stack` is the reference Kubernetes deployment:

- **Router** — cache-aware (Phase 17 · 11). Consumes KV events.
- **Engines** — vLLM workers. One per GPU or per TP/PP group.
- **KV cache offload** — LMCache deployment or native connector.
- **Observability** — Prometheus scrape, Grafana dashboards, OTel traces.
- **Control plane** — service discovery, config, rolling updates.

Shipped as Helm chart + operator.

### The KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 introduced a Connector API for pluggable KV cache backends. Your engine offloads blocks to the connector; connector stores them (RAM, disk, object storage, LMCache). Request needs a block, connector loads it back.

vLLM 0.11.0 (January 2026) adds an asynchronous offload path — offload can happen in the background so the engine does not block on it in the common case. End-to-end latency and throughput still depend on workload shape, KV cache hit rate, and system pressure; vLLM's own notes call out that custom-kernel offload can degrade throughput at low hit rates and that async scheduling has known interaction issues with speculative decoding.

### Native CPU offload vs LMCache

**Native vLLM CPU offload**: engine-local. Stores KV blocks in host RAM. Fast to implement, zero network hop. Does not cross engines.

**LMCache connector**: cluster-scale. Stores blocks in a shared LMCache server (CPU DRAM + Ceph/S3 tier). Blocks are accessible to any engine. 16x H100 benchmarks published.

Pick native when a single engine has HBM pressure. Pick LMCache when multiple engines share prefixes (RAG with common system prompts, multi-tenant with shared templates).

### Benchmark behavior

The 16x H100 (80 GB HBM) spread across 4 a3-highgpu-4g test:

- Low KV footprint (short prompts, low concurrency): all configs match baseline, LMCache adds ~3-5% overhead.
- Moderate footprint: LMCache starts to help on prefix reuse across engines.
- KV exceeds HBM: native CPU offload and LMCache both improve throughput substantially; LMCache larger gain because cross-engine sharing.

### When LMCache is decisive

- Multi-tenant serving where system prompts are shared across tenants.
- RAG where document chunks repeat across queries.
- Fine-tuned variants (LoRA) on the same base where base-model KV reuse cuts redundant work.
- Preemption-heavy workloads: restore from CPU cheaper than re-prefill.

### When NOT to enable

- Small HBM pressure — you pay overhead without benefit.
- Short contexts (<1K tokens) — transfer time > re-prefill.
- Single-tenant single-prompt workload — no reuse to capture.

### Integration with disaggregated serving

Phase 17 · 17 disaggregated serving + LMCache compounds: KV transfers from prefill pool to decode pool land in LMCache if not used; subsequent queries pull from LMCache. Phase 17 · 11 cache-aware router can route to the engine whose local OR LMCache-shared cache matches.

### Numbers you should remember

- vLLM 0.9.0: Connector API shipped.
- vLLM 0.11.0 (Jan 2026): asynchronous offload path; end-to-end latency impact depends on workload, KV hit rate, and system pressure (not an absolute guarantee).
- 16x H100 benchmark: LMCache helps when KV footprint exceeds HBM.
- Small HBM pressure: 3-5% overhead without benefit.

## Use It

`code/main.py` simulates a preemption-heavy workload with and without LMCache. Reports re-prefills avoided, throughput gain, and the break-even HBM utilization.

## Ship It

This lesson produces `outputs/skill-vllm-stack-decider.md`. Given workload shape and vLLM deployment, decides native vs LMCache vs neither.

## Exercises

1. Run `code/main.py`. At what HBM utilization does LMCache start paying?
2. A tenant shares a 6K-token system prompt across 200 queries/hour. Compute expected LMCache savings per tenant.
3. The LMCache server is a single point of failure. Design the HA strategy (replicas, fallback to native).
4. LMCache stores to Ceph on spinning disk. For a 4K-token KV at 70B FP8 (500 MB), what's the read time vs re-prefill?
5. Argue whether the vLLM 0.11.0 asynchronous path is "free" — where does the overhead hide?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Production-stack | "the reference deployment" | vLLM's Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | "engine-local spill" | Store KV in host RAM of same engine |
| LMCache | "cluster KV cache" | Cross-engine KV cache server on CPU DRAM + disk |
| 0.11.0 async | "non-blocking offload" | Offload hidden behind engine stream |
| Preemption | "evict to make room" | KV cache shuffle when HBM full |
| Prefix reuse | "same system prompt" | Multiple queries share beginning; cache hit |
| Ceph tier | "disk tier" | Durable storage below DRAM in the cache hierarchy |

## Further Reading

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/18-vllm-production-stack-lmcache)
