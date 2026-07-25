# GPU Autoscaling on Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

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
