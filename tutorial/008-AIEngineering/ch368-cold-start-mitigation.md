# Cold Start Mitigation for Serverless LLMs

> A 20 GB model image takes 5-10 minutes (7B) to 20+ minutes (70B) to go from cold to serving. In a true serverless world, that is not a warm-up — it is an outage. Mitigations operate at five layers: pre-seeded node images, model streaming, GPU memory snapshots, warm pools, tiered loading, and live migration. Modal publishes 2-4s cold starts as a floor; Baseten 5-10s default, sub-second with pre-warming.

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## Learning Objectives

- Enumerate the five layers of cold-start mitigation and name one tool or pattern at each layer.
- Compute total cold-start time as a sum of (node provision) + (weights download) + (weights load into HBM) + (engine init) for a 70B model.
- Explain why live migration transfers input tokens (KB) not KV cache (GB) and what the penalty is.
- Name the warm-pool trade-off (pay for idle GPU or accept cold-start tail) and the SLA threshold at which `min_workers > 0` becomes mandatory.

## The Problem

Your serverless LLM endpoint scales to zero overnight. At 8 a.m. traffic spikes. The first request waits while: Karpenter provisions a GPU node (45-60s), container pulls 30 GB image (120-300s), engine loads weights into HBM (45-120s), vLLM initializes CUDA graphs (10-30s). Total: 220-510s before one token comes back.

## The Concept

### Layer 1 — pre-seeded node images (Bottlerocket)

On AWS, Bottlerocket's dual-volume architecture separates OS from data. Snapshot the data volume with your container image pre-pulled. New nodes boot with weights already on local NVMe.

### Layer 2 — model streaming (Run:ai Model Streamer)

Stream weights into GPU memory layer-by-layer and start processing as soon as the first transformer block is resident. Cuts weight-load time roughly in half.

### Layer 3 — GPU memory snapshots (Modal)

Take a checkpoint of GPU state after first load. Subsequent restarts deserialize directly into HBM — 10x faster.

### Layer 4 — warm pools (min_workers=1)

Keep one replica always ready. Cost is one GPU's hourly rate 24x7.

### Layer 5 — tiered loading (ServerlessLLM)

Storage hierarchy: NVMe → DRAM → HBM. 10-200x latency reduction versus naive disk-to-HBM.

### Live migration

Move input tokens (KB) to a destination that has the model loaded and recompute KV cache. Recomputation is cheaper than transferring GB of KV cache.

### Cold start anatomy for 70B

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total mitigated** | **~15s** | 22x reduction |

## Use It

`code/main.py` models cold-start with and without each mitigation.

```python
"""Cold-start mitigation path simulator — stdlib Python."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Phase:
    name: str
    raw_sec: float
    pre_seeded_sec: float
    streamer_sec: float
    snapshot_sec: float

PHASES_70B = [
    Phase("node provision",   50.0, 50.0,  50.0,  0.5),
    Phase("image pull",      180.0,  0.0, 180.0,  0.0),
    Phase("weights to HBM",   75.0, 75.0,  35.0,  0.0),
    Phase("engine init",      20.0, 20.0,  20.0,  2.0),
    Phase("first forward",     3.0,  3.0,   3.0,  0.5),
]

def total_for_stack(stack: set[str]) -> float:
    seconds = 0.0
    for phase in PHASES_70B:
        if "gpu_snapshot" in stack:
            seconds += phase.snapshot_sec
        elif "streamer" in stack and "pre_seeded" in stack:
            used = phase.pre_seeded_sec
            if phase.name == "weights to HBM":
                used = phase.streamer_sec
            seconds += used
        elif "pre_seeded" in stack:
            seconds += phase.pre_seeded_sec
        elif "streamer" in stack:
            seconds += phase.streamer_sec if phase.name == "weights to HBM" else phase.raw_sec
        else:
            seconds += phase.raw_sec
    return seconds

def report_stack(label: str, stack: set[str]) -> None:
    total = total_for_stack(stack)
    print(f"{label:20}  {total:6.1f} s  ({total/60:4.1f} min)  stack={sorted(stack) if stack else '{baseline}'}")

def main() -> None:
    print("=" * 80)
    print("COLD START MITIGATION — 70B model on fresh H100 node")
    print("=" * 80)
    report_stack("RAW", set())
    report_stack("+ PRE_SEEDED", {"pre_seeded"})
    report_stack("+ STREAMER", {"streamer"})
    report_stack("+ PRE_SEEDED + STREAMER", {"pre_seeded", "streamer"})
    report_stack("+ GPU_SNAPSHOT", {"gpu_snapshot"})

if __name__ == "__main__":
    main()
```

## Ship It

This lesson produces `outputs/skill-cold-start-planner.md`. Given SLA, model size, and traffic shape, picks which mitigations to stack.

## Exercises

1. Compute the break-even request rate above which a warm replica is cheaper than cold-start tax.
2. Deploy a 13B model with P99 TTFT SLA of 3s. Pick the minimum mitigation stack.
3. Bottlerocket pre-seeding eliminates image pull. Compute wall-clock for 70B if NVMe reads at 7 GB/s.
4. Argue both sides of snapshot PII risk and the mitigation (ephemeral snapshots, encryption, isolation).
5. Design a tiered warm-pool policy: paid users, trial users, batch workloads.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cold start | "the big pause" | Time from request to first token on a fresh replica |
| Warm pool | "always-on minimum" | `min_workers >= 1` to keep at least one replica ready |
| Pre-seeded image | "baked AMI" | Node image with container weights pre-resident |
| Model streamer | "streaming load" | Overlap weights I/O with compute setup |
| GPU snapshot | "checkpoint to HBM" | Serialize post-load GPU state; deserialize on restart |
| Tiered loading | "NVMe + DRAM + HBM" | Hierarchy of storage tiers; load on demand |
| Live migration | "move tokens" | Transfer input (KB), recompute KV on destination |

## Further Reading

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start)
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket)
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/10-cold-start-mitigation)
