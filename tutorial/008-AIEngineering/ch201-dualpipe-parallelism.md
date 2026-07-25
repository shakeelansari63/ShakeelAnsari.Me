# DualPipe Parallelism

> DeepSeek-V3 was trained on 2,048 H800 GPUs. Cross-node expert all-to-all cost 1 GPU-hour of comm for every 1 GPU-hour of compute. DualPipe is a bidirectional pipeline that overlaps forward and backward computation with all-to-all comms.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training), Phase 10 · 14 (MoE architectures)
**Time:** ~60 minutes

## Learning Objectives

- Name the four components of a DualPipe forward-backward chunk and why each gets its own overlap window
- Explain the pipeline bubble problem and what "bubble-free" means
- Trace a DualPipe schedule by hand for 8 PP ranks and 16 micro-batches
- State the DualPipeV tradeoff: drops 2x parameter cost for slightly larger bubble

## The Problem

Training a 671B MoE model on 2k H800 GPUs hits three bottlenecks:

1. **Memory pressure.** Each GPU holds a slice of the model. Activation memory is enormous.
2. **Pipeline bubbles.** Traditional pipeline parallelism leaves GPUs idle waiting for input or gradients. At 8 stages, ~12% GPU time is bubble.
3. **Cross-node all-to-all.** MoE with expert parallelism scatters experts across nodes. Every forward pass triggers all-to-all dispatch and combine.

DualPipe overlaps compute and comm within a single forward-backward chunk, injects micro-batches from both ends simultaneously, and hides all-to-all inside compute windows.

## The Concept

### Pipeline Parallelism Refresher

Split an N-layer model across P devices. Device `i` holds layers `i*N/P` to `(i+1)*N/P - 1`. Micro-batch flows forward through 0 to P-1, then backward from P-1 to 0.

GPipe wastes most GPU time. 1F1B interleaves forward and backward passes. Zero Bubble splits backward into B (input gradient) and W (weight gradient). After Zero Bubble, the pipeline is almost tight.

DualPipe adds two ideas on top.

### Idea 1: Chunk Decomposition

Each forward chunk splits into four components: Attention, All-to-all dispatch, MLP (MoE), All-to-all combine. Backward chunk adds gradient versions. DualPipe schedules so all-to-all dispatch overlaps with attention compute of the next chunk, and all-to-all combine overlaps with MLP compute of the following chunk.

### Idea 2: Bidirectional Scheduling

Most schedules inject micro-batches from stage 0 flowing toward P-1. DualPipe injects from BOTH ends. Stage 0 sees forward micro-batches originating there; stage P-1 sees forward micro-batches originating there too. The two streams meet in the middle.

Each device keeps two model copies (one per direction). At DeepSeek-V3 scale, Expert Parallelism already spreads experts so thin that replicating non-expert layers twice is small potatoes.

### Bubble Accounting

Standard 1F1B bubble: `(P - 1) * forward_chunk_time`. DualPipe in the stable phase: zero bubble if micro-batch count is divisible by 2 * pipeline depth. Bubbles do not grow with micro-batch count.

"Bubble-free" in marketing terms. In technical terms: bubbles do not grow with micro-batch count.

### DualPipeV (Sea AI Lab, 2025)

Folds bidirectional injection into a V-shape schedule on a single parameter copy. Slightly larger bubble, substantial memory savings.

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies | 2 | 1 | 1 | 1 |
| Bubble vs batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |

## Build It

The code is a pipeline schedule simulator. It takes `(P, n_micro_batches, schedule)` and prints stable-phase utilization for 1F1B, Zero Bubble, DualPipe, and DualPipeV. Run with different P and batch counts and watch bubble fraction grow for 1F1B but not DualPipe.

## Use It

Integration considerations:

- Pick PP depth that divides cleanly into micro-batch count
- Ensure EP mesh supports bidirectional all-to-all
- Expect a week of debugging the schedule
- Monitor GPU utilization per rank, not aggregate

For smaller runs (under 1k GPUs), DualPipe is overkill. For frontier MoE training at multi-thousand GPU scale, it is effectively required.

## Ship It

This lesson produces `outputs/skill-dualpipe-planner.md` -- recommends pipeline parallelism strategy for a given cluster.

## Exercises

1. Run simulator on (P=8, batches=16) for DualPipe and 1F1B. Compute GPU utilization difference.
2. Sketch schedule table for (P=4, batches=8, dualpipe) by hand. Mark first bubble-free slot.
3. Read DeepSeek-V3 Figure 5. Explain how compute schedule hides all-to-all.
4. Compute 2x parameter overhead for 70B dense with P=8 vs 671B MoE with P=16.
5. Compare DualPipe to Chimera (2021). Identify two properties DualPipe added.

## Further Reading

- DeepSeek-AI, "DeepSeek-V3 Technical Report" (arXiv:2412.19437)
- Qi et al., "Zero Bubble Pipeline Parallelism" (arXiv:2401.10241)
- Narayanan et al., "PipeDream / 1F1B" (arXiv:1806.03377)
- Huang et al., "GPipe" (arXiv:1811.06965)
- DeepSeek DualPipe GitHub repository
