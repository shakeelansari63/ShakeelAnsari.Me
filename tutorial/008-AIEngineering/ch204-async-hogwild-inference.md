# Async and Hogwild! Inference

> Run N instances of the same LLM in parallel against a SHARED key-value cache. Each worker sees every other worker's generated tokens instantly. Modern reasoning models self-coordinate through that shared cache without any fine-tuning.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Learning Objectives

- Describe three common parallel-LLM topologies and which problems each targets
- State the core Hogwild! setup: multiple workers, one shared KV cache, emergent coordination via self-prompting
- Compute the wall-time speedup as a function of worker count, parallelizable fraction, and coordination overhead
- Implement a two-worker Hogwild! simulator and observe emergent task division

## The Problem

Modern LLMs solve hard problems by producing long chains of reasoning -- 5000 tokens is common, 50k happens on deep math. At 35 tokens/sec on 70B, 50k tokens is 24 minutes. Speculative decoding gets 3-5x but hits the sequential ceiling: each new token depends on every prior token.

Run multiple copies of the same model on the same problem and let them cooperate. Prior work: voting ensembles, tree-of-thought, multi-agent frameworks. All introduce explicit coordination machinery.

Hogwild! Inference: N workers share a single KV cache. Each worker sees every other worker's tokens immediately. Workers figure out how to divide the work -- no training or fine-tuning needed.

## The Concept

### The Setup

Initialize N worker processes, same LLM, ONE shared KV cache. When worker i generates token t_j, it is written into the shared cache. When worker k takes its next step, it reads the current cache (including everything all workers generated).

Workers race to write tokens. Order determined by write arrival time. No per-worker position index.

### Why Coordination Emerges

Shared prompt: "You are one of N instances working together. Each instance reads shared memory and sees what others wrote. Avoid redundant work."

The prompt plus shared cache is enough. Reasoning models read the cache, notice which parts are already attempted, and pivot to unexplored parts. No fine-tuning required.

Reported emergent behaviors: workers formulate plans and communicate via cache, notice errors in other workers and call them out, adapt when plans fail, detect redundancy and pivot.

### The Naming

Riffs on Hogwild! SGD (Recht et al., 2011) -- asynchronous workers write to a shared parameter vector. Hogwild! Inference workers write to a shared KV cache. Both rely on empirical convergence rather than synchronization guarantees.

### RoPE Makes This Tractable

RoPE encodes position via rotation in Q and K vectors. When worker i writes at position p, other workers reading that position can use the cached entry directly -- no re-rotation needed. If positions were absolute, Hogwild! would need cache invalidation on every concurrent write.

### Wall-Time Math

Let `T_serial` = time for one worker alone. `p` = parallelizable fraction. `c` = per-step coordination overhead.

Single-worker: `T_serial`.
N-worker: `T_serial * ((1-p) + p/N) + c * steps_per_worker`.

For a 10k-token reasoning problem with p=0.7, c=200 tokens, N=4:
- Serial: 10000 steps
- Hogwild!: 10000 * (0.3 + 0.7/4) + 200*4 = 5550 steps
- Speedup: 1.8x

### When to Reach for Hogwild!

- Long reasoning problems (thousands of tokens) with parallelizable sub-goals
- Reasoning models trained to think step by step
- Single-node with enough VRAM for shared cache + N workers

### When Not

- Short interactive chat (coordination overhead dominates)
- Non-parallelizable tasks (single linear proof)
- Non-reasoning models (no emergent coordination)
- Multi-node (shared cache needs fast sync)

### Experimental Status (April 2026)

Research method with open-source PyTorch implementation. Not production-ready. Blockers: shared KV cache management across processes is non-trivial, emergent coordination is task-dependent, speedups modest vs speculative decoding.

## Build It

See `code/main.py` for the full simulation.

### Step 1: Shared Cache

A list that both workers append to. Simple locking with `threading.Lock` in real implementation; simulated with a counter.

### Step 2: Worker Loop

Each worker reads current shared cache, decides what category of token to write based on what is there, writes one token.

### Step 3: Coordination Heuristic

If category X already has K tokens in cache and worker intended X, switch to Y. Toy stand-in for "notice this is already covered, do something else."

### Step 4: Measured Speedup

Run N=1 and N=2, same step budget. N=2 should produce roughly 1.5-1.8x more work-tokens.

### Step 5: Stress the Coordination

Reduce heuristic sensitivity. N=2 redundantly produces same tokens and speedup drops below 1. Matches the paper: trick only works if workers have reasoning capacity to self-coordinate.

## Use It

Pragmatic adoption path:
1. Profile reasoning workload. Measure exploratory vs linear tokens.
2. If exploration dominates, run two-worker Hogwild! experiment.
3. If improvement < 1.3x, revert to single.
4. If > 1.5x, push to N=4. Diminishing returns at N=4-8.

Combine with speculative decoding: each Hogwild! worker uses spec decode independently. Speedups multiply (roughly).

## Ship It

This lesson produces `outputs/skill-parallel-inference-router.md` -- routes between voting, tree-of-thought, multi-agent, Hogwild!, and speculative decoding strategies based on workload profile.

## Exercises

1. Run the simulator with default settings. Confirm N=2 produces more work-tokens than N=1.
2. Reduce coordination heuristic strength (`coordination_weight=0.1`). Show speedup collapses.
3. Compute expected Hogwild! speedup for 50k-token task (p=0.8, c=500, N=4) vs 1k chat (p=0.3, c=200, N=4).
4. Read Hogwild! paper Section 4. Identify two failure modes and how better prompts might mitigate.
5. Combine Hogwild! with speculative decoding: each worker uses 2-token spec-decode. Report multiplicative speedup.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Hogwild! | N LLM instances, one shared KV cache, emergent coordination via self-prompting |
| Shared KV cache | Single growing KV buffer all workers read and write |
| Emergent coordination | Reasoning models divide work without fine-tuning or explicit protocol |
| Coordination overhead (c) | Per-worker cost of reading extended cache and deciding what to do |
| Parallelizable fraction (p) | Fraction of work not intrinsically sequential |
| RoPE enables Hogwild! | Rotary positions are shift-invariant, no recompute needed on shared writes |
| Voting ensemble | Run N models, pick majority answer |
| Tree of thought | Branch reasoning paths, prune and recombine |
| Multi-agent framework | Each agent has a role, coordinator orchestrates |

## Further Reading

- Rodionov et al., "Hogwild! Inference: Parallel LLM Generation via Concurrent Attention" (arXiv:2504.06261)
- Recht et al., "Hogwild!: A Lock-Free Approach to Parallelizing SGD" (NeurIPS 2011)
- Su et al., "RoFormer: Enhanced Transformer with RoPE" (arXiv:2104.09864)
- Yao et al., "Tree of Thoughts" (arXiv:2305.10601)
- Leviathan et al., "Fast Inference via Speculative Decoding" (arXiv:2211.17192)
- Hogwild! reference PyTorch implementation: github.com/eqimp/hogwild_llm
