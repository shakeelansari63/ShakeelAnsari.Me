# Society of Mind and Multi-Agent Debate

> Minsky's 1986 premise — intelligence is a society of specialists — gets rediscovered every decade. In 2023 Du et al. turned it into a concrete algorithm: multiple LLM instances propose answers, read each other's answers, critique, and update. Over N rounds they converge on a consensus that beats zero-shot CoT and reflection.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Self-consistency — sample one model many times and take the majority answer — is the cheapest reasoning improvement you can bolt on. It works, but it saturates fast. You can double your samples and not see another meaningful jump.

Debate breaks the saturation. Instead of N independent samples from one model, N agents read each other's reasoning and revise. The correlation between samples drops, and the convergence point is often correct where i.i.d. voting was confidently wrong.

## Concept

### The Du et al. 2023 algorithm

From arXiv:2305.14325 (ICML 2024):

1. Each of N agents produces an initial answer to the question.
2. For round r = 2..R: each agent is shown the other agents' round r-1 answers and asked "considering these, give your updated answer."
3. After R rounds, majority-vote the final answers.

The paper tests on MMLU, GSM8K, biographies, MATH, and factuality benchmarks. Debate consistently beats CoT and Self-Reflection.

### Two independent knobs

- **Agent count alone** (1 round, majority vote of N) beats single-agent on most tasks, but plateaus.
- **Round count alone** (1 agent seeing its own prior reasoning) barely helps.
- **Both together** produces the big jumps.

### Why it works

1. **Exposure to disagreement.** When an agent sees another agent's reasoning chain with a different conclusion, it has to either justify or update.
2. **Correlated error reduction.** In self-consistency, all samples come from the same model, so the errors correlate. Different debated views decorrelate further.

### Heterogeneous debate

A-HMAD and related follow-ups use *different base models* for different agents. Llama + Claude + GPT debating reduces monoculture collapse because the correlated errors of one model family are not shared by the others.

Downside: a weak model participating in a debate can drag the consensus toward its wrong answer.

### NLSOM — the 129-agent extension

Zhuge et al. ("Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066) scaled this idea to 129-member societies. Specialization and self-organization emerge with scale.

### Failure modes

- **Sycophancy cascade.** All agents defer to whichever agent sounds most confident. Prompting for adversarial roles helps.
- **Topic drift.** Debates over many rounds drift from the original question. Re-inject the question every round.
- **Compute blowup.** N agents × R rounds = N·R LLM calls. A 5-agent, 5-round debate is 25 calls at growing context.

## Build It

`code/main.py` runs a 3-agent × 3-round debate on a math question where each agent starts with a different (possibly wrong) answer. Agents are scripted — each "updates" by averaging the neighbors' answers weighted by a scripted confidence.

Key effects:
- A single round of exchange moves agents closer to the correct answer.
- Extra rounds past round 2 show diminishing returns.

```
python3 code/main.py
```

## Ship It

- **Cap rounds at 3.** Du et al. show 3 rounds capture most of the gain.
- **Cap agents at 5.** Beyond 5, context bloat and cost dominate.
- **Heterogeneous by default.** At least two different base models in the pool.
- **Adversarial slot.** One agent prompted to disagree regardless.
- **Log every round.** Debate systems that hide intermediate rounds cannot be debugged.

## Exercises

1. Run `code/main.py`, then set the round count to 5 and watch diminishing returns.
2. Add a fourth agent with an adversarial role: always disagree with the current majority.
3. Plot the agreement score per round. When does it hit 1.0 and is that equivalent to "correct"?
4. Read Du et al. Section 4 ablations. Replicate the "agents-only" vs "rounds-only" vs "both" result.
5. Read "Should we be going MAD?" (arXiv:2311.17371) and list two debate variants beyond round-robin.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Society of Mind | "Minsky's idea" | Intelligence as interacting specialists. |
| Multi-agent debate | "Agents argue" | N agents propose, critique each other, revise over R rounds, majority-vote. |
| Consensus | "They agree" | Not epistemic truth — just fraction-on-majority-answer. |
| Rounds | "Exchange steps" | One round = each agent reads the others and updates once. |
| Heterogeneous debate | "Mix model families" | Using different base models to decorrelate errors. |
| Sycophancy cascade | "Everyone agrees with the loud one" | Agents defer to the most confident agent regardless of correctness. |
| NLSOM | "129-agent society" | Natural-language society of mind; Zhuge et al.'s scaled version. |
| Correlated error | "Same model, same bug" | Why self-consistency saturates; debate across different views decorrelates. |

## Further Reading

- [Du et al. — Improving Factuality and Reasoning through Multiagent Debate](https://arxiv.org/abs/2305.14325)
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066)
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371)
- [Debate project page](https://composable-models.github.io/llm_debate/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/07-society-of-mind-debate)
