# Theory of Mind and Emergent Coordination

> Li et al. showed that LLM agents in a cooperative text game exhibit **emergent high-order Theory of Mind** (ToM) — reasoning about what another agent believes about a third agent's beliefs — but fail on long-horizon planning. Riedl measured higher-order synergy across a population and found that **only** the ToM-prompt condition produces identity-linked differentiation and goal-directed complementarity.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## Problem

Multi-agent coordination often looks magical: agents divide labor, anticipate each other, avoid redundancy. Usually this "emergence" is an artifact of prompt engineering. Remove the prompt, remove the coordination.

Riedl's 2025 finding is stricter: under controlled conditions, coordination only emerges when agents are prompted to reason about **other agents' minds** (ToM). Without the ToM prompt, even strong models show coordination patterns that do not survive statistical controls.

## Concept

### What ToM means

- **Zeroth-order:** no model of others. The agent acts on its own observations only.
- **First-order:** the agent has a model of each other agent's beliefs. "Alice believes X."
- **Second-order:** the agent models recursive beliefs. "Alice believes that Bob believes X."

Li et al. 2023 found that first- and second-order ToM emerge in LLM agents in cooperative games but degrade with long horizon and unreliable communication.

### The Sally-Anne test

A 1985 false-belief test: Sally puts a marble in basket A, leaves. Anne moves it to basket B. Where will Sally look when she returns? GPT-4-era LLMs pass this when posed plainly. They fail when the narrative is long or the question is phrased indirectly.

### Riedl's coordination measurement

Riedl built a population-scale test: N agents, a cooperative objective, variable prompt conditions. Measure:
1. **Identity-linked differentiation.** Do agents develop stable role distinctions over time?
2. **Goal-directed complementarity.** Do agents' actions complement each other?
3. **Higher-order synergy.** Does the group achieve what no subset could?

Result: only under the ToM prompt condition do all three metrics produce signal above baseline.

### The coordination illusion

Without statistical controls, "emergent coordination" often reflects prompt engineering, observer bias, or post-hoc selection of successful runs.

### A minimal ToM-aware agent

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]
```

The `other_models` attribute is the ToM state.

### Why long-horizon hurts

Context limits cause agents to forget which belief belongs to whom. Hallucination adds false beliefs. Mitigations: explicit ToM state in the prompt, shorter reasoning chains, external ToM store.

### Where ToM fails in production

- Adversarial settings (agents with good ToM are easier to manipulate).
- Heterogeneous teams (different models, ToM does not generalize).
- Ground-truth-dependent tasks (ToM can be a distraction).

## Build It

`code/main.py` implements:
- `ToMAgent` — tracks own beliefs and per-other-agent belief models.
- A cooperative task: three agents must collect three tokens from three boxes. Agents cannot communicate; they infer intent from each other's actions.
- Two configurations: `zeroth_order` (no ToM) and `first_order` (ToM with one-level belief model).
- Measurement over 200 randomized trials: completion rate, duplication rate, average turns.

```
python3 code/main.py
```

Expected output: zeroth-order agents duplicate effort at ~35% rate and complete ~60% of trials. First-order ToM agents duplicate at ~5% and complete ~95%.

## Ship It

Coordination claims checklist:
- **Control condition.** A version without the coordination prompt.
- **Statistical test.** Is the difference significant at p < 0.05?
- **Complementarity measure.** Action-disjointness over time.
- **Failure-case log.** What does the ToM state look like when agents miscoordinate?
- **Model-capacity disclosure.** Does the effect vanish on smaller models?

## Exercises

1. Run `code/main.py`. Confirm first-order ToM reduces duplication rate by ~7x.
2. Implement second-order ToM (agent A models what B thinks about C).
3. Inject a hallucination into the ToM state: randomly flip one belief per turn.
4. Read Li et al. Reproduce the "long-horizon degradation" finding.
5. Read Riedl 2025. Implement the higher-order synergy statistic on your simulation logs.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | "Understanding others' minds" | The capacity to model another agent's beliefs. |
| Sally-Anne test | "The false-belief test" | 1985 developmental psychology test. |
| First-order ToM | "A believes X" | Modeling one other's beliefs about facts. |
| Second-order ToM | "A believes B believes X" | Recursive modeling one level deeper. |
| Identity-linked differentiation | "Stable roles over time" | Riedl's metric: roles persist, not random. |
| Goal-directed complementarity | "Disjoint actions" | Agents target different subtasks. |
| Higher-order synergy | "Group exceeds any subset" | Riedl's statistical measure for real coordination. |
| Coordination illusion | "It looks coordinated" | Prompt-dressed appearance without measurable signal. |

## Further Reading

- [Li et al. — Theory of Mind for Multi-Agent Collaboration](https://arxiv.org/abs/2310.10701)
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174)
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595)
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination)
