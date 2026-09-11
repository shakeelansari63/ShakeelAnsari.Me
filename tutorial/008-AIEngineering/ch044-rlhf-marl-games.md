# Reward Modeling, RLHF, MARL & Game RL

> Combined lessons (4 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch179): Reward Modeling & RLHF

> Humans cannot write a reward function for "good assistant response," but they can compare two responses and pick the better one. Fit a reward model to those comparisons, then RL the language model against it. Christiano 2017. InstructGPT 2022. The recipe that turned GPT-3 into ChatGPT. In 2026 it is mostly being replaced by DPO — but the mental model stays.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## The Problem

You trained a language model on the next-token-prediction objective. It writes grammatical English. It also lies, rambles, and refuses to refuse. You cannot fix this with more pretraining — web text is the problem, not the cure.

You want a *scalar reward* that says "response A is better than response B for instruction X." Writing that reward function by hand is impossible. "Helpfulness" is not a closed-form expression over tokens. But humans can compare two outputs and mark a preference. That is cheap to collect at scale.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) converts preferences into a reward model, then optimizes the LM via PPO against that reward. In three steps: SFT → RM → PPO. It is the recipe that shipped ChatGPT, Claude, Gemini, and every other aligned-LLM in 2023–2025.

In 2026 the PPO step is mostly replaced by DPO (Phase 10 · 08) because it is cheaper and nearly as good for alignment tuning. But the *reward model* piece still underlies every Best-of-N sampler, every RL-from-verifiable-rewards pipeline, and every reasoning model using a process reward model. Understand RLHF and you understand the entire alignment stack.

## The Concept

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../images/09-rl/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).** Start from a pretrained base model. Fine-tune on human-written demonstrations of the target behavior (instruction-following responses, helpful replies, etc.). Result: a model `π_SFT` that is *biased toward good behavior* but still has an unbounded action space.

**Stage 2: Reward Model training.**

- Collect pairs of responses `(y_+, y_-)` to prompts `x`, labeled by humans as "y_+ is preferred over y_-."
- Train a reward model `R_φ(x, y)` to assign higher scores to `y_+`.
- Loss: the **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ is the sigmoid. The difference in reward implies a log-odds of preference. BT has been the standard since 1952 (Bradley-Terry) and is the dominant choice in modern RLHF.

- `R_φ` is usually initialized from the SFT model with a scalar head on top. Same transformer backbone; a single linear layer outputs the reward.

**Stage 3: PPO against the RM with KL penalty.**

- Initialize the trainable policy `π_θ` from `π_SFT`. Keep a frozen *reference* `π_ref = π_SFT`.
- Reward at the end of a response `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  The KL penalty prevents `π_θ` from drifting arbitrarily from `π_SFT` — it is a *regularizer*, not a hard trust region. `β` typically `0.01`-`0.05`.
- Run PPO (Lesson 08) with this reward. Advantages are computed on the token-level trajectory, but the RM scores only the full response.

**Why the KL?** Without it, PPO will happily find reward-hacking strategies — the RM was only trained on in-distribution completions. An out-of-distribution response might score higher than any human-written one. The KL keeps `π_θ` near the manifold where the RM was trained. It is the single most important knob in RLHF.

**2026 status:**

- **DPO** (Rafailov 2023): closed-form algebra collapses Stage 2+3 into a single supervised loss over preference data. No RM, no PPO. Same quality on alignment benchmarks for a fraction of the compute. Covered in Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO with a group-relative baseline instead of a critic, reward from a *verifier* (code runs / math answer matches) instead of a human-trained RM. Dominant for reasoning models. Covered in Phase 9 · 12.
- **Process reward models (PRMs):** score partial solutions (each reasoning step), used in both RLHF and GRPO variants for reasoning.
- **Constitutional AI / RLAIF:** use an aligned LLM to generate preferences instead of humans. Scales the preference budget.

## Build It

This lesson uses tiny synthetic "prompts" and "responses" represented as strings. The RM is a linear scorer over a bag-of-tokens representation. No real LLM — the *shape* of the pipeline matters, not the scale.

### Step 1: synthetic preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

In real RLHF this is replaced by human labelers. The shape — `(prompt, preferred_response, rejected_response)` — is identical.

### Step 2: Bradley-Terry reward model

Linear score: `R(x, y) = w · bag(y)`. Train to minimize the BT pairwise log-loss:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

After a few hundred updates, `w` assigns positive weights to good-word tokens and negative to bad.

### Step 3: PPO-like policy on top of RM

Our toy policy produces a single token from a vocabulary. We score the token under the RM, compute `log π_θ(token | prompt)`, add a KL-to-reference penalty, and apply the clipped PPO surrogate.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Step 4: monitor the KL

Track mean `KL(π_θ || π_ref)` every update. If it creeps past `~5-10` the policy has drifted far from `π_SFT` — lower `β` is rising or reward hacking is starting. This is the top diagnostic in real RLHF.

### Step 5: the production recipe with TRL

Once you understand the toy pipeline, here is the same loop as a real library user writes it. Hugging Face's [TRL](https://huggingface.co/docs/trl) is the reference implementation — `RewardTrainer` for Stage 2 and `PPOTrainer` (with a KL-to-reference built in) for Stage 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Three things the library does for you. `adap_kl_ctrl=True` implements the adaptive-β schedule: if observed KL exceeds `target_kl`, β doubles; if below half, β halves. The reference model is frozen by convention — you must not accidentally share parameters with `policy`. And the value head lives on the same backbone as the policy (`AutoModelForCausalLMWithValueHead` attaches a scalar MLP head), which is why TRL reports `policy/kl` and `value/loss` separately.

## Pitfalls

- **Over-optimization / reward hacking.** The RM is imperfect; `π_θ` finds adversarial completions that score high but are bad. Symptoms: reward climbs indefinitely while human eval score plateaus or drops. Fix: stop early, raise `β`, broaden RM training data.
- **Length hacking.** RMs trained on helpful responses often implicitly reward length. The policy learns to pad responses. Remediation: length-normalized reward, or RLAIF with a length-aware RM.
- **Too-small RM.** The RM needs to be at least as large as the policy. A tiny RM cannot faithfully score the policy's outputs.
- **KL tuning.** Too low β → drift and reward hacking. Too high β → policy barely changes. The standard trick is an *adaptive* β that targets a fixed KL per step.
- **Preference-data noise.** ~30% of human labels are noisy or ambiguous. Calibrate by training the RM on agreement-filtered data or use a temperature on BT.
- **Off-policy problems.** PPO data is slightly off-policy after the first epoch. Monitor clip fraction as in Lesson 08.

## Use It

RLHF in 2026 is layered:

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier reward (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with separate safety RM, or Constitutional AI. |
| Best-of-N at inference | Fast alignment | Use RM at decode time; no policy training needed. |
| Reward distillation | Inference compute | Train a small "reward head" on top of a frozen LM. |

RLHF was *the* method in 2022–2024. In 2026, production alignment pipelines are DPO-first, PPO-only for the RM-intensive or safety-critical steps.

## Ship It

Save as `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | Three-stage SFT + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Learned scalar function fit to pairwise preferences via Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; the standard RM objective. |
| KL penalty | "Stay near the reference" | `β · KL(π_θ \|\| π_ref)` in the reward; the anti-reward-hacking regularizer. |
| Reward hacking | "Goodhart's law" | Policy exploits RM flaws; symptoms: reward up, human eval flat. |
| RLAIF | "AI-labeled preferences" | RLHF where labels come from another LM instead of humans. |
| PRM | "Process Reward Model" | Scores partial reasoning steps; used in reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences guided by explicit rules. |

## Further Reading

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — the paper that started RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — the recipe behind ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — earlier RLHF for summarization.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; the post-RLHF default in 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF and self-critique loop.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — the HH paper.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — production `RewardTrainer` and `PPOTrainer`. Read the trainer source for the adaptive-KL and value-head details.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — the canonical walk-through of the three-stage pipeline with diagrams.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — the library; `examples/` has end-to-end RLHF scripts for Llama, Mistral, and Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — the reward-hypothesis view; essential prerequisite for thinking about reward hacking.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/09-reward-modeling-rlhf)

---

## Part 2 (ch180): Multi-Agent RL

> Single-agent RL assumes the environment is stationary. Put two learning agents in the same world and that assumption breaks: each agent is part of the other's environment, and both are changing. Multi-agent RL is the set of tricks to make learning converge when the Markov assumption no longer holds.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## The Problem

A robot learning to navigate a room is a single-agent RL problem. A soccer team is not. AlphaStar vs StarCraft opponents is not. A marketplace of bidding agents is not. Two cars negotiating a four-way stop is not. Many-on-many real-world problems are not.

In every multi-agent setting, from the perspective of any one agent, the other agents *are* part of the environment. As they learn and change their behavior, the environment becomes non-stationary. The Markov property — "next state depends only on current state and my action" — gets violated because the next state also depends on what the *other* agents chose, and their policies are moving targets.

This breaks tabular convergence proofs (Q-learning's guarantee assumes a stationary environment). It breaks naive deep RL too: agents chase each other in loops, never converge to a stable policy. You need multi-agent-specific techniques: centralized training / decentralized execution, counterfactual baselines, league play, self-play.

2026 applications: robot swarms, traffic routing, autonomous vehicle fleets, market simulators, multi-agent LLM systems (Phase 16), and any game with more than one intelligent player.

## The Concept

![Four MARL regimes: indep, centralized critic, self-play, league](../images/09-rl/marl.svg)

**Formalism: Markov Game.** A generalization of MDP: states `S`, a joint action `a = (a_1, …, a_n)`, transition `P(s' | s, a)`, and per-agent rewards `R_i(s, a, s')`. Each agent `i` maximizes its own return under its own policy `π_i`. If rewards are identical, it is **fully cooperative**. If zero-sum, it is **adversarial**. If mixed, it is **general-sum**.

**Core challenges:**

- **Non-stationarity.** `P(s' | s, a_i)` from agent `i`'s view depends on `π_{-i}`, which is changing.
- **Credit assignment.** With a shared reward, which agent caused it?
- **Exploration coordination.** Agents must explore complementary strategies, not redundantly explore the same state.
- **Scalability.** The joint action space grows exponentially in `n`.
- **Partial observability.** Each agent sees only its own observation; the global state is hidden.

**Four dominant regimes:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** Each agent learns its own Q or policy, treating others as part of the environment. Simple, sometimes it works (especially with experience replay acting as a smoothing agent-modeling trick). Theoretical convergence: none. In practice: fine for loosely-coupled tasks, bad for tightly-coupled ones.

**2. Centralized training, decentralized execution (CTDE).** Most common modern paradigm. Each agent has its own *policy* `π_i` that conditions on local observation `o_i` — standard decentralized execution at deployment. During *training*, a centralized critic `Q(s, a_1, …, a_n)` conditions on the full global state and joint action. Examples:
- **MADDPG** (Lowe et al. 2017): DDPG with a centralized critic per agent.
- **COMA** (Foerster et al. 2017): counterfactual baseline — ask "what would my reward have been if I'd taken action `a'` instead?" — isolates my contribution.
- **MAPPO** / **IPPO** with shared critic (Yu et al. 2022): PPO with a centralized value function. Dominant in 2026 for cooperative MARL.
- **QMIX** (Rashid et al. 2018): value decomposition — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` with monotonic mixing.

**3. Self-play.** Two copies of the same agent play each other. The opponent's policy *is* my policy from a past snapshot. AlphaGo / AlphaZero / MuZero. OpenAI Five. Works best for zero-sum games; the training signal is symmetric.

**4. League play.** An extension of self-play to general-sum / adversarial environments: keep a population of past and current policies, sample an opponent from the league, train against them. Adds exploiters (specialize in beating the current best) and main exploiters (specialize in beating exploiters). AlphaStar (StarCraft II). Needed when the game admits "rock-paper-scissors" strategy cycles.

**Communication.** Allow agents to send learned messages `m_i` to each other. Works in cooperative settings. Foerster et al. (2016) showed that differentiable inter-agent communication can be trained end-to-end. Today's LLM-based multi-agent systems (Phase 16) essentially communicate in natural language.

## Build It

This lesson uses a 6×6 GridWorld with two cooperative agents. They start in opposite corners and must reach a shared goal. Shared reward: `-1` per step while either agent is still moving, `+10` when both arrive.

### Step 1: the multi-agent env

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

The *joint* action space is `|A|² = 16`. The global state is two positions.

### Step 2: independent Q-learning

Each agent runs its own Q-table keyed on joint state. At each step: both pick ε-greedy actions, collect joint transition, each updates its own Q with the shared reward.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Works on this task because rewards are dense and aligned. Fails on tightly-coupled tasks (e.g., where one agent has to *wait* for the other).

### Step 3: centralized Q with decomposed-value update

Use one Q over joint actions `Q(s, a_1, a_2)`. Update from shared reward. Decentralize at execution by marginalizing: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Trades exponential joint action space for a *correct* global view.

### Step 4: simple self-play (adversarial 2-agent)

Same agent, two roles. Train agent A against agent B; after `K` episodes, copy A's weights into B. Symmetric training, consistent progress. The AlphaZero recipe in miniature.

## Pitfalls

- **Non-stationary replay.** Experience replay with independent agents is worse than single-agent because old transitions were generated by now-obsolete opponents. Fix: relabel or weight by recency.
- **Credit assignment ambiguity.** Shared reward after a long episode; no clear way to say which agent contributed. Fix: counterfactual baselines (COMA), or reward shaping per agent.
- **Policy drift / chasing.** Each agent's best response changes with each other's update. Fix: centralized critic, slow learning rates, or freeze-one-at-a-time.
- **Reward hacking via coordination.** Agents find coordinated exploits the designer did not anticipate. Auction agents converge to bid zero. Fix: careful reward design, behavioral constraints.
- **Exploration redundancy.** Both agents explore the same state-action pairs. Fix: entropy bonuses per-agent, or role-conditioning.
- **League cycles.** Pure self-play can get stuck in a dominance cycle. Fix: league play with diverse opponents.
- **Sample explosion.** `n` agents × state space × joint actions. Approximate with function approximation; factored action spaces (one policy output head per agent).

## Use It

The 2026 MARL application map:

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| Two-player games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | Zero-sum; symmetric training. |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs; variable team sizes. |
| Auction markets | Game-theoretic equilibrium + RL | Mean-field RL when `n` → ∞. |
| LLM multi-agent systems (Phase 16) | Natural-language comm + role conditioning | RL loop at the agent-planning layer. |

In 2026, MARL's biggest growth area is LLM-based: swarms of language-model agents negotiating, debating, building software. The RL shows up as preference optimization on *trajectory-level* outputs, not token-level (Phase 16 · 03).

## Ship It

Save as `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; each agent has its own reward. |
| CTDE | "Centralized training, decentralized execution" | Joint critic at training time; each agent's policy uses only local obs. |
| IPPO | "Independent PPO" | Each agent runs PPO separately. Simple baseline; often underrated. |
| MAPPO | "Multi-agent PPO" | PPO with a centralized value function conditioned on global state. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` allows decentralized argmax. |
| COMA | "Counterfactual multi-agent" | Advantage = my Q minus expected Q marginalizing over my action. |
| Self-play | "Agent vs past self" | Single agent, two roles; standard for zero-sum games. |
| League play | "Population training" | Cache past policies, sample opponents from the pool; handles strategy cycles. |

## Further Reading

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE with a centralized critic.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — counterfactual baselines for credit assignment.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — value decomposition with monotonicity.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO is surprisingly strong for MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — league play at scale.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — pure self-play in zero-sum games.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — includes the textbook's short treatment of multi-agent settings and the non-stationarity problem that CTDE is designed to solve.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — survey covering cooperative, competitive, and mixed MARL with convergence results.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/10-multi-agent-rl)

---

## Part 3 (ch181): Sim-to-Real Transfer

> A policy trained in a simulator that fails on hardware is a policy that memorized the simulator. Domain randomization, domain adaptation, and system identification are the three tools to make learned controllers cross the reality gap.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## The Problem

Training a real robot is slow, dangerous, and expensive. A biped takes millions of training episodes to learn to walk; a real biped that falls over even once breaks hardware. Simulation gives you unlimited resets, deterministic reproducibility, parallel environments, and no physical damage.

But simulators are wrong. Bearings have more friction than MuJoCo models. Cameras have lens distortion the simulator does not include. Motors have delays, backlash, and saturation that 99% of sim models skip. Wind, dust, and variable lighting sabotage a policy trained on sterile rendering. The **reality gap** — systematic difference between sim distribution and real distribution — is the central problem of deployed RL for robotics.

You need a policy that is *robust to sim-to-real distribution shift*. Three historical approaches: randomize the simulator (domain randomization), adapt the policy with a little real data (domain adaptation / fine-tuning), or identify the real system's parameters and match them (system identification). In 2026 the dominant recipe combines all three with massive parallel simulation (Isaac Sim, Isaac Lab, Mujoco MJX on GPU).

## The Concept

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../images/09-rl/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. During training, randomize every sim parameter that might differ on the real robot: masses, friction coefficients, motor PD gains, sensor noise, camera position, lighting, textures, contact models. The policy learns a conditional distribution over "which sim it is in today" and generalizes across the full span. If the real robot falls within the training envelope, the policy works.

- **Upside:** no real data needed. One recipe, many robots.
- **Downside:** over-randomized training produces a "universal" but overly cautious policy. Too much noise ≈ too much regularization.

**System Identification (SI).** Fit the simulator's parameters to real-world data before training. If you can measure arm-joint friction on the real robot, plug that into the sim. Then train a policy that expects those values. Needs access to the real system but reduces the reality gap directly.

- **Upside:** precise, low-noise training target.
- **Downside:** residual model error is invisible to the policy; small un-identified effects (e.g., motor deadband) still break deployment.

**Domain Adaptation.** Train in sim, fine-tune with a small amount of real data. Two flavors:

- **Real2Sim2Real:** learn a residual simulator `f(s, a, z) - f_sim(s, a)` using real rollouts, train in the corrected sim. Closes the gap without much real data.
- **Observation adaptation:** train a policy that maps real obs → sim-like obs via a learned feature extractor (e.g., GAN pixel-to-pixel). The controller stays in sim.

**Privileged learning / teacher-student.** Miki et al. 2022 (ANYmal quadruped). Train a *teacher* in simulation that has access to privileged information (ground truth friction, terrain height, IMU drift). Distill a *student* that only sees real-sensor observations. The student learns to infer privileged features from history, robust across physical parameters.

**Massively parallel simulation.** 2024–2026. Isaac Lab, Mujoco MJX, Brax all run thousands of parallel robots on a single GPU. PPO with 4,096 parallel humanoids collects years of experience in hours. The "reality gap" shrinks as training distribution widens; DR becomes almost free when each of those 4,096 envs has different randomized parameters.

**The real-world 2026 recipe (quadruped walking example):**

1. Massively parallel sim with domain-randomized gravity, friction, motor gains, payload.
2. Teacher policy trained with privileged info (terrain map, body velocity ground truth).
3. Student policy distilled from teacher using only proprioception (leg joint encoders).
4. Optional observation adaptation via autoencoder on real IMU.
5. Deploy. Zero-shot on 10+ environments. If it fails, do minutes of real-world fine-tuning with safety-constrained PPO.

## Build It

This lesson's code is a tiny demonstration of domain randomization on a GridWorld with *noisy* transitions. We train a policy that experiences randomized slip probabilities in "sim" and evaluate on "real" with a slip level it never saw during training. The shape maps directly to MuJoCo-to-hardware transfer.

### Step 1: parameterized sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` is a parameter the simulator exposes. In real robotics it could be friction, mass, motor gain — anything that shifts between sim and real.

### Step 2: train with DR

At the start of each episode, sample `slip ~ Uniform[0.0, 0.4]`. Train PPO / Q-learning / anything. Do this for many episodes.

### Step 3: evaluate zero-shot on "real" slips

Evaluate on `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. The first four are within training support; `0.5` and `0.7` are outside. A DR-trained policy should stay near-optimal inside support and degrade gracefully outside. A fixed-slip-trained policy will be brittle outside its training slip.

### Step 4: compare to narrow training

Train a second policy with `slip = 0.0` only. Evaluate on the same `slip` sweep. You should see a catastrophic drop as soon as real slip > 0.

## Pitfalls

- **Too much randomization.** Train on `slip ∈ [0, 0.9]` and your policy is so risk-averse it never tries the optimal path. Match the *expected* real-world distribution, not "anything could happen."
- **Too little randomization.** Train on a thin slice and the policy can't generalize at all. Use adaptive curriculum (Automatic Domain Randomization) that widens the distribution as the policy improves.
- **Misidentified parameter space.** Randomize the wrong thing (camera hue when the real gap is motor delay) and DR does not help. Profile the real robot first.
- **Privileged info leakage.** A teacher that uses global state for actions, not just observations, can produce a student that cannot catch up. Ensure the teacher's policy is realizable by the student given observation history.
- **Sim-to-sim transfer failure.** If your policy is not robust to a harder sim variant, it will not be robust to the real world either. Always test on a held-out sim variant before deploying.
- **No real-world safety envelope.** A policy that works in sim and "works in real" without a low-level safety shield can still break hardware. Add rate limits, torque limits, joint limits in a non-learned controller.

## Use It

The 2026 sim-to-real stack:

| Domain | Stack |
|--------|-------|
| Legged locomotion (ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| Manipulation (dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl (DR at unprecedented scale) |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

For control at all scales, the workflow is consistent: fit the sim as best you can, randomize what you can't fit, train enormous policies, distill, deploy with a safety shield.

## Ship It

Save as `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Reality gap | "Sim-to-real difference" | Distribution shift between training and deployment physics/sensing. |
| Domain randomization (DR) | "Train across random sims" | Randomize sim parameters during training so policy generalizes. |
| System identification (SI) | "Measure real and fit sim" | Estimate real physical parameters; set sim to match. |
| Domain adaptation | "Fine-tune on real data" | Small real-world fine-tune after sim training; may adapt obs or dynamics. |
| Privileged info | "Ground truth for teacher" | Information only the sim has; student must infer it from obs history. |
| Teacher/student | "Distill privileged -> observable" | Teacher trained with shortcuts; student learns to mimic without them. |
| ADR | "Automatic Domain Randomization" | Curriculum that widens DR ranges as the policy improves. |
| Real2Sim | "Close the gap with real data" | Learn a residual to make the sim mimic real rollouts. |

## Further Reading

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — the original DR paper (vision for robotics).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR for dynamics, quadruped locomotion.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR at scale.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — teacher-student for ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — the massively parallel sim that drives 2025–2026 deployments.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum method.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — the Dyna framing (use a model for planning + rollouts) that underpins modern sim-to-real pipelines.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — taxonomy of sim-to-real methods with benchmark results.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/11-sim-to-real-transfer)

---

## Part 4 (ch182): RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era

> 1992: TD-Gammon beat human champions at backgammon with pure TD. 2016: AlphaGo beat Lee Sedol. 2017: AlphaZero dominated chess, shogi, and Go from scratch. 2024: DeepSeek-R1 proved the same recipe, with GRPO replacing PPO, works on reasoning. Games are the benchmark that drives every breakthrough in this phase.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## The Problem

Games have everything RL wants. Clean reward (win/loss). Infinite episodes (self-play resets). Perfect simulation (the game *is* the simulator). Discrete or small continuous action spaces. Multi-agent structure that forces adversarial robustness.

And games are how every major RL breakthrough was tested. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (learned model, 2019). AlphaTensor (matrix multiplication, 2022). AlphaDev (sorting algorithms, 2023). DeepSeek-R1 (math reasoning, 2025) — the latest demonstration that game-RL techniques work on text.

This capstone surveys the three landmark architectures — AlphaZero, MuZero, and GRPO — through a single unifying lens: **self-play + search + policy improvement**. Each generalizes the previous; GRPO in particular is AlphaZero's recipe applied to LLM reasoning, with tokens as actions and mathematical verification as the win signal.

## The Concept

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../images/09-rl/rl-games.svg)

**The unifying loop.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver et al. Given a game (chess, shogi, Go) with known rules:

- Policy-value network: one tower `f_θ(s) → (p, v)`. `p` is a prior over legal moves. `v` is the expected game outcome.
- Monte Carlo Tree Search (MCTS): at each move, expand a tree of possible continuations. Use `(p, v)` as the prior + bootstrap. Select nodes by UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Self-play: play games agent-vs-agent. At move `t`, the MCTS visit distribution `π_t` becomes the policy training target.
- Loss: `L = (v - z)² - π · log p + c · ||θ||²`. `z` is the game outcome (+1 / 0 / -1).

Zero human knowledge. Zero handcrafted heuristics. A single recipe that mastered chess, shogi, and Go after a few tens of millions of self-play games each.

**MuZero (2019).** Schrittwieser et al. Removes the requirement that the rules are known.

- Instead of a fixed environment, learn a *latent dynamics model* `(h, g, f)`:
  - `h(s)`: encode observation to latent state.
  - `g(s_latent, a)`: predict next latent state + reward.
  - `f(s_latent)`: predict policy prior + value.
- MCTS runs in the *learned latent space*. Same search, same training loop.
- Works on Go, chess, shogi *and* Atari — one algorithm, no rule knowledge.

**Stochastic MuZero (2022).** Adds stochastic dynamics and chance nodes; extends to backgammon-class games.

**Muesli, Gumbel MuZero (2022-2024).** Improvements on sample efficiency and deterministic search.

**GRPO (2024-2025).** DeepSeek-R1 recipe. Same AlphaZero-shaped loop, applied to language-model reasoning:

- "Game": answer a math / coding / reasoning problem. "Win" = verifier (test case passes, numerical answer matches) returns 1.
- Policy: the LLM. Actions: tokens. State: prompt + response-so-far.
- No critic (PPO-style V_φ). Instead, for each prompt, sample `G` completions from the policy. Compute reward for each. Use the **group-relative advantage** `A_i = (r_i - mean_r) / std_r` as the signal for REINFORCE-style update.
- KL penalty to reference policy to prevent drift (like RLHF).
- Full loss:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

No reward model, no critic, no MCTS. Group-relative baseline replaces all three. Matches or exceeds PPO-RLHF quality on reasoning benchmarks at a fraction of the compute.

**The R1 recipe in full.** DeepSeek-R1 (DeepSeek 2025) is two models in one paper:

- **R1-Zero.** Start from the DeepSeek-V3 base model. No SFT. Apply GRPO directly with two reward components: *accuracy reward* (rule-based — did the final answer parse to the correct number / did the code pass unit tests) and *format reward* (did the completion wrap its chain-of-thought in `<think>…</think>` tags). Over thousands of steps, average response length grows from ~100 to ~10,000 tokens and math benchmark scores climb to near-o1-preview levels. The model learns to reason from scratch. The downside: its chains of thought are often unreadable, mix languages, and lack stylistic polish.
- **R1.** Fix R1-Zero's readability problems with a four-stage pipeline:
  1. **Cold-start SFT.** Collect a few thousand long-CoT demonstrations with clean formatting. Supervised-finetune the base model on them. This gives a readable starting point.
  2. **Reasoning-oriented GRPO.** Apply GRPO with the accuracy+format rewards plus a *language-consistency* reward to prevent code-switching.
  3. **Rejection sampling + SFT round 2.** Sample ~600K reasoning trajectories from the RL checkpoint, keep only those with correct final answers and readable CoT, and combine with ~200K non-reasoning SFT examples (writing, QA, self-cognition). Fine-tune the base again.
  4. **Full-spectrum GRPO.** One more RL round covering both reasoning (rule-based rewards) and general alignment (helpfulness/harmlessness preference-based rewards).

The result matches o1 on AIME and MATH-500 at open weights, and is small enough to distill. The same paper also releases six distilled dense models (Qwen-1.5B through Llama-70B) by SFT'ing on R1's reasoning traces — no RL at the student. Distillation of a strong RL teacher consistently beats RL from scratch at the student's scale.

**Why GRPO instead of PPO for reasoning.** Three reasons in the DeepSeekMath paper (Feb 2024): (1) no value network to train, halving memory; (2) the group baseline naturally handles the sparse end-of-trajectory reward that reasoning tasks produce; (3) per-prompt normalization makes advantages comparable across problems of wildly different difficulty, which PPO's single critic cannot.

**Search-free vs search-based.** Games have branched:

- *Perfect-information games with long horizons* (Go, chess): still search-based. AlphaZero / MuZero dominate.
- *LLM reasoning*: no MCTS yet in production; GRPO on full rollouts, best-of-N for inference compute. Process reward models (PRMs) hint at step-level search being added back.

## Build It

The code in `code/main.py` implements **GRPO in miniature** — a bandit with multiple groups of samples. The algorithm is the same as on an LLM; only the policy and environment are simpler. It teaches the *loss* and the *group-relative advantage*, which is the 2025 innovation.

### Step 1: a tiny verifier environment

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

In real GRPO the verifier runs unit tests or checks math equality.

### Step 2: policy: softmax over K answer tokens per prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Equivalent to the final-layer output of an LLM conditioned on a prompt.

### Step 3: group sampling and group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

The group-relative advantage is the 2024 DeepSeek trick. No critic needed. The "baseline" is the group mean, and normalization uses group std.

### Step 4: compare to REINFORCE baseline (value-free)

Same setup, same compute, plain REINFORCE. GRPO converges faster and more stably.

### Step 5: observe entropy and KL

Same diagnostics as RLHF: mean KL to reference, policy entropy, reward-over-time. Once these stabilize, training is done.

## Pitfalls

- **Reward hacking via verifier gaming.** GRPO inherits RLHF's risk: if the verifier is wrong or exploitable, the LLM will find the exploit. Robust verifiers (multiple test cases, formal proofs) matter.
- **Group size too small.** Variance of the group baseline goes like `1/√G`. Below `G = 4`, the advantage signal is noisy; standard choice is `G = 8` to `64`.
- **Length bias.** LLM completions of different lengths have different log-probabilities. Normalize by token count, or use sequence-level log-prob, or truncate to max length.
- **Pure self-play cycles.** AlphaZero-style training can get stuck in dominance loops on general-sum games. Mitigated by diverse opponent pools (league play, Lesson 10).
- **Search-policy mismatch.** AlphaZero trains the policy to mimic search output. If the policy net is too small to represent the search's distribution, training stalls.
- **Compute floor.** MuZero / AlphaZero need massive compute. A single ablation is often hundreds of GPU-hours. Miniature demos exist (e.g., AlphaZero on Connect Four) for learning.
- **Verifier coverage.** Unit tests that pass for a buggy solution reinforce the bug. Design verifiers that catch edge cases.

## Use It

The 2026 game-RL landscape, by domain:

| Domain | Dominant method |
|--------|-----------------|
| Two-player zero-sum board games (Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| Imperfect info card games (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (not GRPO; verifier is preference not verifiable) |
| Robotics | PPO + DR (not game-RL, but uses same policy-gradient tools) |
| Combinatorial problems | AlphaZero variants (AlphaTensor, AlphaDev) |

The *recipe* — self-play, search-augmented improvement, policy distillation — spans text, pixels, and physical control. GRPO is the youngest instance; more are coming.

## Ship It

Save as `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCTS | "Tree search with learned net" | Monte Carlo Tree Search; UCB1/PUCT selection with learned `(p, v)` priors. |
| AlphaZero | "Self-play + MCTS" | Policy-value net trained to match MCTS visits and game outcome. |
| MuZero | "Learned-model AlphaZero" | Same loop but in latent space via learned dynamics. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; REINFORCE with group-mean baseline + KL. |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — balances value estimate with prior. |
| Self-play | "Agent vs past self" | Standard for zero-sum; symmetric training signal. |
| League play | "Population-based self-play" | Past + current + exploiters sampled as opponents. |
| Verifier reward | "Verifiable RL" | Reward comes from a deterministic checker (tests pass, answer matches). |
| Process reward | "PRM" | Scores each reasoning step, not just the final answer. |

## Further Reading

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — the paper that introduced GRPO and the group-relative baseline.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — the full four-stage R1 recipe plus the R1-Zero ablation.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning at scale.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — the paper that started it all.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — the production reference for applying GRPO with custom reward functions.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — open replication of the R1 recipe at multiple scales.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook framing for self-play, search, and "designed reward" that R1 instantiates at LLM scale.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/12-rl-for-games)
