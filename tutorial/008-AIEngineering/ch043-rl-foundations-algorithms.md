# MDPs, DP, Monte Carlo, TD & Deep RL

> Combined lessons (8 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch171): MDPs, States, Actions & Rewards

> A Markov Decision Process is five things: states, actions, transitions, rewards, a discount. Everything in RL — Q-learning, PPO, DPO, GRPO — optimizes over this shape. Learn it once, read the rest of reinforcement learning for free.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## The Problem

You are writing a chess bot. Or an inventory planner. Or a trading agent. Or the PPO loop that trains a reasoning model. Four different domains, one surprising fact: all four collapse to the same mathematical object.

Supervised learning gives you `(x, y)` pairs and asks you to fit a function. Reinforcement learning gives you no labels — only a stream of states, the actions you took, and a scalar reward. Did the move win the game? Did the restock decision save money? Did the trade make a profit? Did the token the LLM just produced lead to a higher reward from the judge?

You cannot learn from this stream until you formalize it. "What I saw," "what I did," "what happened next," "how good that was" — each has to become an object you can reason about. That formalization is a Markov Decision Process. Every RL algorithm in this phase, including the RLHF and GRPO loops at the end, optimizes over this shape.

## The Concept

![Markov decision process: states, actions, transitions, rewards, discount](../images/09-rl/mdp.svg)

**The five objects.**

- **States** `S`. Everything the agent needs to decide. In GridWorld, the cell. In chess, the board. In an LLM, the context window plus any memory.
- **Actions** `A`. The choices. Move up/down/left/right. Play a move. Emit a token.
- **Transitions** `P(s' | s, a)`. Given state `s` and action `a`, the distribution over next state. Deterministic in chess, stochastic in inventory, almost-deterministic in LLM decoding.
- **Rewards** `R(s, a, s')`. The scalar signal. Win = +1, loss = -1. Revenue minus cost. The log-likelihood ratio term in GRPO.
- **Discount** `γ ∈ [0, 1)`. How much future reward counts vs present. `γ = 0.99` buys a horizon of ~100 steps; `γ = 0.9` buys ~10.

**The Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. The future depends only on the present state. If it does not, the state representation is incomplete — not a failure of the method, a failure of the state.

**Policies and returns.** A policy `π(a | s)` maps states to action distributions. The return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` is the discounted sum of future rewards. The value `V^π(s) = E[G_t | s_t = s]` is the expected return starting from `s` under policy `π`. The Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` is the expected return starting with a specific action. Every RL algorithm estimates one of these two, then improves `π` accordingly.

**The Bellman equations.** The fixed-point equations that everything in this phase uses:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

These split expected return into "this step's reward" plus "discounted value of where you land." Recursive. Every algorithm in Phase 9 either iterates this equation to convergence (dynamic programming), samples from it (Monte Carlo), or bootstraps it one step (temporal difference).

## Build It

### Step 1: a tiny deterministic MDP

A 4×4 GridWorld. Agent starts top-left, terminal at bottom-right, reward of -1 per step, actions `{up, down, left, right}`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Five lines. That is the entire environment. Deterministic transitions, constant step penalty, absorbing terminal state.

### Step 2: roll out a policy

A policy is a function from state to action distribution. The simplest: uniform random.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Run the random policy 1000 times. Average return is around -60 to -80 for this 4×4 board. The optimal return is -6 (straight-line path down-right). Closing that gap is everything in Phase 9.

### Step 3: compute `V^π` exactly via the Bellman equation

For small MDPs the Bellman equation is a linear system. Enumerate states, apply the expectation, iterate until the values stop changing.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

This is iterative policy evaluation. It is the first algorithm in Sutton & Barto and the theoretical foundation of every RL method that follows.

### Step 4: `γ` is a hyperparameter with physical meaning

Effective horizon is roughly `1 / (1 - γ)`. `γ = 0.9` → 10 steps. `γ = 0.99` → 100 steps. `γ = 0.999` → 1000 steps.

Too low and the agent acts myopically. Too high and credit assignment becomes noisy, because many early steps share responsibility for far-future reward. LLM RLHF typically uses `γ = 1` because episodes are short and bounded. Control tasks use `0.95–0.99`. Long-horizon strategy games use `0.999`.

## Pitfalls

- **Non-Markovian state.** If you need the last three observations to decide, the "state" is not just the current observation. Fix: stack frames (DQN on Atari stacks 4) or use a recurrent state (LSTM/GRU over observations).
- **Sparse rewards.** Win-only rewards make learning nearly impossible in large state spaces. Shape rewards (intermediate signal) or bootstrap with imitation (Phase 9 · 09).
- **Reward hacking.** Optimizing a proxy reward often produces pathological behavior. OpenAI's boat-racing agent spun in circles collecting powerups forever instead of finishing the race. Always define reward from the target outcome, not the proxy.
- **Discount mis-spec.** `γ = 1` on an infinite-horizon task makes every value infinite. Always cap with either a finite horizon or `γ < 1`.
- **Reward scale.** Rewards of {+100, -100} vs {+1, -1} give identical optimal policies but vastly different gradient magnitudes. Normalize to `[-1, 1]`-ish before plugging into PPO/DQN.

## Use It

The 2026 stack reduces every RL pipeline to an MDP before touching code:

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

Write the five tuples before writing any training loop. Most "RL does not work" bug reports trace back to an MDP formulation that was broken on paper.

## Ship It

Save as `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MDP | "Reinforcement learning setup" | Tuple `(S, A, P, R, γ)` satisfying the Markov property. |
| State | "What the agent sees" | Sufficient statistic for future dynamics under the chosen policy class. |
| Policy | "Agent's behavior" | Conditional distribution `π(a \| s)` or deterministic map `s → a`. |
| Return | "Total reward" | Discounted sum `Σ γ^t r_t` from the current step. |
| Value | "How good a state is" | Expected return under `π` starting from `s`. |
| Q-value | "How good an action is" | Expected return under `π` starting from `s` with first action `a`. |
| Bellman equation | "Dynamic programming recursion" | Fixed-point decomposition of value / Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on far-future reward; effective horizon `~1/(1-γ)`. |

## Further Reading

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook. Ch. 3 covers MDPs and Bellman equations; Ch. 1 motivates the reward hypothesis that underlies every subsequent lesson.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — the origin of the Bellman equation.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — concise MDP primer from a deep-RL angle.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — the operations-research reference on MDPs and exact solution methods.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — the cleanest derivation of MDPs as a dynamic-programming specialization.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/01-mdps-states-actions-rewards)

---

## Part 2 (ch172): Dynamic Programming — Policy Iteration & Value Iteration

> Dynamic programming is RL with cheating. You already know the transition and reward functions; you just iterate the Bellman equation until `V` or `π` stops moving. It is the benchmark every sampling-based method tries to approach.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## The Problem

You have an MDP with a known model: you can query `P(s' | s, a)` and `R(s, a, s')` for any state-action pair. An inventory manager knows the demand distribution. A board game has deterministic transitions. A gridworld is four lines of Python. You have a *model*.

Model-free RL (Q-learning, PPO, REINFORCE) was invented for the case where you don't have a model — you can only sample from the environment. But when you do have one, there are faster, better methods: dynamic programming. Bellman designed them in 1957. They still define correctness: when people say "optimal policy for this MDP," they mean the policy DP would return.

You need them in 2026 for three reasons. First, every tabular environment in RL research (GridWorld, FrozenLake, CliffWalking) is solved with DP to produce the gold-standard policy. Second, exact values let you *debug* sampling methods: if Q-learning's estimate for `V*(s_0)` disagrees with the DP answer by 30%, your Q-learning has a bug. Third, modern offline RL and planning methods (MCTS, AlphaZero's search, model-based RL in Phase 9 · 10) all iterate a Bellman backup over a learned or given model.

## The Concept

![Policy iteration and value iteration, side by side](../images/09-rl/dp.svg)

**Two algorithms, both fixed-point iteration on Bellman.**

**Policy iteration.** Alternates two steps until the policy stops changing.

1. *Evaluation:* given policy `π`, compute `V^π` by repeatedly applying `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` until it converges.
2. *Improvement:* given `V^π`, make `π` greedy w.r.t. `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Convergence is guaranteed because (a) each improvement step either keeps `π` the same or strictly increases `V^π` for some state, (b) the space of deterministic policies is finite. Usually converges in ~5–20 outer iterations even for large state spaces.

**Value iteration.** Collapses evaluation and improvement into one sweep. Apply the Bellman *optimality* equation:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Repeat until `max_s |V_{new}(s) - V(s)| < ε`. Extract the policy at the end by taking the greedy action. Strictly faster per iteration — no inner evaluation loop — but typically needs more iterations to converge.

**Generalized policy iteration (GPI).** The unifying framing. Value function and policy are locked in a two-way improvement loop; any method that drives both toward mutual consistency (async value iteration, modified policy iteration, Q-learning, actor-critic, PPO) is an instance of GPI.

**Why `γ < 1` matters.** The Bellman operator is a `γ`-contraction in the sup-norm: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Contraction implies unique fixed point and geometric convergence. Drop `γ < 1` and you lose the guarantee — you need a finite horizon or an absorbing terminal state.

## Build It

### Step 1: build the GridWorld MDP model

Use the same 4×4 GridWorld from Lesson 01. We add a stochastic variant: with probability `0.1` the agent slips to a random perpendicular direction.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` returns a list of `(s', r, p)`. This is the entire model.

### Step 2: policy evaluation

Given a policy `π(s) = {action: prob}`, iterate the Bellman equation until `V` stops moving:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: policy improvement

Replace `π` with the greedy policy w.r.t. `V`. If `π` did not change, return — we are at the optimum.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: stitch them together

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Typical convergence on 4×4: 4–6 outer iterations. Outputs `V*(0,0) ≈ -6` and a policy that strictly decreases the step count.

### Step 5: value iteration (the one-loop version)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Same fixed point, fewer lines of code.

## Pitfalls

- **Forgetting to handle terminals.** If you apply Bellman to an absorbing state, it still picks up a "best action" that changes nothing. Guard with `if s == terminal: V[s] = 0`.
- **Sup-norm vs L2 convergence.** Use `max |V_new - V|`, not average. The theoretical guarantee is on the sup-norm.
- **In-place vs synchronous updates.** Updating `V[s]` in-place (Gauss-Seidel) converges faster than a separate `V_new` dict (Jacobi). Production code uses in-place.
- **Policy ties.** If two actions have equal Q-value, `argmax` may break ties differently each iteration, causing the "policy stable" check to oscillate. Use a stable tie-break (first action in fixed order).
- **State-space explosion.** DP is `O(|S| · |A|)` per sweep. Works up to ~10⁷ states. Beyond that, you need function approximation (Phase 9 · 05 onwards).

## Use It

In 2026, DP is the correctness baseline and the inner loop of planners:

| Use case | Method |
|----------|--------|
| Solve a small tabular MDP exactly | Value iteration (simpler) or policy iteration (fewer outer steps) |
| Verify a Q-learning / PPO implementation | Compare to DP-optimal V* on a toy environment |
| Model-based RL (Phase 9 · 10) | Bellman backup on a learned transition model |
| Planning in AlphaZero / MuZero | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP with a penalty on OOD actions |

Every time someone says "the optimal value function," they mean "the DP fixed point." When you see `V*` or `Q*` in a paper, picture this loop.

## Ship It

Save as `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy iteration | "DP algorithm" | Alternating evaluation (`V^π`) and improvement (greedy `π` w.r.t. `V^π`) until the policy stops changing. |
| Value iteration | "Faster DP" | Bellman optimality backup applied in one sweep; converges to `V*` geometrically. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; a `γ`-contraction in sup-norm. |
| Contraction | "Why DP converges" | Any operator `T` with `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` has a unique fixed point. |
| GPI | "Everything is DP" | Generalized Policy Iteration: any method driving `V` and `π` to mutual consistency. |
| Synchronous update | "Jacobi-style" | Use old `V` throughout a sweep; cleanly analyzable but slower. |
| In-place update | "Gauss-Seidel-style" | Use `V` as it's being updated; converges faster in practice. |

## Further Reading

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — the canonical presentation of policy iteration and value iteration.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — rigorous treatment of contraction-mapping arguments.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration and its convergence analysis.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — the original policy iteration paper.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — the bridge from DP to approximate-DP / deep RL used by every subsequent lesson.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/02-dynamic-programming)

---

## Part 3 (ch173): Monte Carlo Methods — Learning from Complete Episodes

> Dynamic programming needs a model. Monte Carlo needs nothing but episodes. Run the policy, watch the returns, average them. The simplest idea in RL — and the one that unlocks everything downstream.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## The Problem

Dynamic programming is elegant, but it assumes you can query `P(s' | s, a)` for every state and action. Almost nothing in the real world works that way. A robot cannot analytically compute the distribution over camera pixels after a joint torque. A pricing algorithm cannot integrate over every possible customer reaction. An LLM cannot enumerate all possible continuations after a token.

You need a method that only needs the ability to *sample* from the environment. Run the policy. Get a trajectory `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Use it to estimate values. That is Monte Carlo.

The shift from DP to MC is philosophically important: we move from *known model + exact backup* to *sampled rollouts + averaged return*. The variance jumps, but the applicability explodes. Every RL algorithm after this lesson — TD, Q-learning, REINFORCE, PPO, GRPO — is a Monte Carlo estimator at heart, sometimes with bootstrapping layered on top.

## The Concept

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../images/09-rl/monte-carlo.svg)

**The core idea, in one line:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` where `G^{(i)}(s)` are observed returns following visits to `s` under policy `π`.

**First-visit vs every-visit MC.** Given an episode that visits state `s` multiple times, first-visit MC only counts the return from the first visit; every-visit MC counts all visits. Both are unbiased in the limit. First-visit is simpler to analyze (iid samples). Every-visit uses more data per episode and typically converges faster in practice.

**Incremental mean.** Instead of storing all returns, update the running average:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Reorganize: `V_new = V_old + α · (target - V_old)` with `α = 1/n`. Swap `1/n` for a constant step-size `α ∈ (0, 1)` and you get a non-stationary MC estimator that tracks changes in `π`. That move is the entire jump from MC to TD to every modern RL algorithm.

**Exploration is now a problem.** DP touched every state by enumeration. MC only sees states the policy visits. If `π` is deterministic, whole regions of the state space never get sampled, and their value estimates stay at zero forever. Three fixes, in historical order:

1. **Exploring starts.** Start each episode from a random (s, a) pair. Guarantees coverage; unrealistic in practice (you cannot "reset" a robot into an arbitrary state).
2. **ε-greedy.** Act greedy w.r.t. current Q, but with probability `ε` pick a random action. All state-action pairs get sampled asymptotically.
3. **Off-policy MC.** Collect data under a behavior policy `μ`, learn about target policy `π` via importance sampling. High variance, but it's the bridge to replay-buffer methods like DQN.

**Monte Carlo Control.** Evaluate → improve → evaluate, just like policy iteration, but evaluation is sampling-based:

1. Run `π`, get an episode.
2. Update `Q(s, a)` from observed returns.
3. Make `π` ε-greedy w.r.t. `Q`.
4. Repeat.

Converges to `Q*` and `π*` with probability 1 under mild conditions (every pair visited infinitely often, `α` satisfies Robbins-Monro).

## Build It

### Step 1: rollout → list of (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

No model, only `env.reset()` and `env.step(s, a)`. Same interface as a gym environment but stripped down.

### Step 2: compute returns (reverse sweep)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

One pass, `O(T)`. The backward recurrence `G_t = r_{t+1} + γ G_{t+1}` avoids re-summing.

### Step 3: first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Three lines do the work: mark state as seen on first visit, increment count, update running mean.

### Step 4: ε-greedy MC control (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5: compare to DP gold standard

Your MC estimate of `V^π` should agree with the DP result from Lesson 02 as episodes → ∞. In practice: 50,000 episodes on 4×4 GridWorld gets you within `~0.1` of the DP answer.

## Pitfalls

- **Infinite episodes.** MC requires episodes to *terminate*. If your policy can loop forever, cap `max_steps` and treat the cap as implicit failure. GridWorld with a random policy routinely times out — that is normal, just make sure you count it correctly.
- **Variance.** MC uses full returns. On long episodes, variance is huge — one unlucky reward at the end shifts `V(s_0)` by the same amount. TD methods (Lesson 04) cut this by bootstrapping.
- **State coverage.** Greedy MC on a fresh Q with ties will only ever try one action. You *must* explore (ε-greedy, exploring starts, UCB).
- **Non-stationary policies.** If `π` changes (as in MC control), old returns are from a different policy. Constant-α MC handles this; sample-average MC does not.
- **Off-policy importance sampling.** The weights `π(a|s)/μ(a|s)` multiply across a trajectory. Variance explodes with horizon. Cap with per-decision weighted IS or switch to TD.

## Use It

The 2026 role of Monte Carlo methods:

| Use case | Why MC |
|----------|--------|
| Short-horizon games (blackjack, poker) | Episodes terminate naturally; returns are clean. |
| Offline evaluation of a logged policy | Average discounted returns over stored trajectories. |
| Monte Carlo Tree Search (AlphaZero) | MC rollouts from tree leaves guide selection. |
| LLM RL evaluation | Compute average reward over sampled completions for a given policy. |
| Baseline estimation in PPO | The advantage target `A_t = G_t - V(s_t)` uses an MC `G_t`. |
| Teaching RL | Simplest algorithm that actually works — strip bootstrapping to see the core. |

Modern deep-RL algorithms (PPO, SAC) interpolate between pure MC (full returns) and pure TD (one-step bootstrap) via `n`-step returns or GAE. Both endpoints are instances of the same estimator.

## Ship It

Save as `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Monte Carlo | "Random sampling" | Estimate expectations by averaging over iid samples from the distribution. |
| Return `G_t` | "Future reward" | Sum of discounted rewards from step `t` to episode end: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Count each state once" | Only the first visit in an episode contributes to the value estimate. |
| Every-visit MC | "Use all visits" | Every visit contributes; slightly biased but more sample-efficient. |
| ε-greedy | "Exploration noise" | Pick greedy action with prob `1-ε`; random action with prob `ε`. |
| Importance sampling | "Correcting for sampling from the wrong distribution" | Reweight returns by `π(a\|s)/μ(a\|s)` products to estimate `V^π` from `μ` data. |
| On-policy | "Learn from my own data" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Learn from someone else's data" | Target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## Further Reading

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — the canonical treatment.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit vs every-visit analysis.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC and variance control.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — modern low-variance IS estimators.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — the first large-scale empirical demonstration of MC/TD self-play converging to superhuman play; conceptual precursor to every lesson in the second half of this phase.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/03-monte-carlo-methods)

---

## Part 4 (ch174): Temporal Difference — Q-Learning & SARSA

> Monte Carlo waits until the episode ends. TD updates after every step by bootstrapping the next value estimate. Q-learning is off-policy and optimistic; SARSA is on-policy and cautious. Both are one line of code. Both underpin every deep-RL method in this phase.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## The Problem

Monte Carlo works but it has two expensive demands. It needs episodes that terminate, and it only updates after the final return is in. If your episode is 1,000 steps, MC waits 1,000 steps to update anything. It is high-variance, low-bias, and slow in practice.

Dynamic programming has the opposite profile — zero-variance bootstrapped backups — but requires a known model.

Temporal difference (TD) learning splits the difference. From a single transition `(s, a, r, s')`, form a one-step target `r + γ V(s')` and nudge `V(s)` toward it. No model. No complete episodes. Bias from using an approximate `V` on the RHS, but dramatically lower variance than MC and online updates from step one.

This is the pivot on which all of modern RL — DQN, A2C, PPO, SAC — turns. The rest of Phase 9 is layers of function approximation and tricks built on top of the one-step TD update you will write in this lesson.

## The Concept

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../images/09-rl/td.svg)

**The TD(0) update for V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

The bracketed quantity is the TD error `δ = r + γ V(s') - V(s)`. It is the online analogue of `G_t - V(s_t)` in MC. Convergence requires `α` satisfying Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) and all states visited infinitely often.

**Q-learning.** An off-policy TD method for control:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

The `max` assumes the *greedy* policy will be followed from `s'` onward, regardless of what action the agent actually takes. That decoupling makes Q-learning learn `Q*` while the agent explores via ε-greedy. Mnih et al. (2015) converted this into deep Q-learning on Atari (Lesson 05).

**SARSA.** An on-policy TD method:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

The name is the tuple `(s, a, r, s', a')`. SARSA uses the action `a'` the agent *actually* takes next, not the greedy `argmax`. Converges to `Q^π` for whatever ε-greedy `π` is running, which in the limit `ε → 0` becomes `Q*`.

**The cliff-walking difference.** On the classic cliff-walking task (fall-off-cliff = reward -100), Q-learning learns the optimal path along the cliff edge but occasionally takes the penalty during exploration. SARSA learns a safer path one step away from the cliff because it factors exploration noise into its Q-value. With training, both reach optimal at `ε → 0`. In practice it matters: when exploration is actually happening at deployment, SARSA's behavior is more conservative.

**Expected SARSA.** Replace `Q(s', a')` with its expected value under `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Lower variance than SARSA (no sample of `a'`), same on-policy target. Often the default in modern textbooks.

**n-step TD and TD(λ).** Interpolate between TD(0) and MC by waiting `n` steps before bootstrapping. `n=1` is TD, `n=∞` is MC. TD(λ) averages over all `n` with geometric weights `(1-λ)λ^{n-1}`. Most deep-RL uses `n` between 3 and 20.

## Build It

### Step 1: SARSA on ε-greedy policy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Eight lines. The *only* difference from Q-learning is the target line.

### Step 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

The `max` decouples target from behavior. That one symbol is the difference between on-policy and off-policy.

### Step 3: learning curves

Track mean return per 100 episodes. Q-learning converges faster on simple deterministic GridWorld; SARSA is more conservative on cliff-walking. On the 4×4 GridWorld in `code/main.py`, both are near-optimal after ~2,000 episodes with `α=0.1, ε=0.1`.

### Step 4: compare to DP truth

Run value iteration (Lesson 02) to get `Q*`. Check `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. A healthy tabular TD agent lands within `~0.5` on the 4×4 GridWorld after 10,000 episodes.

## Pitfalls

- **Initial Q values matter.** Optimistic init (`Q = 0` for a negative-reward task) encourages exploration. Pessimistic init can trap a greedy policy forever.
- **α schedule.** Constant `α` is fine for non-stationary problems. Decaying `α_n = 1/n` gives convergence in theory but is too slow in practice — pin `α` in `[0.05, 0.3]` and monitor the learning curve.
- **ε schedule.** Start high (`ε=1.0`), decay to `ε=0.05`. "GLIE" (greedy in the limit with infinite exploration) is the convergence condition.
- **Max bias in Q-learning.** The `max` operator is biased upward when `Q` is noisy. Leads to overestimation — Hasselt's Double Q-learning (used by DDQN in Lesson 05) fixes this with two Q tables.
- **Non-terminating episodes.** TD can learn without terminals, but you need to either cap steps or handle bootstrap correctly at the cap. Standard: treat cap as non-terminal, keep bootstrapping.
- **State hashing.** If states are tuples/tensors, use a hashable key (tuple, not list; tuple of floats rounded, not raw).

## Use It

The 2026 TD landscape:

| Task | Method | Reason |
|------|--------|--------|
| Small tabular environments | Q-learning | Learns optimal policy directly. |
| On-policy safety-critical | SARSA / Expected SARSA | Conservative during exploration. |
| High-dimensional state | DQN (Phase 9 · 05) | Neural-net Q-function with replay and target net. |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | TD update on a Q-network; policy net emits actions. |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic with TD-style advantage via GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning with conservative regularization. |

Ninety percent of the "RL" you read about in 2026 papers is some elaboration of Q-learning or SARSA. Understand the tabular update in your fingers before reading deeper.

## Ship It

Save as `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`, the bootstrapped residual. |
| TD(0) | "One-step TD" | Update after every transition using only the next state's estimate. |
| Q-learning | "Off-policy RL 101" | TD update with `max` over next-state actions; learns `Q*` regardless of behavior policy. |
| SARSA | "On-policy Q-learning" | TD update using the actual next action; learns `Q^π` for current ε-greedy π. |
| Expected SARSA | "The low-variance SARSA" | Replace sampled `a'` with its expectation under π. |
| GLIE | "Correct exploration schedule" | Greedy in the Limit with Infinite Exploration; needed for Q-learning convergence. |
| Bootstrapping | "Using current estimate in the target" | What distinguishes TD from MC. Source of bias but massive variance reduction. |
| Maximization bias | "Q-learning overestimates" | `max` over noisy estimates is upward-biased; fixed by Double Q-learning. |

## Further Reading

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — the original paper and convergence proof.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — fix for maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA motivation.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — the paper that coined SARSA (then called "modified connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — generalizes TD(0) to TD(n), the path from Q-learning to eligibility traces and, later, GAE in PPO.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/04-q-learning-sarsa)

---

## Part 5 (ch175): Deep Q-Networks (DQN)

> 2013: Mnih trained one Q-learning network on raw pixels, beat every classical RL agent on seven Atari games. 2015: extended to 49 games, published in Nature, sparked the deep-RL era. DQN is Q-learning plus three tricks that make function approximation stable.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## The Problem

Tabular Q-learning needs a separate Q-value for every (state, action) pair. A chess board has ~10⁴³ states. An Atari frame is 210×160×3 = 100,800 features. Tabular RL dies at thousands of states, let alone billions.

The fix is obvious in hindsight: replace the Q-table with a neural network, `Q(s, a; θ)`. But obvious-in-hindsight took decades. Naive function approximation with Q-learning diverges under the "deadly triad" — function approximation + bootstrapping + off-policy learning. Mnih et al. (2013, 2015) identified three engineering tricks that stabilize learning:

1. **Experience replay** decorrelates transitions.
2. **Target network** freezes the bootstrap target.
3. **Reward clipping** normalizes gradient magnitudes.

DQN on Atari was the first time a single architecture with a single hyperparameter set solved dozens of control problems from raw pixels. Everything "deep-RL" built since — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — is stacked on top of this three-trick base.

## The Concept

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../images/09-rl/dqn.svg)

**The objective.** DQN minimizes the one-step TD loss on a neural Q-function:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = online network, updated every step by gradient descent. `θ^-` = target network, periodically copied from `θ` (every ~10,000 steps). `D` = replay buffer of past transitions.

**The three tricks, in order of importance:**

**Experience replay.** A ring buffer of `~10⁶` transitions. Each training step samples a minibatch uniformly at random. This breaks temporal correlation (successive frames are nearly identical), lets the network learn from rare rewarding transitions many times, and decorrelates consecutive gradient updates. Without it, on-policy TD with a neural net diverges on Atari.

**Target network.** Using the same network `Q(·; θ)` on both sides of the Bellman equation makes the target move every update — "chasing your own tail." The fix: keep a second network `Q(·; θ^-)` with frozen weights. Every `C` steps, copy `θ → θ^-`. This stabilizes the regression target for thousands of gradient steps at a time. Soft updates `θ^- ← τ θ + (1-τ) θ^-` (used in DDPG, SAC) are a smoother variant.

**Reward clipping.** Atari reward magnitudes vary from 1 to 1000+. Clipping to `{-1, 0, +1}` stops any single game from dominating the gradient. Wrong when reward magnitude matters; fine for Atari where only sign matters.

**Double DQN.** Hasselt (2016) fixes maximization bias: use the online net to *select* the action, the target net to *evaluate* it.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Drop-in replacement, consistently better. Use it by default.

**Other improvements (Rainbow, 2017):** prioritized replay (sample high-TD-error transitions more), dueling architecture (separate `V(s)` and advantage heads), noisy networks (learned exploration), n-step returns, distributional Q (C51/QR-DQN), multi-step bootstrapping. Each adds a few percent; the gains are roughly additive.

## Build It

The code here is stdlib-only numpy-free — we use a hand-rolled single-hidden-layer MLP on a tiny continuous GridWorld, so every training step runs in microseconds. The algorithm is identical to Atari DQN at scale.

### Step 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~50,000 capacity for Atari; 5,000 suffices for our toy env.

### Step 2: a tiny Q-network (manual MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass: linear → ReLU → linear. That is the entire net.

### Step 3: the DQN update

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

The shape is Q-learning from Lesson 04 with two differences: (a) we backprop through a differentiable `Q(·; θ)` instead of indexing a table, (b) the target uses `Q(·; θ^-)`.

### Step 4: the outer loop

For each episode, act ε-greedy on `Q(·; θ)`, push transitions into the buffer, sample a minibatch, take a gradient step, periodically sync `θ^- ← θ`. The pattern:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

On our tiny GridWorld with a 16-dim one-hot state, the agent learns a near-optimal policy in ~500 episodes. On Atari, scale this to 200M frames and add a CNN feature extractor.

## Pitfalls

- **Deadly triad.** Function approximation + off-policy + bootstrapping can diverge. DQN mitigates with target net + replay; do not remove either.
- **Exploration.** ε must decay, typically from 1.0 to 0.01 over the first ~10% of training. Without enough early exploration the Q-net converges to a local basin.
- **Overestimation.** `max` over noisy Q is upward-biased. Always use Double DQN in production.
- **Reward scale.** Clip or normalize rewards; the gradient magnitude is proportional to reward magnitude.
- **Replay buffer coldstart.** Don't train until the buffer has a few thousand transitions. Early gradients on ~20 samples overfit.
- **Target sync frequency.** Too frequent ≈ no target net; too infrequent ≈ stale targets. Atari DQN uses 10,000 env steps. Rule of thumb: sync every ~1/100 of training horizon.
- **Observation preprocessing.** Atari DQN stacks 4 frames to make state Markov. Any env with velocity info needs frame-stacking or recurrent state.

## Use It

In 2026, DQN is rarely state-of-the-art but remains the reference off-policy algorithm:

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | Same framework, more tricks. |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN has no policy network. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | No replay buffer; easier to scale. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, no bootstrapping blowups. |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | Fine; decoration matters. |
| LLM RL | PPO / GRPO | Sequence-level, not step-level; different loss. |

The lessons still travel. Replay and target networks appear in SAC, TD3, DDPG, SAC-X, AlphaZero's self-play buffer, and every offline RL method. Reward clipping lives on as advantage normalization in PPO. The architecture is the blueprint.

## Ship It

Save as `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning with a neural Q-function, replay buffer, and target network. |
| Experience replay | "Shuffled transitions" | Ring buffer sampled uniformly each gradient step; decorrelates data. |
| Target network | "Frozen bootstrap" | Periodic copy of Q used in the Bellman target; stabilizes training. |
| Deadly triad | "Why RL diverges" | Function approximation + bootstrapping + off-policy = no convergence guarantee. |
| Double DQN | "Fix for maximization bias" | Online net selects action, target net evaluates it. |
| Dueling DQN | "V and A heads" | Decompose Q = V + A - mean(A); same output, better gradient flow. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional in one. |
| PER | "Prioritized Replay" | Sample transitions proportional to TD-error magnitude. |

## Further Reading

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — the 2013 NeurIPS workshop paper that kicked off deep RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — the Nature paper, 49-game DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — the stacked-tricks paper.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — clear modern exposition.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook treatment of the "deadly triad" (function approximation + bootstrapping + off-policy) that DQN's target network and replay buffer are designed to tame.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — reference single-file DQN used in ablation studies; good to read alongside this lesson's from-scratch version.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/05-dqn)

---

## Part 6 (ch176): Policy Gradient — REINFORCE from Scratch

> Stop estimating value. Parameterize the policy directly, compute the gradient of expected return, step uphill. Williams (1992) wrote it in one theorem. It is why PPO, GRPO, and every LLM RL loop exist.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## The Problem

Q-learning and DQN parameterize the *value* function. You pick actions by `argmax Q`. That is fine for discrete actions and discrete states. It breaks when actions are continuous (which `argmax` over a 10-dimensional torque?) or when you want a stochastic policy (`argmax` is deterministic by construction).

Policy gradients parameterize the *policy* instead. `π_θ(a | s)` is a neural net that outputs a distribution over actions. Sample from it to act. Compute the gradient of expected return with respect to `θ`. Step uphill. No `argmax`. No Bellman recursion. Just gradient ascent on `J(θ) = E_{π_θ}[G]`.

The REINFORCE theorem (Williams 1992) tells you this gradient is computable: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Run an episode. Compute the return. Multiply by `∇ log π_θ(a | s)` at every step. Average. Gradient-ascent. Done.

Every LLM-RL algorithm in 2026 — PPO, DPO, GRPO — is a refinement of REINFORCE. Understanding it in your fingers is the prerequisite for the rest of this phase, and for Phase 10 · 07 (RLHF implementation) and Phase 10 · 08 (DPO).

## The Concept

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../images/09-rl/policy-gradient.svg)

**The policy gradient theorem.** For any policy `π_θ` parameterized by `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

where `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` is the discounted return from step `t`. The expectation is over full trajectories `τ` sampled from `π_θ`.

**The proof is short.** Differentiate `J(θ) = Σ_τ P(τ; θ) G(τ)` under the expectation. Use `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (the log-derivative trick). Factor `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`. The environment terms vanish. Two lines of algebra give you the theorem.

**Variance reduction tricks.** Vanilla REINFORCE has murderous variance — returns are noisy, `∇ log π` is noisy, their product is very noisy. Two standard fixes:

1. **Baseline subtraction.** Replace `G_t` with `G_t - b(s_t)` for any baseline `b(s_t)` that does not depend on `a_t`. Unbiased because `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Typical choice: `b(s_t) = V̂(s_t)` learned by a critic → actor-critic (Lesson 07).
2. **Reward-to-go.** Replace `Σ_t G_t · ∇ log π_θ(a_t | s_t)` with `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Only future returns matter for a given action — past rewards contribute zero-mean noise.

Combined, you get:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

which is REINFORCE with a baseline — the direct ancestor of A2C (Lesson 07) and PPO (Lesson 08).

**Softmax policy parameterization.** For discrete actions, the standard choice:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

where `f_θ` is any neural net that outputs a score per action. The gradient has a clean form:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

i.e., score of the taken action minus its expected value under the policy.

**Gaussian policy for continuous actions.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` has a closed form. That is all Phase 9 · 07's SAC needs.

## Build It

### Step 1: softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Use a linear policy (one weight vector per action) for a tabular env. For Atari, swap in a CNN and keep the softmax head.

### Step 2: sampling and log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: rollout with log-probs captured

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

The gradient `∇ log π(a|s) = e_a - π(·|s)` (onehot of `a` minus probabilities) is the heart of softmax policy gradients. Burn it into muscle memory.

### Step 5: baselines

A running mean of `G` over recent episodes is enough variance reduction to get a 4×4 GridWorld running; it takes ~500 episodes to converge. Upgrade the baseline to a learned `V̂(s)` and you get actor-critic.

## Pitfalls

- **Exploding gradients.** Returns can be huge. Always normalize `G` to `~N(0, 1)` across the batch before multiplying by `∇ log π`.
- **Entropy collapse.** The policy converges to a near-deterministic action too early, stops exploring, gets stuck. Fix: add entropy bonus `β · H(π(·|s))` to the objective.
- **High variance.** Vanilla REINFORCE needs thousands of episodes. A critic baseline (Lesson 07) or TRPO/PPO's trust region (Lesson 08) is the standard fix.
- **Sample inefficiency.** On-policy means you throw away every transition after one update. Off-policy corrections via importance sampling bring back data, at the cost of variance (PPO's ratio is a clipped IS weight).
- **Non-stationary gradients.** The same gradient from 100 episodes ago uses old `π`. On-policy methods update every few rollouts for this reason.
- **Credit assignment.** Without reward-to-go, past rewards contribute noise. Always use reward-to-go.

## Use It

In 2026, REINFORCE is rarely run directly but its gradient formula is everywhere:

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

When you read `loss = -advantage * log_prob` in a 2026 training script, that is REINFORCE with a baseline. Entire papers (DPO, GRPO, RLOO) are variance-reduction tricks on top of this one line.

## Ship It

Save as `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## Further Reading

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — the original REINFORCE paper.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — the modern policy-gradient theorem with function approximation.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — textbook presentation.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — clear pedagogical exposition with PyTorch code.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance-reduction and the natural-gradient view that connects REINFORCE to the trust-region family (TRPO, PPO).

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/06-policy-gradients-reinforce)

---

## Part 7 (ch177): Actor-Critic — A2C and A3C

> REINFORCE is noisy. Add a critic that learns `V̂(s)`, subtract it from the return, and you get an advantage that has the same expectation but far lower variance. That is actor-critic. A2C runs it synchronously; A3C runs it across threads. Both are the mental model for every modern deep-RL method.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## The Problem

Vanilla REINFORCE works, but its variance is terrible. Monte Carlo returns `G_t` can swing over a factor of 10 between episodes. Multiplying that noise by `∇ log π` and averaging produces a gradient estimator that takes thousands of episodes to move the policy the same distance you could move it with far fewer DQN updates.

The variance comes from using raw returns. If you subtract a baseline `b(s_t)` — any function of state, including a learned value — the expectation is unchanged and the variance drops. The best tractable baseline is `V̂(s_t)`. Now the quantity multiplying `∇ log π` is the *advantage*:

`A(s, a) = G - V̂(s)`

An action is good if it produced above-average return; bad if below. REINFORCE with a learned critic is *actor-critic*. The critic gives the actor a low-variance teacher. This is every deep-policy method after 2015 (A2C, A3C, PPO, SAC, IMPALA).

## The Concept

![Actor-critic: policy net plus value net, TD residual as advantage](../images/09-rl/actor-critic.svg)

**Two networks, one shared loss:**

- **Actor** `π_θ(a | s)`: the policy. Sampled to act. Trained with policy gradient.
- **Critic** `V_φ(s)`: estimates expected return from state. Trained to minimize `(V_φ(s) - target)²`.

**The advantage.** Two standard forms:

- *MC advantage:* `A_t = G_t - V_φ(s_t)`. Unbiased, higher variance.
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Biased (uses `V_φ`), far lower variance. Also called the *TD residual* `δ_t`.

**n-step advantage.** Interpolate between the two:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` is pure TD. `n = ∞` is MC. Most implementations use `n = 5` for Atari, `n = 2048` for PPO on MuJoCo.

**Generalized Advantage Estimation (GAE).** Schulman et al. (2016) proposed an exponentially weighted average over all n-step advantages:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

with `λ ∈ [0, 1]`. `λ = 0` is TD (low variance, high bias). `λ = 1` is MC (high variance, unbiased). `λ = 0.95` is the 2026 default — tune until the bias/variance dial is where you want it.

**A2C: synchronous advantage actor-critic.** Collect `T` steps across `N` parallel environments. Compute advantages for each step. Update actor and critic on the combined batch. Repeat. The simpler, more-scalable sibling of A3C.

**A3C: asynchronous advantage actor-critic.** Mnih et al. (2016). Spawn `N` worker threads, each running an env. Each worker computes gradients locally on its own rollout, then asynchronously applies them to a shared parameter server. No replay buffer needed — workers decorrelate by running different trajectories. A3C proved you could train on CPUs at scale. In 2026, GPU-based A2C (batched parallel envs) dominates because GPUs want large batches.

**The combined loss.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Three terms: policy-gradient loss, value regression, entropy bonus. `c_v ~ 0.5`, `c_e ~ 0.01` are canonical starting points.

## Build It

### Step 1: a critic

Linear critic `V_φ(s) = w · features(s)` updated with MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

On a tabular env the critic converges in a few hundred episodes. On Atari, replace the linear critic with a shared CNN trunk + value head.

### Step 2: n-step advantage

Given a rollout of length `T` and a bootstrapped final `V(s_T)`:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` is the critic target. `advantages` is what multiplies `∇ log π`.

### Step 3: combined update

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, one rollout per update, separate learning rates for actor and critic.

### Step 4: parallelization (A3C vs A2C)

- **A3C:** spin up `N` threads. Each runs its own env and its own forward pass. Periodically push gradient updates to a shared master. No locks on the master — races are ok, they just add noise.
- **A2C:** run `N` env instances in a single process, stack observations into a `[N, obs_dim]` batch, batched forward pass, batched backward pass. Higher GPU utilization, deterministic, easier to reason about. The default in 2026.

Our toy code is single-threaded for clarity; rewriting to batched A2C is three lines of numpy.

## Pitfalls

- **Critic bias before actor gradient.** If the critic is random, its baseline is uninformative and you are training on pure noise. Warm up the critic for a few hundred steps before turning on the policy gradient, or use a slow actor learning rate.
- **Advantage normalization.** Normalize advantages to zero-mean/unit-std per batch. Stabilizes training massively at near-zero cost.
- **Shared trunk.** Use a shared feature extractor for actor and critic on image inputs. Separate heads. The shared features free-ride on both losses.
- **On-policy contract.** A2C reuses data for exactly one update. More and your gradient is biased (importance-sampling correction is what PPO adds).
- **Entropy collapse.** Without `c_e > 0`, policy becomes near-deterministic in a few hundred updates and stops exploring.
- **Reward scale.** Advantage magnitudes depend on reward scale. Normalize rewards (e.g., running-std dividing) for consistent gradient magnitudes across tasks.

## Use It

A2C/A3C are rarely the final choice in 2026 but they are the architecture everything later refines:

| Method | Relation to A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio for multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C with a soft-value critic (next lesson) |
| GRPO (Phase 9 · 12) | A2C without the critic — group-relative advantage |
| DPO | A2C collapsed into a preference-ranking loss, no sampling |
| AlphaStar / OpenAI Five | A2C with league training + imitation pre-training |

If you see "advantage" in a 2026 paper, think actor-critic.

## Ship It

Save as `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Actor | "The policy net" | `π_θ(a\|s)`, updated by policy gradient. |
| Critic | "The value net" | `V_φ(s)`, updated by MSE regression to returns / TD targets. |
| Advantage | "How much better than average" | `A(s, a) = Q(s, a) - V(s)` or its estimators. Multiplier for `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; one-step advantage estimate. |
| GAE | "The interpolation knob" | Exponentially weighted sum of n-step advantages, parameterized by `λ`. |
| A2C | "Synchronous actor-critic" | Batched across envs; one gradient step per rollout. |
| A3C | "Async actor-critic" | Worker threads push gradients to a shared param server. Original paper; less common in 2026. |
| Bootstrap | "Use V at the horizon" | Truncate the rollout, add `γ^n V(s_{t+n})` to close the sum. |

## Further Reading

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, the original async actor-critic paper.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — foundations; pair this with Ch. 9 on function approximation when the critic is a neural net.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — scalable distributed actor-critic with V-trace off-policy correction.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — production A2C/PPO implementations worth reading.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — the foundational convergence result for the two-timescale actor-critic decomposition.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/07-actor-critic-a2c-a3c)

---

## Part 8 (ch178): Proximal Policy Optimization (PPO)

> A2C throws away each rollout after one update. PPO wraps the policy gradient in a clipped importance ratio so you can do 10+ epochs on the same data without the policy exploding. Schulman et al. (2017). Still the default policy-gradient algorithm in 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## The Problem

A2C (Lesson 07) is on-policy: the gradient `E_{π_θ}[A · ∇ log π_θ]` requires data sampled from the *current* `π_θ`. Take one update, and `π_θ` changes; the data you used is now off-policy. Re-use it and your gradient is biased.

Rollouts are expensive. On Atari, one rollout across 8 envs × 128 steps = 1024 transitions and a dozen seconds of environment time. Throwing that away after one gradient step is wasteful.

Trust Region Policy Optimization (TRPO, Schulman 2015) was the first fix: constrain each update so the KL divergence between old and new policy stays below `δ`. Theoretically clean, but requires a conjugate-gradient solve per update. Nobody runs TRPO in 2026.

PPO (Schulman et al. 2017) replaces the hard trust-region constraint with a simple clipped objective. One extra line of code. Ten epochs per rollout. No conjugate gradients. Good-enough theoretical guarantees. Nine years later it is still the default policy-gradient algorithm for everything from MuJoCo to RLHF.

## The Concept

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../images/09-rl/ppo.svg)

**The importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

This is the likelihood ratio of the new policy vs the policy that collected the data. `r_t = 1` means no change. `r_t = 2` means the new policy is twice as likely to take `a_t` as the old.

**The clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Two terms:

- If the advantage `A_t > 0` and the ratio tries to grow past `1 + ε`, the clip flattens the gradient — don't push a good action further than `+ε` above old probability.
- If the advantage `A_t < 0` and the ratio tries to grow past `1 - ε` (meaning we would make a bad action more likely compared to its clipped reduction), the clip caps the gradient — don't push a bad action below `-ε`.

The `min` handles the other direction: if the ratio has moved in the *beneficial* direction, you still get the gradient (no clipping on the side that would hurt you).

Typical `ε = 0.2`. Plot the objective as a function of `r_t`: a piecewise-linear function with a flat roof on the "good side" and a flat floor on the "bad side."

**The full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Same actor-critic structure as A2C. Three coefficients, usually `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**The training loop.**

1. Collect `N × T` transitions across `N` parallel envs for `T` steps each.
2. Compute advantages (GAE), freeze them as constants.
3. Freeze `π_{θ_old}` as a snapshot of current `π_θ`.
4. For `K` epochs, for each minibatch of `(s, a, A, V_target, log π_old(a|s))`:
   - Compute `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Apply `L^{CLIP}` + value loss + entropy.
   - Gradient step.
5. Discard the rollout. Return to step 1.

`K = 10` and minibatches of 64 is a standard hyperparameter set. PPO is robust: the exact numbers rarely matter within ±50%.

**KL-penalty variant.** The original paper proposed an alternative using an adaptive KL penalty: `L = L^{PG} - β · KL(π_θ || π_old)` with `β` adjusted based on observed KL. The clipping version became dominant; the KL variant survives in RLHF (where KL to the reference policy is a separate constraint you always want anyway).

## Build It

### Step 1: capture `log π_old(a | s)` at rollout time

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

The snapshot is taken once, at rollout time. It does not change during the update epochs.

### Step 2: compute GAE advantages (Lesson 07)

Same as A2C. Normalize across the batch.

### Step 3: clipped surrogate update

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

The "clipped → zero gradient" pattern is the heart of PPO. If the new policy has already drifted too far in the beneficial direction, the update stops.

### Step 4: value and entropy

Add standard MSE to the critic target and an entropy bonus on the actor, same as A2C.

### Step 5: diagnostics

Three things to watch every update:

- **Mean KL** `E[log π_old - log π_θ]`. Should stay in `[0, 0.02]`. If it blows past `0.1`, reduce `K_EPOCHS` or `LR`.
- **Clip fraction** — the fraction of samples whose ratio lies outside `[1-ε, 1+ε]`. Should be `~0.1-0.3`. If `~0`, the clip never triggers → raise `LR` or `K_EPOCHS`. If `~0.5+`, you are over-fitting the rollout → lower them.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Critic quality metric. Should climb toward 1 as the critic learns.

## Pitfalls

- **Clip coefficient mistuned.** `ε = 0.2` is the de-facto standard. Going to `0.1` makes updates too timid; `0.3+` invites instability.
- **Too many epochs.** `K > 20` routinely destabilizes because the policy drifts far from `π_old`. Cap epochs, especially for large networks.
- **No reward normalization.** Large reward scales eat into the clip range. Normalize rewards (running std) before computing advantages.
- **Forgetting advantage normalization.** Per-batch zero-mean/unit-std normalization is standard. Skipping it wrecks PPO on most benchmarks.
- **Learning rate not decayed.** PPO benefits from linear LR decay to zero. Constant LR is often worse.
- **Importance ratio math errors.** Always `exp(log_new - log_old)` for numerical stability, not `new / old`.
- **Wrong gradient sign.** Maximize the surrogate = *minimize* `-L^{CLIP}`. A flipped sign is the most common PPO bug.

## Use It

PPO is 2026's default RL algorithm across a surprising number of domains:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

The PPO *loss shape* — clipped surrogate + value + entropy — is the scaffolding for DPO, GRPO, and nearly every RLHF pipeline.

## Ship It

Save as `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; deviation from the policy that collected the data. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient past the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Limit each update's KL to guarantee monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — should be 0.1-0.3; outside means mistuned. |
| Multi-epoch training | "Data reuse" | K epochs on each rollout; variance cost traded for sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO is nominally on-policy but K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; used in RLHF where KL-to-reference is already a constraint. |

## Further Reading

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — the paper.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO's predecessor.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — every PPO hyperparameter ablated.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; the PPO-in-RLHF recipe.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — clean modern exposition with PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — reference single-file PPO used by many papers.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — the production recipe for PPO on language models; read alongside Lesson 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — the "37 code-level optimizations" paper; which PPO tricks are load-bearing and which are folklore.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/09-reinforcement-learning/08-ppo)
