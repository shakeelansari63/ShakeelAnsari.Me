# Canary, Load Testing, SRE, Chaos & Compliance

> Combined lessons (7 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch378): Shadow Traffic, Canary Rollout, and Progressive Deployment for LLMs

> LLM rollouts combine the hardest parts of software deployment: no unit tests, diffuse failure modes, delayed signals. The sequence is (1) shadow mode — duplicate prod requests to candidate model, log, compare with zero user impact; catches obvious distribution issues but is not a quality guarantee; (2) canary rollout — progressive traffic shift 10% → 25% → 50% → 75% → 100% with gates at each step; track latency percentiles, cost/request, error/refusal rate, output length distribution, user-feedback rate; (3) A/B testing for distinct alternatives after stability confirmed. Non-determinism is irreducible — up to 15% accuracy variation across runs with identical inputs due to GPU FP non-associativity plus batch-size variance. Cost is a variable, not constant — a 20% better model can be 3x more expensive per call. Rollback speed is decisive: if rollback requires redeploy, you are too slow. Policy lives in config/flags; model lives in registry with pinned digests; rollback = flip policy + revert threshold + pin old model in seconds.

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish shadow mode (zero-impact compare), canary (live traffic progressive), and A/B (stability-confirmed comparison).
- Enumerate five LLM-specific canary metrics (latency, cost/request, error/refusal, output-length distribution, user feedback).
- Explain why LLM non-determinism (up to 15%) changes what "stable" means in a rollout.
- Design a rollback path that takes seconds (policy flip) not hours (redeploy).

## The Problem

You ship a new model. Offline evals show 3% accuracy gain. You flip it on in production. Within 24 hours, cost is up 40%, user thumbs-down is up 8%, three customer tickets report "weird answers." You roll back. Redeploy takes 3 hours. Your weekend is ruined.

Every piece of that was avoidable. Shadow mode would have caught the 40% cost spike before any user saw it. Canary would have stopped at 10% when thumbs-down moved. Policy-flag rollback would have taken 30 seconds. The discipline is what fills in the gap between "offline evals look good" and "real users are happy."

## The Concept

### Shadow mode

Candidate receives the same requests as production; outputs are logged, not returned to users. Zero user impact. Log:

- Output content (diff against production).
- Token counts (cost delta).
- Latency.
- Refusal and error.

Catches: cost blow-ups, length regressions, obvious refusal changes, hard errors. Does NOT catch: quality delta users would perceive. Shadow is a smoke test, not a quality test.

### Canary rollout

Progressive traffic shift with gates. Typical progression: 1% → 10% → 25% → 50% → 75% → 100%. Gate on 5 metrics at each step:

1. **Latency percentiles** — P50, P95, P99. Breach: canary has P99 > 1.5x baseline.
2. **Cost per request** — blended $. Breach: >20% above baseline.
3. **Error / refusal rate** — 5xx plus explicit refusals. Breach: 2x baseline.
4. **Output length distribution** — mean + P99. Breach: distributional shift.
5. **User-feedback rate** — thumbs-down / ticket filings. Breach: 1.5x baseline.

### Non-determinism is the new variance

Identical inputs produce non-identical outputs. Reasons:

- GPU FP non-associativity (floating-point reduction order varies by batch).
- Batch-size variance (same prompt in a batch of 128 vs batch of 16).
- Sampling (temperature > 0).

Measured: up to 15% accuracy variation run-to-run on identical eval sets. "Stable" in a rollout means metrics are within expected variance, not identical to baseline. Set gates above the noise floor.

### Cost is a variable

A 20% better model can be 3x more expensive per call. Cost/request is one of the five gates. Shipping a "better" model that breaks unit economics is a rollback case.

### Rollback is the weapon

- Policy flag (feature flag system): flip percentage in config; takes seconds.
- Model pinning (registry digest): pinned model does not auto-upgrade.
- Rollback = revert flag + set pinned digest to previous. Seconds, not hours.

If your stack requires redeploy to rollback, fix that before rolling.

### Tooling

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery controllers. Integrate with Istio/Linkerd weighted routing.

**Istio weighted routing** — service-mesh-level traffic split.

**KServe / Seldon Core** — model serving with built-in canary.

**Feature flags** — LaunchDarkly, Flagsmith, Unleash. Policy-level flip, no redeploy.

### Metrics cadence

Canary gates check every 5-15 minutes depending on traffic volume. 1% traffic with 10 req/min gives 50-150 data points per window — enough for latency but noisy for user feedback. 10% gives ~10x more. Progressions should pause long enough to accumulate enough samples at each step.

### The A/B step is optional

If the new model is distinctly different (different behavior, different cost curve, different tone), A/B test it at 50% after canary passes. If it's just an improved version, skip to 100% when canary gates pass.

### Numbers you should remember

- Canary progression: 1% → 10% → 25% → 50% → 75% → 100%.
- Non-determinism ceiling: up to 15% run-to-run variance on identical inputs.
- Five canary metrics: latency, cost, error/refusal, output length, user feedback.
- Cost gate: >20% above baseline is a breach.
- Rollback: seconds, not hours.

## Use It

`code/main.py` simulates a canary rollout with injected regressions. Reports which stage the rollout halts at and which gate triggered.

## Ship It

This lesson produces `outputs/skill-rollout-runbook.md`. Given candidate model, baseline, and risk tolerance, designs shadow→canary→100% plan.

## Exercises

1. Run `code/main.py`. Inject a 25% cost regression. At which stage does the canary halt?
2. Your new model has 3% accuracy gain offline but cost/request is +18%. Is it a ship? Depends on the policy — write both paths.
3. Design a rollback that takes under 60 seconds end-to-end. List the required infrastructure.
4. Non-determinism shows ±7% on your eval. Set canary gates so you don't false-alarm. What multipliers do you use?
5. Shadow mode catches a 40% cost spike before canary. Write the alert rule that fires in shadow.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Shadow mode | "duplicate to new" | Zero-impact send-to-candidate for logging |
| Canary | "progressive traffic" | Gradual user-exposed rollout with gates |
| Gates | "rollout checks" | Metric thresholds that block progression |
| Non-determinism | "LLM variance" | Irreducible run-to-run differences |
| Policy flag | "flag flip rollback" | Config-level rollback, seconds not hours |
| Model pin | "registry digest" | Immutable reference to a model version |
| Argo Rollouts | "K8s progressive" | Kubernetes-native canary/rollback controller |
| KServe | "inference K8s" | Model serving with canary primitives |
| Istio weighted | "mesh split" | Service-mesh traffic splitter |

## Further Reading

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/20-shadow-canary-progressive)

---

## Part 2 (ch379): A/B Testing LLM Features — GrowthBook, Statsig, and the Vibes Problem

> Traditional A/B testing was not built for non-deterministic LLMs. The critical distinction: evals answer "can the model do the job?" A/B tests answer "do users care?" Both are required; shipping on vibe checks is over. What to test in 2026: prompt engineering (wording), model selection (GPT-4 vs GPT-3.5 vs OSS; accuracy vs cost vs latency), generation parameters (temperature, top-p). Real cases: a chatbot reward-model variant delivered +70% conversation length and +30% retention; Nextdoor AI subject-line experiments delivered +1% CTR after reward-function refinement; Khan Academy Khanmigo iterated on a latency-vs-math-accuracy axis. Platform split: **Statsig** (acquired by OpenAI for $1.1B in September 2025) — sequential testing, CUPED, all-in-one. **GrowthBook** — open-source, warehouse-native, Bayesian + Frequentist + Sequential engines, CUPED, SRM checks, Benjamini-Hochberg + Bonferroni corrections. You pick based on warehouse-SQL preference and whether "acquired by OpenAI" matters to your organization.

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish evals ("can the model do the job") from A/B tests ("do users care").
- Enumerate three testable axes (prompt, model, parameters) and pick the metric for each.
- Explain CUPED, sequential testing, and Benjamini-Hochberg multiple-comparison corrections.
- Pick Statsig or GrowthBook based on warehouse-SQL posture and corporate acquisition stance.

## The Problem

You hand-tuned a system prompt. It feels better. You ship it. Conversion changes by noise. You blame the metric. Or you shipped a new model and conversion didn't move — did the model degrade or was the change too small to detect? You don't know, because you shipped without an A/B.

Evals answer whether the model can do a task on a labeled set. They do not answer whether users prefer the output. Only a controlled online experiment answers that, and only if the experiment has enough power, controls for non-determinism, and corrects for multiple comparisons.

## The Concept

### Evals vs A/B tests

**Evals** — offline, labeled set, judge (rubric or LLM-as-judge or human). Answer: "Is the output correct / helpful / safe on this fixed distribution?"

**A/B test** — online, live users, randomized. Answer: "Does the new variant move the user-level metric that matters?"

Both required. Evals catch regressions before exposure; A/B confirms product impact after.

### What to test

1. **Prompt engineering** — wording, system-prompt structure, examples. Metric: task success, user retention, cost/request.
2. **Model selection** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Metric: accuracy (task) + cost/request + latency P99. Multi-objective.
3. **Generation parameters** — temperature, top-p, max_tokens. Metric: task-specific (output diversity vs determinism).

### CUPED — variance reduction

Controlled-experiments Using Pre-Experiment Data. Regress out pre-period variance before comparing post-period. Typical variance reduction: 30-70%. Effective sample size goes up for free.

Implementation: both Statsig and GrowthBook implement.

### Sequential testing

Classical A/B assumes fixed sample size. Sequential tests ("peek-and-decide") control false-positive rate under repeated looks. Always-valid sequential procedures (mSPRT, Howard's confidence sequences) let you stop early on clear winners.

### Multiple-comparison corrections

Running 20 A/B tests at 95% confidence produces one false positive by chance. Bonferroni correction tightens α per-test; Benjamini-Hochberg controls false-discovery rate. GrowthBook implements both.

### SRM — sample ratio mismatch

Assignment hash randomizes users to variants. If 50/50 split delivers 47/53, something is broken — SRM check flags it. Both platforms implement.

### Statsig vs GrowthBook

**Statsig**:
- Acquired by OpenAI for $1.1B (September 2025). Hosted, SaaS.
- Sequential testing, CUPED, held-out populations.
- All-in-one: feature flags + experimentation + observability.
- Best fit: team already wants a bundled product, doesn't care about OpenAI ownership.

**GrowthBook**:
- Open-source (MIT); warehouse-native (reads from Snowflake/BigQuery/Redshift directly).
- Multiple engines: Bayesian, Frequentist, Sequential.
- CUPED, SRM, Bonferroni, BH corrections.
- Self-host or managed cloud.
- Best fit: warehouse-SQL shop, data team controls the metric layer, wants OSS.

### Non-determinism complicates power

Same prompt produces varying outputs. Traditional power calculations assume IID observations. With LLM non-determinism, effective sample size is lower than nominal. Multiply required sample size by ~1.3-1.5x as a safety margin.

### Real case outcomes

- Chatbot reward model variant: +70% conversation length, +30% retention.
- Nextdoor subject lines: +1% CTR after reward-function refinement.
- Khan Academy Khanmigo: iterative latency-vs-math-accuracy trade.

### The anti-pattern: shipping on vibes

Every senior engineer can name a feature that was shipped because "it feels better" with no A/B. Most of them regressed product metrics the team didn't notice for months. A/B is the forcing function.

### Numbers you should remember

- Statsig acquired by OpenAI: $1.1B, September 2025.
- GrowthBook: open-source MIT; Bayesian + Frequentist + Sequential.
- CUPED variance reduction: 30-70%.
- LLM non-determinism → +30-50% sample-size buffer.

## Use It

`code/main.py` simulates a sequential A/B test with fixed and sequential boundaries. Shows how sequential lets you stop early.

## Ship It

This lesson produces `outputs/skill-ab-plan.md`. Given feature change, workload, baseline, picks platform, gates, sample size.

## Exercises

1. Run `code/main.py`. For an expected 5% lift with baseline 3% conversion, what sample size to 80% power?
2. Pick Statsig or GrowthBook for a healthcare-regulated on-prem customer.
3. Design an A/B that tests GPT-4 vs GPT-3.5 on cost-per-resolved-ticket. What's the primary metric, guardrail metric, secondary?
4. Your canary passes but A/B shows -1.2% conversion. Do you ship? Write the escalation criteria.
5. Apply CUPED to a pre-period with 60% of the variance of post. Compute the effective-sample-size boost.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Eval | "offline test" | Labeled-set evaluation of model capability |
| A/B test | "experiment" | Live randomized comparison on users |
| CUPED | "variance reduction" | Pre-period regression to reduce variance |
| Sequential test | "peek-ok test" | Always-valid procedure allowing early stop |
| Multiple comparison | "the family error" | Running many tests inflates false positives |
| Bonferroni | "tight correction" | Divide α by number of tests |
| Benjamini-Hochberg | "BH FDR" | False-discovery-rate control, less conservative |
| SRM | "bad split" | Sample ratio mismatch; assignment bug |
| Statsig | "OpenAI owned" | Commercial all-in-one, acquired 2025 |
| GrowthBook | "the OSS one" | MIT warehouse-native platform |
| mSPRT | "sequential probability ratio test" | Classical sequential procedure |

## Further Reading

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/21-ab-testing-llm-features)

---

## Part 3 (ch380): Load Testing LLM APIs — Why k6 and Locust Lie

> Traditional load testers were not designed for streaming responses, variable output lengths, token-level metrics, or GPU saturation. Two traps bite most teams. The GIL trap: Locust's token-level measurement runs tokenization under the Python GIL, which competes with request generation under heavy concurrency; tokenization backlog then inflates reported inter-token latency — your client is the bottleneck, not the server. The prompt-uniformity trap: identical prompts in a loop test one point on the token distribution; real traffic has variable length and diverse prefix matches. LLMPerf fixes this with `--mean-input-tokens` + `--stddev-input-tokens`. Tool mapping in 2026: LLM-specialized (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) for token-level accuracy; **k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)** — streaming-aware, Kubernetes-native distributed via TestRun/PrivateLoadZone CRDs, best for CI/CD gates; Vegeta for Go constant-rate saturation; Locust 2.43.3 only with LLM-Locust extension for streaming. Load patterns: steady-state, ramp, spike (autoscaling test), soak (memory leaks).

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## Learning Objectives

- Explain the two anti-patterns (GIL trap, prompt-uniformity trap) that make generic load testers lie for LLM APIs.
- Pick a tool for a given purpose: LLMPerf (benchmark run), k6 + streaming extension (CI gate), guidellm (large-scale synthetic), GenAI-Perf (NVIDIA reference).
- Design four load patterns (steady, ramp, spike, soak) and name the failure mode each catches.
- Build a realistic prompt distribution using mean + stddev of input tokens rather than fixed length.

## The Problem

You k6-tested your LLM endpoint at 500 concurrent users. It held. You shipped. In production at 200 actual users the service fell over — P99 TTFT exploded, GPUs pinned.

Two things happened. First, k6 sent 500 identical prompts — your request-coalescing and prefix caching made it look like you were handling 500 concurrent decodes when you were actually handling one. Second, k6 doesn't track inter-token latency on streaming responses the way the eye experiences it; it sees one HTTP connection, not 500 tokens arriving at varying intervals.

Load testing for LLMs is its own discipline.

## The Concept

### The GIL trap (Locust)

Locust uses Python and runs tokenization client-side under the GIL. Under high concurrency the tokenizer queues behind request generation. Reported inter-token latency includes client-side tokenization backlog. You think the server is slow; it's the test harness.

Fix: LLM-Locust extension moves tokenization to separate processes, or use a compiled-language harness (k6, LLMPerf using tokenizers.rs).

### The prompt-uniformity trap

All known load testers let you configure one prompt. In a loop test of 10,000 iterations the exact same prompt sends each time. Server sees the same prefix every time — prefix cache hits approach 100%, throughput looks great.

Fix: sample from a prompt distribution. LLMPerf uses `--mean-input-tokens 500 --stddev-input-tokens 150` — diverse lengths, diverse content.

### Four load patterns

1. **Steady-state** — constant RPS for 30-60 min. Catches: baseline performance regressions.
2. **Ramp** — linearly increase RPS from 0 to target over 15 min. Catches: capacity breakpoint, warm-up anomalies.
3. **Spike** — sudden 3-10x RPS for 2 min then back. Catches: autoscaling latency, queue saturation, cold-start impact.
4. **Soak** — steady-state for 4-8 hours. Catches: memory leaks, connection-pool drift, observability overflow.

### 2026 tool mapping

**LLMPerf** (Anyscale) — Python but Rust-backed tokenization. Mean/stddev prompts. Streaming-aware. Best default for performance runs.

**NVIDIA GenAI-Perf** — NVIDIA's reference. Uses Triton client; comprehensive metric coverage. Note its ITL excludes TTFT; LLMPerf's includes it. Two tools produce different TPOT for the same server.

**LLM-Locust** (TrueFoundry) — Locust extension that fixes the GIL trap. Familiar Locust DSL + streaming metrics.

**guidellm** — large-scale synthetic benchmarking.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)**:
- k6 itself (Go, compiled, no GIL) added streaming-aware metrics.
- k6 Operator uses TestRun / PrivateLoadZone CRDs for Kubernetes-native distributed testing.
- Best for CI/CD gates and SLA testing.

**Vegeta** — Go, simpler than k6. Constant-rate HTTP saturation. Not LLM-aware but good for gateway / rate-limit testing.

**Locust 2.43.3 stock** — has the GIL trap for LLM. Only with LLM-Locust extension.

### SLA gate in CI

Run k6 on the PR with:

- 30-50 iterations each at baseline RPS.
- Gate: P50/P95 TTFT, 5xx < 5%, TPOT under threshold.
- Break the build on breach.

### Realistic prompt distribution

Build from real traffic samples (if you have them) or from published distributions (e.g., ShareGPT prompts for chat, HumanEval for code). Feed the mean + stddev to LLMPerf. Avoid loop-with-one-prompt at all costs.

### Numbers you should remember

- k6 Operator 1.0 GA: September 2025.
- k6 v2026.1.0: streaming-aware metrics.
- Typical LLMPerf run: 100-1000 requests at concurrency X.
- Typical CI gate: 30-50 iterations per PR.
- Four patterns: steady, ramp, spike, soak.

## Use It

`code/main.py` simulates a load test with realistic prompt distribution, measures effective TPOT, and demonstrates the uniform-prompt trap.

## Ship It

This lesson produces `outputs/skill-load-test-plan.md`. Given workload and SLA, picks tool and designs the four load patterns.

## Exercises

1. Run `code/main.py`. Compare uniform vs realistic distribution — where is the gap?
2. Write the k6 script for a CI gate: TTFT P95 < 800 ms at 100 concurrent, runtime 5 minutes.
3. Your soak test shows memory growing 50 MB/hour. Name three causes and the instrumentation to pick between them.
4. Spike test from 10 RPS to 100 RPS. What's the expected recovery time if Karpenter + vLLM production-stack are in place (Phase 17 · 03 + 18)?
5. GenAI-Perf reports TPOT=6ms; LLMPerf reports TPOT=11ms on the same server. Explain.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| LLMPerf | "the LLM harness" | Anyscale benchmark tool, streaming-aware |
| GenAI-Perf | "NVIDIA tool" | NVIDIA reference harness |
| LLM-Locust | "Locust for LLMs" | Locust extension fixing GIL trap |
| guidellm | "synthetic benchmark" | Large-scale synthetic tool |
| k6 Operator | "K8s k6" | CRD-based distributed k6 |
| GIL trap | "Python client overhead" | Tokenization backlog inflates reported latency |
| Prompt-uniformity trap | "single-prompt lie" | Loop with same prompt hits cache, inflates throughput |
| Steady-state | "constant load" | Flat RPS for N minutes |
| Ramp | "linear up" | 0 to target over duration |
| Spike | "burst test" | Sudden multiplier then revert |
| Soak | "long test" | Hours for leak detection |

## Further Reading

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/22-load-testing-llm-apis)

---

## Part 4 (ch381): SRE for AI — Multi-Agent Incident Response, Runbooks, Predictive Detection

> AI SRE uses LLMs grounded in infrastructure data (logs, runbooks, service topology) via RAG to automate investigation, documentation, and coordination phases. The 2026 architecture pattern is multi-agent orchestration — specialized agents (logs, metrics, runbooks) coordinated by a supervisor; AI proposes hypotheses and queries, humans approve judgment calls. Datadog Bits AI and Azure SRE Agent ship this as managed products. Runbooks are evolving: NeuBird Hawkeye uses adversarial evaluation (two models analyze the same incident; agreement = confidence, disagreement = uncertainty); operational memory persists across team changes. Auto-remediation stays cautious: AI suggests, humans approve. Fully autonomous action is narrow (restart pod, rollback specific deploy) with tight guardrails — anyone selling "set it and forget it" is overselling. Emerging frontier: pre-incident prediction. MIT research reports an LLM trained on historical logs + GPU temps + API error patterns predicted 89% of outages 10-15 min early. Projection: 95% of enterprise LLMs have automated failover by end-2026.

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Diagram the multi-agent AI SRE architecture: supervisor + specialized agents (logs, metrics, runbooks) + human approval gate.
- Explain why auto-remediation is narrow (restart pod, revert deploy) rather than broad (re-architect service).
- Name the adversarial evaluation pattern (NeuBird Hawkeye): two models agree = confidence; disagree = escalate.
- Cite the MIT 89% early-detection result and the operational constraint: predictions without actuation are just dashboards.

## The Problem

An on-call engineer gets paged at 3 a.m. "High error rate in checkout." They check Datadog, Loki, three runbooks, the deploy log. 30 minutes later they realize the root cause is a vLLM OOM from a KV cache spike. They restart the pod; error clears.

In 2026 the first 20 minutes of that investigation are automatable. Grouping logs by service, correlating to recent deploys, matching against runbooks — all are RAG + tool-use. A supervised agent can do first-pass triage and present a hypothesis before the human opens Datadog.

Fully autonomous remediation is a different problem. Restart pod: safe. Scale GPU pool: safe if policy allows. Re-architect the service: absolutely not. The discipline is drawing the narrow line.

## The Concept

### Multi-agent architecture

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor breaks the incident into sub-queries. Specialized agents have tool access (log search, PromQL, doc retrieval). Supervisor synthesizes, presents hypothesis + evidence to human. Human approves or redirects.

### Auto-remediation scope

**Safe (narrow)**: restart pod, revert specific deploy, scale pool within pre-approved bounds, enable pre-approved feature flag.

**Not safe (broad)**: change service topology, modify resource limits, deploy new code, change IAM, alter databases.

Anyone selling "set it and forget it" is overselling. The safe set grows as AI SRE matures, but the boundary is real.

### Adversarial evaluation (NeuBird Hawkeye)

Two models independently analyze the same incident. If they agree on root cause, confidence is high. If they disagree, escalate to human with both hypotheses visible. Simple pattern, effective filter against hallucinated root causes.

### Operational memory

Team turnover is the silent kill of traditional SRE — tribal knowledge leaves. AI SRE stores runbooks + post-mortems in a vector DB; agents retrieve on every new incident. When new engineers join, the AI has full history.

### Pre-incident prediction

MIT 2025 research: LLM trained on historical logs, GPU temperatures, API error patterns predicted 89% of outages 10-15 minutes before they happened on the test set.

Reality check: predictions without actuation are dashboards. The operational question is "when we predict, what do we do?" Pre-emptive drain? Pager? Auto-scale? The answer is policy-specific.

### Products in 2026

- **Datadog Bits AI** — managed SRE copilot inside Datadog.
- **Azure SRE Agent** — Azure-native.
- **NeuBird Hawkeye** — adversarial eval + operational memory.
- **PagerDuty AIOps** — triage + deduplication.
- **Incident.io Autopilot** — incident commander + coordination.

### Runbooks as code

Runbooks evolve from Confluence pages to versioned markdown with structured sections (symptom, hypothesis, verify, act). Structured runbooks feed better RAG retrieval. Start any AI-SRE rollout by turning unstructured runbooks into structured.

### Numbers you should remember

- MIT early-detection: 89% of outages, 10-15 min lead time.
- Multi-agent triage: supervisor + (logs, metrics, runbooks) + human.
- Safe auto-remediation set: restart pod, revert deploy, scale within bounds.
- Adversarial eval: two models independent; agreement = confidence.

## Use It

`code/main.py` simulates a multi-agent triage: log agent finds error, metric agent finds CPU spike, runbook agent matches to known issue. Supervisor ranks hypotheses.

## Ship It

This lesson produces `outputs/skill-ai-sre-plan.md`. Given current on-call, incident volume, team maturity, designs an AI SRE rollout.

## Exercises

1. Run `code/main.py`. What if the log and metric agents disagree? How does the supervisor resolve?
2. Define three "safe" auto-remediation actions for your service. Justify each.
3. Write a structured runbook template: sections, required fields, verification commands.
4. Predictive detection fires at 12 min lead. What's your policy — pager, pre-drain, or both?
5. Argue whether a 3-person team should adopt AI SRE in 2026 or wait. Consider maturity, volume, risk.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| AI SRE | "agent for on-call" | LLM-backed incident investigation + coordination |
| Supervisor agent | "the orchestrator" | Top-level agent breaking incidents into sub-queries |
| Specialized agent | "domain agent" | Sub-agent with tool access (logs, metrics, runbooks) |
| Auto-remediation | "AI fixes it" | Narrow pre-approved action; NOT broad re-architecture |
| Operational memory | "vector runbooks" | Post-mortems + runbooks in vector DB for RAG |
| Adversarial eval | "two-model check" | Independent analyses; agreement = confidence |
| NeuBird Hawkeye | "the adversarial one" | Product with adversarial-eval + memory pattern |
| Bits AI | "Datadog's SRE agent" | Datadog-managed AI SRE |
| Pre-incident prediction | "early detection" | 10-15 min lead time on outage prediction |

## Further Reading

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/23-sre-for-ai)

---

## Part 5 (ch382): Chaos Engineering for LLM Production

> Chaos engineering for LLMs is its own discipline in 2026. Prerequisites before running experiments in production: defined SLI/SLO, trace+metric+log observability, automated rollback, runbooks, on-call. Architecture has four planes: control (experiment scheduler), target (services, infra, data stores), safety (guards + abort + traffic filters), observability (metrics + traces + logs), feedback (into SLO adjustments). Guardrails are mandatory: burn-rate alerts pause experiments if daily error-budget burn > 2x expected; suppression windows + trace-ID correlation dedupe alert noise. Cadence: weekly small canary + SLO review; monthly game day + postmortem; quarterly cross-team resilience audit + dependency mapping. LLM-specific experiments: memory overload, network failures, provider outages, malformed prompts, KV cache eviction storms. Tooling: Harness Chaos Engineering (LLM-derived recommendations, blast-radius downscaling, MCP tool integration); LitmusChaos (CNCF); Chaos Mesh (CNCF Kubernetes-native).

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Name the five chaos engineering prerequisites (SLI/SLO, observability, rollback, runbooks, on-call) and explain why skipping any breaks the practice.
- Diagram the four planes (control, target, safety, observability) and the feedback loop into SLO.
- Enumerate five LLM-specific experiments (memory overload, network fail, provider outage, malformed prompt, KV eviction storm).
- Pick a tool — Harness, LitmusChaos, Chaos Mesh — given stack.

## The Problem

Chaos testing in traditional stacks is established. LLM stacks add new failure modes. A 4K-token prompt with a poison character stalls the tokenizer for 12 seconds. An upstream provider 429s; your gateway retries; your service OOMs on retry-amplified concurrency. A KV cache eviction storm under burst load causes re-prefill cascades that saturate compute.

None of these show up in unit tests. Chaos engineering is how you discover them before users do.

## The Concept

### Prerequisites

Don't run chaos in production without:

1. **SLI/SLO** — defined service-level indicators and objectives.
2. **Observability** — traces, metrics, logs, wired to dashboards.
3. **Automated rollback** — Phase 17 · 20 policy-flag rollback.
4. **Runbooks** — structured, Phase 17 · 23.
5. **On-call** — someone to respond.

Missing any means chaos becomes real incident.

### Four planes + feedback

**Control plane** — experiment scheduler (Litmus workflow, Chaos Mesh schedule, Harness UI).

**Target plane** — services, pods, nodes, load balancers, data stores.

**Safety plane** — kill switch, suppression windows, blast-radius limits, error-budget gates.

**Observability plane** — normal metrics + trace-ID correlation to distinguish chaos-induced from natural failures.

**Feedback loop** — findings feed back into SLO adjustment, runbook updates, code fixes.

### Guardrails are mandatory

- **Burn-rate alert**: pause experiment if daily error-budget burn exceeds 2x expected.
- **Suppression windows**: silence non-experiment alerts in the blast radius during experiment.
- **Trace-ID correlation**: all experiment-induced errors carry a tag so on-call can dedupe.

### Five LLM-specific experiments

1. **Memory overload** — force a KV cache preemption storm by sending long-context requests with high concurrency. Observe: does the service gracefully shed or crash?

2. **Network failure** — cut connectivity between inference gateway and provider. Observe: does fallback kick in within SLA? (Phase 17 · 19)

3. **Provider outage simulation** — 100% 429 from OpenAI. Observe: does routing failover to Anthropic? (Phase 17 · 16, 19)

4. **Malformed prompt** — inject tokenizer-stalling payload (e.g., deeply nested unicode, huge UTF-8 codepoint). Observe: does a single request lock up a worker?

5. **KV eviction storm** — force eviction by saturating vLLM block budget. Observe: does LMCache recover or does service degrade?

### Cadence

- **Weekly** — small canary experiments in staging, maybe 5% prod.
- **Monthly** — scheduled game day on a specific scenario; cross-team attendance; postmortem.
- **Quarterly** — cross-team resilience audit; dependency map update.

### Tooling

- **Harness Chaos Engineering** — commercial; AI-derived experiment recommendations; blast-radius downscaling; MCP tool integration.
- **LitmusChaos** — CNCF graduated; Kubernetes workflow-based.
- **Chaos Mesh** — CNCF sandbox; Kubernetes-native CRD style.
- **Gremlin** — commercial; broad support.
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings.

### Starting small

First experiment: pod-kill one decode replica under steady traffic. Observe rerouting and recovery. If this works and looks safe, graduate to network chaos.

First LLM-specific experiment: inject one provider 429 for 5 minutes. Observe fallback. Most teams discover their fallback wasn't fully tested.

### Numbers you should remember

- Four planes: control, target, safety, observability.
- Burn-rate pause: 2x expected daily budget burn.
- Cadence: weekly canary, monthly game day, quarterly audit.
- Five LLM experiments: memory, network, provider, malformed prompt, KV storm.

## Use It

`code/main.py` simulates three chaos experiments with safety plane gates. Reports which experiments would trip the burn-rate abort.

## Ship It

This lesson produces `outputs/skill-chaos-plan.md`. Given stack and maturity, picks first three experiments and the tooling.

## Exercises

1. Run `code/main.py`. Which experiment trips the burn-rate gate and why?
2. Design the first five chaos experiments for a vLLM-based RAG service. Include success criteria.
3. Your burn-rate alert paused an experiment. How do you determine root cause — chaos or natural?
4. Argue whether chaos should run in production or only staging. When is production the right answer?
5. Name three LLM-specific failure modes that generic network-chaos cannot reproduce.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SLI / SLO | "service targets" | Indicator + objective; required prerequisite |
| Blast radius | "scope" | Set of services / users affected by experiment |
| Burn-rate alert | "budget gate" | Fires when error-budget burn rate > 2x expected |
| Game day | "monthly drill" | Scheduled cross-team chaos exercise |
| LitmusChaos | "CNCF workflow" | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes-native chaos |
| Harness CE | "commercial AI-assisted" | Harness chaos with AI recommendations |
| Malformed prompt | "tokenizer bomb" | Input that stalls tokenization |
| KV eviction storm | "preemption cascade" | Mass eviction triggering re-prefills |

## Further Reading

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/24-chaos-engineering-llm)

---

## Part 6 (ch383): Security — Secrets, API Key Rotation, Audit Logs, Guardrails

> Eliminate secret sprawl via centralized vaults (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Never store credentials in config files, env files in VCS, spreadsheets. Use IAM roles over static keys; OIDC for CI/CD. The AI-gateway pattern is the 2026 solution: apps → gateway → model provider, with gateway pulling credentials from vault at runtime. Rotate in vault and all apps pick up in minutes — no redeploys, no Slack "who has the new key" messages. Rotation policy ≤90 days; scan with TruffleHog / GitGuardian / Gitleaks on every commit. Zero-trust: MFA, SSO, RBAC/ABAC, short-lived tokens, device posture. PII scrubbing uses entity recognition to mask PHI/PII before forwarding; consistent tokenization (Mesh approach) maps sensitive values to stable placeholders so the LLM preserves code/relationship semantics. Network egress: LLM services in dedicated VPC/VNet subnet whitelisting only `api.openai.com`, `api.anthropic.com` etc; block all other outbound. The 2026 incident driver: Vercel supply-chain attack via compromised CI/CD credentials exfiltrated env vars across thousands of customer deployments.

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Enumerate the four secret-management anti-patterns (config files in VCS, hardcoded env, spreadsheets, static keys) and name their replacements.
- Explain the AI-gateway-pulls-from-vault pattern as 2026 production standard.
- Implement a PII scrubber with consistent tokenization (same value → same placeholder) so semantics survive.
- Name the 2026 Vercel supply-chain incident and what it taught about CI/CD credential hygiene.

## The Problem

An intern commits `.env` with API keys. They delete it quickly. The keys are already in git history — GitGuardian scan catches it, your rotation process is "Slack the team, update 40 config files, redeploy all services." 8 hours later, half your services are live and half are waiting for deploy windows.

Separately, user prompts include "My SSN is 123-45-6789." Prompt goes to OpenAI. You have a BAA but your internal policy is to mask PII before forwarding. You didn't.

Separately, your EKS cluster's LLM pod can reach any internet host. Someone exfils data via DNS lookup to an attacker-controlled domain. Nothing blocked it.

Security for LLM services has to address all three vectors. Vault-backed credentials. PII scrubbing. Network egress filtering. Audit logs.

## The Concept

### Centralized vault + IAM-role pull

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. One source of truth.

**IAM role**: app/gateway authenticates via its IAM identity, not a static key. Vault returns the secret for the lifetime of the token.

**The AI-gateway pattern**: gateway pulls `OPENAI_API_KEY` from vault at request time. Rotate in vault; next request gets the new key. No redeploys.

### Rotation policy ≤ 90 days

All API keys, vault root tokens, CI/CD credentials. Automated rotation where possible. Manual rotation logged and tracked.

### Secret scanning

- **TruffleHog** — regex + entropy on commits.
- **GitGuardian** — commercial, high accuracy.
- **Gitleaks** — OSS, runs in CI.

Run on every commit. Block PR if new secret detected.

### Zero-trust posture

- MFA required on all accounts.
- SSO via SAML/OIDC.
- RBAC (role-based) or ABAC (attribute-based) for fine grained access.
- Short-lived tokens (hours, not days).
- Device posture — only corp devices with disk encryption.

### PII / PHI scrubbing

Before the prompt leaves your infra:

1. Entity recognition (spaCy NER, Presidio, commercial).
2. Mask matched entities: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. Consistent tokenization (Mesh approach): same value maps to the same placeholder so the LLM preserves relationships.
4. Optional reverse mapping for LLM response.

Static regex filters catch basic patterns; NER catches more. Use both.

### Input + output guardrails

Input: block known jailbreaks, forbidden topics; rate-limit per-user.

Output: regex scrub for leaked secrets (API key patterns, email patterns in refusal contexts), classifier for policy violations.

### Network egress whitelist

LLM services in a dedicated subnet:
- Whitelist: `api.openai.com`, `api.anthropic.com`, vector DB endpoints, vault endpoints.
- Everything else: drop.
- DNS via allowlist-only resolver (avoid DNS-tunneling exfil).

### Audit log

Immutable log of every LLM call with:
- Timestamp.
- User / tenant.
- Prompt hash (not raw prompt for privacy).
- Model + version.
- Token counts.
- Cost.
- Response hash.
- Any guardrail trips.

Retain per regulatory requirement (SOC 2 1 year, HIPAA 6 years).

### The 2026 Vercel incident

Supply-chain attack: compromised CI/CD credentials exfiltrated env vars across thousands of customer deployments. Lesson: CI/CD credentials are prod-equivalent. Store in vault. Scope narrowly. Rotate aggressively.

### Numbers you should remember

- Rotation policy: ≤ 90 days.
- Scan on every commit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: CI/CD creds compromised → thousands of customer env vars leaked.
- Audit log retention: SOC 2 = 1 year, HIPAA = 6 years.

## Use It

`code/main.py` implements a toy PII scrubber with consistent tokenization and an append-only audit log.

## Ship It

This lesson produces `outputs/skill-llm-security-plan.md`. Given regulatory scope and current state, plans the vault migration, scrubber, egress, audit log.

## Exercises

1. Run `code/main.py`. Send two prompts referencing the same SSN. Confirm both get the same placeholder.
2. Design the network egress policy for a vLLM-on-EKS deployment calling OpenAI + Anthropic + Weaviate.
3. You discover a key in git history (2 years old). What's the correct response — rotate the key, scrub history, or both? Justify.
4. Your audit log grows 10 GB/day. Design retention tiers (hot 30d, warm 12mo, cold 6yr).
5. Argue whether reverse-tokenization (substituting real values back into LLM response) is worth the complexity versus keeping placeholders visible.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Vault | "secrets store" | Centralized credential management service |
| IAM role | "identity-based auth" | Role assumed by app; returns short-lived creds |
| OIDC for CI/CD | "cloud-issued tokens" | No static keys in CI — identity via OIDC |
| TruffleHog / GitGuardian / Gitleaks | "secret scanners" | Commit-time secret detection |
| RBAC / ABAC | "access control" | Role-based vs attribute-based |
| PII scrubbing | "data masking" | Remove or tokenize sensitive entities |
| Consistent tokenization | "stable placeholders" | Same value → same token each time |
| Mesh approach | "Mesh tokenization" | Semantic-preserving tokenization pattern |
| Egress whitelist | "outbound allowlist" | Only permitted domains reachable |
| Audit log | "immutable history" | Append-only record for compliance |

## Further Reading

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/25-security-secrets-audit)

---

## Part 7 (ch384): Compliance — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Multi-framework coverage is table stakes for 2026 enterprise deals. **EU AI Act**: in force since August 1, 2024. Most high-risk requirements enforce August 2, 2026. Fines up to €15M or 3% global annual turnover for high-risk-system obligations (Art. 99(4)); up to €35M or 7% for prohibited AI practices (Art. 99(3)). Applies globally if serving EU users. **Colorado AI Act**: effective June 30, 2026 (delayed from February 2026 by SB25B-004) — impact assessments for high-risk systems, right to appeal AI decisions. Virginia similar for credit/employment/housing/education. **SOC 2 Type II**: de facto B2B AI requirement (Type II, not Type I, for fintech). **GDPR**: largest documented AI-specific fine is €30.5M against Clearview AI (Dutch DPA, Sept 2024); Italy's Garante issued €15M against OpenAI in Dec 2024 (later overturned on appeal in March 2026). Real-time PII redaction at inference is the defensible standard; post-processing cleanup is not enough. **HIPAA**: healthcare bound — cannot send PHI to external AI services without BAA. **PCI-DSS**: AI-interaction-layer coverage requires configuration + contractual agreements, not automatic. **ISO 42001**: emerging AI governance standard, growing procurement requirement alongside ISO 27001. Reference profile: OpenAI maintains SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS for ChatGPT payment components. Cross-framework mapping reduces audit fatigue: access controls map across ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a).

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Enumerate the seven 2026 frameworks relevant to LLM products and match each to a customer segment.
- Cite the EU AI Act enforcement timeline (in force August 2024; high-risk enforcement August 2026) and the two-tier fine ceiling (€15M / 3% for high-risk obligations, €35M / 7% for prohibited practices).
- Explain why post-processing PII cleanup is not enough for GDPR and name real-time inference-layer redaction as the defensible standard.
- Describe cross-framework control mapping (e.g., access control maps to ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)).

## The Problem

An enterprise customer's procurement asks for SOC 2 Type II, GDPR, HIPAA BAA, ISO 27001, and "EU AI Act compliance statement." Your team has SOC 2 Type I. You're six months from Type II and haven't started GDPR Article 30 records.

Multi-framework coverage is not an LLM problem — it's an enterprise-SaaS problem, with LLM-specific overlays. Procurement teams in 2026 want a matrix with a row per framework and a column per control, not a PDF.

## The Concept

### The seven frameworks

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required; PHI cannot leave infrastructure without signed agreement |
| GDPR | EU users | Real-time PII redaction; data subject rights; Article 30 records |
| PCI-DSS | Payment data | Configuration + contracts for AI touching payment |
| EU AI Act | Serving EU users | Risk tier classification; high-risk systems: conformity assessment, documentation, logging |
| Colorado AI Act | Serving CO residents | Impact assessments; right to appeal |
| ISO 42001 | AI governance | Emerging; pairs with ISO 27001 |

### EU AI Act timeline

- August 1, 2024: in force.
- February 2, 2025: prohibited-AI practices enforced.
- August 2, 2026: high-risk systems enforced (conformity assessment, documentation, logging).
- August 2027: high-risk systems in products under harmonized legislation.

Risk tiers: Unacceptable (banned), High-risk (conformity + logging), Limited-risk (transparency), Minimal-risk (no constraint). Most B2B LLM SaaS is limited-risk; high-risk kicks in for employment, credit, education, law enforcement, migration, essential services.

Fines (Article 99): up to €15M or 3% global annual turnover for breaches of high-risk-system obligations (Art. 99(4)); up to €35M or 7% for prohibited AI practices (Art. 99(3)); whichever higher applies.

### GDPR — real-time redaction is the standard

Post-processing cleanup (redact PII after the LLM sees it) is not a defensible posture — the model already saw the data. Real-time inference-layer redaction is the 2026 standard:

- Entity recognition before the LLM call.
- Consistent tokenization (Mesh approach) preserves semantics.
- Store only redacted prompts + consented opt-in raw.

Recent enforcement: €30.5M against Clearview AI (Dutch DPA, Sept 2024) is the largest documented AI-specific GDPR fine to date; €15M against OpenAI (Italy's Garante, Dec 2024) is the largest LLM-specific fine, though it was overturned on appeal in March 2026 and the ruling remains under further review. Post-processing claims have failed at audit.

### HIPAA — BAA is not optional

You cannot send PHI to external AI services without a signed Business Associate Agreement. All three hyperscaler LLM platforms (Bedrock, Azure OpenAI, Vertex) offer BAAs. OpenAI direct API offers BAA. Anthropic direct API offers BAA. Confirm before sending PHI.

### SOC 2 Type II

Type I: controls designed and documented.
Type II: controls operate effectively over 6-12 months.

B2B procurement in 2026 defaults to Type II. Type I is a starter; Type II is the gate.

Common audit drivers: access logs (who saw what), change management (how was it deployed), risk assessments (quarterly), incident response (tested?). Audit log from Phase 17 · 25 is directly reusable.

### Cross-framework mapping

One access control policy satisfies multiple framework controls:

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Compliance tools (Drata, Vanta, Secureframe) automate this mapping. Worth the cost at scale.

### ISO 42001 — emerging

Published late 2023. Growing procurement requirement alongside ISO 27001. Framework for AI governance including risk management, data quality, transparency, human oversight.

### OpenAI's reference profile

OpenAI maintains SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS for ChatGPT payment components. That is roughly the enterprise table stakes in 2026.

### Numbers you should remember

- EU AI Act fines: up to €15M / 3% (high-risk obligations, Art. 99(4)); up to €35M / 7% (prohibited practices, Art. 99(3)).
- EU AI Act high-risk enforcement: August 2, 2026.
- Largest documented AI-specific GDPR fine: €30.5M, Clearview AI (Dutch DPA, Sept 2024).
- Largest LLM-specific GDPR fine: €15M, OpenAI (Italy's Garante, Dec 2024; overturned on appeal March 2026).
- SOC 2 Type II window: 6-12 months of operated controls.
- Colorado AI Act effective date: June 30, 2026 (delayed from February 2026 by SB25B-004).

## Use It

`code/main.py` is a compliance-mapping spreadsheet in Python — given a control, lists frameworks it satisfies.

## Ship It

This lesson produces `outputs/skill-compliance-matrix.md`. Given customer segment and geography, specifies required frameworks and controls.

## Exercises

1. Your first enterprise customer requires SOC 2 Type II, HIPAA BAA, EU AI Act statement. What is the minimum viable compliance posture to win the deal?
2. Classify three hypothetical LLM products under EU AI Act risk tiers. What changes at high-risk?
3. You accidentally sent PHI to a provider without BAA. Walk through the incident response.
4. Argue whether ISO 42001 is "necessary in 2026" for a mid-market AI vendor.
5. Map your LLM audit log fields (Phase 17 · 25) to at least three framework controls.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SOC 2 Type II | "audited controls" | Controls operating over 6-12 months, independently attested |
| HIPAA BAA | "healthcare contract" | Business Associate Agreement; required for PHI |
| GDPR | "EU privacy" | Real-time PII redaction is the defensible 2026 standard |
| EU AI Act | "EU AI rules" | High-risk enforcement August 2026; €15M / 3% (high-risk obligations) — €35M / 7% (prohibited practices) |
| Colorado AI Act | "US AI state law" | June 30, 2026 effective (delayed by SB25B-004); impact assessments |
| ISO 42001 | "AI governance" | Emerging framework for AI risk + transparency |
| ISO 27001 | "security ISMS" | Information Security Management System baseline |
| Conformity assessment | "EU AI doc package" | High-risk requirement: docs, testing, logging |
| Cross-framework mapping | "one control, many frames" | Single policy satisfies multiple framework controls |

## Further Reading

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/26-compliance-frameworks)
