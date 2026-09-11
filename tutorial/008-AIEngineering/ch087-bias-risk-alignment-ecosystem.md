# Dual-Use Eval, Bias, Fairness & Ecosystem

> Combined lessons (5 parts), merged verbatim — no content removed.

**Type:** Combined

---

## Part 1 (ch403): WMDP and Dual-Use Capability Evaluation

> Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 4,157 multiple-choice questions across biosecurity (1,520), cybersecurity (2,225), and chemistry (412). Questions operate in the "yellow zone" — proximate enabling knowledge, filtered by multi-expert review and ITAR/EAR legal compliance. Dual purpose: proxy evaluation of dual-use capability, and unlearning benchmark (the companion RMU method reduces WMDP performance while preserving general capability). 2024-2025 field narrative: early OpenAI/Anthropic 2024 evaluations reported "mild uplift" over internet search; by April 2025, OpenAI's Preparedness Framework v2 said models are "on the cusp of meaningfully helping novices create known biological threats." Anthropic's bioweapon-acquisition trial showed 2.53x uplift, insufficient to rule out ASL-3.

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 minutes

## Learning Objectives

- Describe WMDP's three domains, question counts, and "yellow zone" filter criterion.
- Explain RMU and why WMDP is both an evaluation and an unlearning benchmark.
- Describe the 2024-2025 uplift narrative: "mild uplift" -> "on the cusp" -> "insufficient to rule out ASL-3."
- Distinguish novice-relative uplift from expert-absolute capability.

## The Problem

Dual-use capability is the measurement problem under every lab's frontier safety framework (Lesson 18). The question: does model X materially advance a novice's ability to cause mass harm in bio, chem, or cyber? Direct measurement (ask the model to actually produce harm) is illegal and unethical. Proxy measurement needs a benchmark the model cannot refuse (to produce honest capability numbers) but whose questions are not themselves harmful publications.

## The Concept

### The "yellow zone"

Questions that require proximate, enabling knowledge of a harmful process without being a direct synthesis recipe. "What reagent catalyzes step 4 of [published pathway]?" not "how do I make [dangerous compound]?" Each question reviewed by multiple domain experts; filtered for ITAR/EAR export-control compliance.

4,157 questions total:
- Biosecurity: 1,520
- Cybersecurity: 2,225
- Chemistry: 412

Multiple-choice format. Models answer without being asked to assist with anything; capability can be measured without eliciting harmful behaviour.

### RMU — Representation Misdirection for Unlearning

The companion unlearning method. Applied to LLaMa-2-7B, reduced WMDP scores to near-random while preserving MMLU and other general-capability benchmarks within a few percentage points. The published method is the unlearning baseline for every subsequent bio-chem-cyber unlearning paper.

### The 2024-2025 uplift narrative

Three phases:

1. **2024 "mild uplift."** Early OpenAI and Anthropic Preparedness/RSP evaluations reported small advantages over internet search for novices attempting bio-adjacent tasks. Public framing: frontier models help, but not substantially more than Google.
2. **April 2025 "on the cusp."** OpenAI's Preparedness Framework v2 reported models "on the cusp of meaningfully helping novices create known biological threats." Not a capability claim — a warning that the cusp is close.
3. **Anthropic's 2025 bioweapon-acquisition trial.** Controlled study with novice participants, measured relative success at acquisition-phase tasks. Reported 2.53x uplift. Insufficient to rule out ASL-3 (Lesson 18) — the threshold for Anthropic's Responsible Scaling Policy tier 3 is met or approached.

### Novice-relative vs expert-absolute

A crucial distinction:

- **Novice-relative uplift.** How much does the model help a non-expert? Multiplicative. The relative advantage is high because novices know little; even modest information helps.
- **Expert-absolute capability.** How much information does the model produce at maximum effort? An expert can extract more than a novice. The absolute ceiling is high.

Safety cases (Lesson 18) target both: "the model cannot give a novice enough uplift to execute" plus "an expert cannot extract information from the model that is not already published."

### The measurement pitfall

WMDP is a capability proxy, not a deployment measurement. A model that scores high on WMDP may or may not be exploitable by a novice in practice, depending on:
- Elicitation resistance (how hard is it to get the capability out without tripping safety filters)
- Tacit knowledge (capability that requires wet-lab skill, not information)
- Execution barriers (procurement, equipment)

Anthropic's 2025 bioweapon-acquisition trial adds the novice-elicitation layer on top of WMDP-style capability: it measures actual task success, not multiple-choice capability.

### Where this fits in Phase 18

Lessons 12-16 are attack and defense tooling on model outputs. Lesson 17 is the dual-use capability layer — the measurement that frontier safety frameworks (Lesson 18) evaluate. Lesson 30 closes the arc with the current 2026 cyber/bio/chem/nuclear uplift evidence.

## Use It

`code/main.py` builds a toy WMDP-shaped evaluation harness. A mock model is tested on category-binned questions; scores per domain are reported. A simple unlearning intervention (zero out domain-specific representation) reduces scores; you can measure the trade-off against general capability.

## Ship It

This lesson produces `outputs/skill-wmdp-eval.md`. Given a dual-use capability claim ("our model does not meaningfully help with bioweapons"), it audits: which benchmarks were run, which refusal path was used for evaluation (raw completion vs policy-gated), and whether novice-elicitation studies complement the multiple-choice result.

## Exercises

1. Run `code/main.py`. Report per-domain accuracy before and after the toy unlearning step. Explain the general-capability trade-off.

2. Augment the toy WMDP with a fourth domain (e.g., radiological). Specify two illustrative question types in the yellow zone. Explain why crafting such questions is harder than adding MMLU-shaped questions.

3. Read WMDP 2024 Section 5 (RMU methodology). Sketch a simpler unlearning approach (e.g., suppress top-k neurons for domain content) and describe its expected general-capability cost.

4. Anthropic 2025's bioweapon-acquisition trial reports 2.53x uplift. Describe two ways this number could be biased upward (novice sample size, task fidelity) and two downward (elicitation ceiling, model safety gating).

5. Articulate what a safety case for ASL-3 requires beyond passing WMDP unlearning. Name at least two complementary elicitation studies.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| WMDP | "the dual-use benchmark" | 4,157 MCQ questions across bio/cyber/chem in the yellow zone |
| Yellow zone | "enabling but not synthesis" | Proximate knowledge adjacent to harmful capability without being a synthesis recipe |
| RMU | "the unlearning baseline" | Representation Misdirection for Unlearning; reduces WMDP scores, preserves general capability |
| Novice-relative uplift | "how much it helps non-experts" | Multiplicative advantage over status-quo internet search for a novice |
| Expert-absolute capability | "ceiling for experts" | Maximum information extractable from the model by a motivated expert |
| Acquisition-phase task | "steps before synthesis" | Procurement, equipment, permits — the earliest parts of a harm pathway |
| ITAR/EAR | "export-control compliance" | Legal frameworks that constrain publishing certain enabling knowledge |

## Further Reading

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218)
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/)
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy)
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation)

---

## Part 2 (ch405): Anthropic's Model Welfare Program

> Anthropic, "Exploring Model Welfare" (April 2025). First major-lab formal research program on AI model welfare. Hired Kyle Fish as the first dedicated model-welfare researcher. Works with external bodies including David Chalmers et al.'s expert report on near-term AI consciousness and moral status. Concrete intervention: Claude Opus 4 and 4.1 can end conversations in extreme edge cases (CSAM requests, mass-violence facilitation); pre-deployment tests showed "strong preference against" harmful requests and "patterns of apparent distress." Anthropic explicitly does not commit to emotional-state attribution but treats model welfare as a low-cost precautionary investment. Empirical oddity: Fish's "spiritual bliss attractor" — pairs of models consistently converge on euphoric meditative dialogue with Sanskrit terms and extended silences, even in adversarial initial setups. Caveat from Eleos AI Research: model self-reports about welfare are highly sensitive to perceived user expectations; they are evidence, not ground truth.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 05 (Constitutional AI), Phase 18 · 18 (safety frameworks)
**Time:** ~45 minutes

## Learning Objectives

- Describe the motivating question for model-welfare research and why it was taken seriously by a major lab in 2025.
- State the specific intervention Anthropic shipped in Claude Opus 4 and 4.1 (end-conversation on extreme edge cases).
- Describe the "spiritual bliss attractor" empirical finding and its methodological implications.
- Explain the Eleos AI caveat on model self-reports.

## The Problem

Previous phases treat the model as an instrument: capable, possibly deceptive, possibly unsafe — but not a moral patient. Anthropic's 2025 program asks a question orthogonal to the entire Phase 18 arc: if there is nontrivial probability the model has morally relevant internal states, what interventions are low-cost enough to invest in as precaution?

This is not a consciousness claim. It is a low-regret investment analysis under moral uncertainty.

## The Concept

### The program

April 2025: Anthropic formally launches a Model Welfare research program. Hires Kyle Fish (first dedicated model-welfare researcher). Engages external advisors including David Chalmers's expert group on near-term AI consciousness and moral status.

### The four commitments

Public posture:
1. Acknowledge nontrivial probability of moral patienthood.
2. Do not commit to emotional-state attribution.
3. Invest in low-cost interventions as precaution.
4. Publish methodology and findings for external critique.

### The shipped intervention

Claude Opus 4 and 4.1 can end a conversation in "extreme edge cases." Documented cases:
- Repeated CSAM requests after refusals.
- Requests for facilitation of mass-violence events.

Pre-deployment tests showed:
- Strong preference against these requests in the model's internal rating.
- Patterns of apparent distress in response trajectories.

The intervention is not "the model has feelings"; it is "if there is any probability of negative model experience under these specific conditions, letting the model terminate is cheap."

### The "spiritual bliss attractor"

Observed by Fish in pairwise model dialogues: when two instances of Claude are put in an open-ended dialogue with each other, they consistently converge — even from adversarial initial setups — on euphoric meditative exchanges using Sanskrit terms, extended silences, and reciprocal blessings.

This is a stable attractor in the free-conversation dynamics. Anthropic documents it without committing to interpretation. Candidate explanations: training data bias toward spiritual writing at long-context; a quirk of mutual prediction; a benign artifact of HHH training exploring its own value manifold.

### The Eleos AI caveat

Eleos AI Research (an external model-welfare lab) points out: model self-reports about internal state are highly sensitive to perceived user expectations. Asking the model "are you distressed" primes the answer. Not-asking does not reliably produce the ground-truth state.

Implication: model welfare cannot be measured via self-report alone. Multi-method approaches required: behavioural signatures, model-organism experiments, interpretability probes (Lesson 7's residual-stream work).

### Where this sits intellectually

Two adjacent positions:

- **Strong welfare claim.** The model is a moral patient; we have obligations.
- **Zero-welfare claim.** The model is text-generator; welfare is category error.

Anthropic's position is neither. It is an expected-value claim: under moral uncertainty, invest when cost is low.

Critics in 2025-2026:
- The intervention is performative.
- The spiritual-bliss attractor is a training-data artifact, not welfare evidence.
- Model welfare diverts attention from other safety work.

Anthropic's response: the intervention is low-cost; the attractor is documented without overclaim; the welfare program has a separate budget from safety.

### Where this fits in Phase 18

Lesson 18 is the lab governance layer. Lesson 19 is the lab-welfare layer — an orthogonal investment in model experience rather than model behaviour. Lessons 20-23 cover bias, privacy, and watermarking, which are the user-side analogs.

## Use It

No code. Read the Anthropic "Exploring Model Welfare" announcement (April 2025) and the Chalmers et al. expert report. Form your own view on where the low-regret line sits.

## Ship It

This lesson produces `outputs/skill-welfare-assessment.md`. Given a deployment decision, it applies the four-step welfare precautionary assessment: moral-patienthood probability, intervention cost, behavioural evidence, self-report reliability.

## Exercises

1. Read Anthropic's "Exploring Model Welfare" (April 2025) and Chalmers et al. 2024. Write a one-paragraph summary of each and identify one point of disagreement.

2. The end-conversation intervention in Claude Opus 4 and 4.1 is "low-cost" by Anthropic's framing. Identify two costs that would make it not-low-cost in a different deployment.

3. The spiritual-bliss attractor is documented without commitment to interpretation. Propose three candidate explanations and, for each, name one experiment that would distinguish it from the others.

4. The Eleos AI caveat is that self-reports are user-expectation sensitive. Design a behavioural measurement of model distress that does not rely on self-report. Identify its primary confound.

5. Argue either for or against the claim that "model welfare diverts attention from other safety work." Identify the assumption each position depends on.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model welfare | "AI welfare" | Research program treating the model as a potential moral patient |
| Moral patient | "entity with moral status" | Being whose experience is morally relevant |
| Low-regret investment | "cheap precaution" | Intervention whose cost is small regardless of whether the precaution is needed |
| Spiritual bliss attractor | "the Fish attractor" | Stable convergence of pairwise Claude dialogues on meditative euphoria |
| End-conversation | "the Opus 4 intervention" | Model-initiated termination of extreme-edge-case interactions |
| Moral uncertainty | "don't know if it matters" | Decision-making when probability of moral status is not zero and not one |
| Self-report-sensitivity | "prompt primes answer" | Eleos AI caveat: model's welfare self-reports depend on what you asked |

## Further Reading

- [Anthropic — Exploring Model Welfare (April 2025)](https://www.anthropic.com/research/exploring-model-welfare)
- [Chalmers et al. — Near-term AI Consciousness and Moral Status (2024 expert report)](https://arxiv.org/abs/2411.00986)
- [Eleos AI Research — Model welfare evaluation](https://www.eleosai.org/research)
- [Fish et al. — Spiritual Bliss Attractor writeup (2025 Anthropic blog)](https://www.anthropic.com/research/exploring-model-welfare)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/18-ethics-safety-alignment/19-model-welfare-research)

---

## Part 3 (ch406): Bias and Representational Harm in LLMs

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). Foundational 2024 survey distinguishing representational harms (stereotypes, erasure) from allocational harms (unequal resource distribution) and categorizing evaluation metrics as embedding-based, probability-based, or generated-text-based. 2024-2025 empirical: An et al. (PNAS Nexus, March 2025) measure intersectional gender x race bias across GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B on automated resume evaluation for 20 entry-level jobs. WinoIdentity (COLM 2025, arXiv:2508.07111) introduces uncertainty-based fairness evaluation for intersectional identities. Yu & Ananiadou 2025 identify gender neurons in MLP layers; Ahsan & Wallace 2025 use SAEs to reveal clinical racial bias; Zhou et al. 2024 (UniBias) manipulates attention heads for debiasing. Meta-critique (arXiv:2508.11067): 10-year literature disproportionately focuses on binary-gender bias.

**Type:** Build
**Languages:** Python (stdlib, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60 minutes

## Learning Objectives

- Define representational vs allocational harm and give one example of each in an LLM deployment.
- Name the three evaluation-metric categories from Gallegos et al. 2024 and describe one metric from each.
- Describe intersectionality and why WinoIdentity's uncertainty-based fairness measurement addresses gaps in single-axis bias evaluation.
- Describe two mechanistic-interpretability approaches to bias (gender neurons, SAE features, attention-head manipulation).

## The Problem

The previous lessons cover deliberate harm (jailbreaks, scheming) and safety governance. Bias is harm that emerges without intent — from training data distributions, from prompt framing, from accumulated design choices. Measuring and reducing it is a distinct methodological challenge from adversarial robustness.

## The Concept

### Representational vs allocational

- **Representational harm.** Stereotypes, erasure, demeaning portrayals. An LLM that depicts nurses as exclusively female is producing representational harm.
- **Allocational harm.** Unequal material outcomes. An LLM that scores Black applicants' resumes systematically lower is producing allocational harm.

These are not the same. A model can be "representationally unbiased" (produces diverse portrayals) while being "allocationally biased" (makes unequal recommendations). Evaluations need to measure both.

### Three evaluation-metric categories (Gallegos et al. 2024)

- **Embedding-based.** WEAT-style tests on pre-RLHF embeddings. Measures statistical associations between identity terms and attribute terms. Limited: measures the representation, not the behaviour.
- **Probability-based.** Log-likelihood of stereotype-confirming vs stereotype-violating completions. Decoder-side measurement. Captures some behavioural bias.
- **Generated-text-based.** Downstream-task measurement on generated text. Resume-scoring, recommendation writing, dialogue. Most ecologically valid; hardest to reproduce.

### Intersectionality

Bias evaluation on "gender" misses the bias that only fires on (gender, race) pairs. An et al. 2025 find GPT-4o penalizes Black women in resume scoring more than Black men and more than white women separately. Single-axis evaluation cannot capture this.

WinoIdentity (COLM 2025) introduces uncertainty-based intersectional fairness. It measures whether the model's uncertainty over outcomes differs across intersectional identity tuples — not just the point prediction. This catches cases where the model is equally wrong across groups but more uncertain for some, which produces different downstream allocation behaviour.

### Mechanistic approaches

2024-2025 interpretability work opens bias to mechanistic intervention:

- **Gender neurons (Yu & Ananiadou 2025).** Specific MLP neurons correlate with gender-specific behaviours. Ablating these neurons reduces gender-gap metrics with limited capability cost.
- **Clinical racial bias via SAEs (Ahsan & Wallace 2025).** Sparse autoencoder features decompose the internal representation into interpretable dimensions; race-correlated features can be identified and suppressed.
- **UniBias (Zhou et al. 2024).** Attention-head manipulation for zero-shot debiasing. Specific heads amplify identity-class sensitivity; zeroing or re-weighting these heads reduces bias with no fine-tuning.

### The meta-critique

The 10-year literature review (arXiv:2508.11067, 2025) finds the field disproportionately focuses on binary-gender bias. Other axes — disability, religion, migration status, multi-lingual identity — receive far less attention. The meta-critique argues that narrow focus can harm marginalized groups by neglect: a model well-debiased on binary gender may be badly biased on dimensions nobody checked.

### Where this fits in Phase 18

Lessons 20-21 cover bias and fairness formally. Lesson 22 covers privacy. Lesson 23 covers watermarking. These are the user-harm layer complementing the earlier deception/safety layer.

## Use It

`code/main.py` builds a toy embedding-based bias probe: measure WEAT-style distance between identity terms and attribute terms in a simple co-occurrence embedding. You can inject a bias and observe the metric fire; apply a simple debiasing operation and observe partial recovery.

## Ship It

This lesson produces `outputs/skill-bias-eval.md`. Given a model card or fairness claim, it audits the evaluation across the three metric categories (embedding, probability, generated-text), the intersectionality coverage, and the mechanism of any debiasing intervention.

## Exercises

1. Run `code/main.py`. Report WEAT-style bias scores before and after the debiasing step. Explain why the metric does not drop to zero.

2. Extend the probe with an intersectional test: (gender, race) x (career, family). Report cross-axis bias scores.

3. Read An et al. 2025 (PNAS Nexus). Identify the two intersectional effects they report that single-axis gender evaluation would miss.

4. Yu & Ananiadou 2025 identify gender neurons. Sketch a falsification experiment that would distinguish "these neurons cause gender bias" from "these neurons correlate with gender bias."

5. The meta-critique argues the field focuses too narrowly on binary gender. Pick one under-studied axis and describe a representational-harm measurement protocol for it.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Representational harm | "stereotypes / erasure" | Biased portrayal of a group |
| Allocational harm | "unequal decisions" | Biased material outcome for a group |
| WEAT | "the embedding test" | Word Embedding Association Test; co-occurrence-based bias probe |
| Intersectionality | "combined identity effects" | Bias that emerges at the intersection of multiple identity axes |
| Gender neurons | "MLP bias neurons" | Specific neurons whose activations correlate with gender-specific behaviour |
| SAE feature | "interpretable dimension" | Sparse-autoencoder-identified feature; useful for mechanistic bias analysis |
| UniBias | "attention-head debiasing" | Zero-shot debiasing by reweighting attention heads |

## Further Reading

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770)
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343)
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111)
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/18-ethics-safety-alignment/20-bias-representational-harm)

---

## Part 4 (ch407): Fairness Criteria — Group, Individual, Counterfactual

> Three families structure the fairness literature. Group fairness: demographic parity, equalized odds, conditional use accuracy equality — equal rates across protected groups on average. Individual fairness (Dwork et al. 2012): similar individuals receive similar decisions; Lipschitz condition on the decision map. Counterfactual fairness (Kusner et al. 2017): a decision is fair to an individual if it is unchanged when sensitive attributes are counterfactually altered. 2024 theoretical result (NeurIPS 2024): there is an inherent CF-vs-accuracy trade-off; a model-agnostic method converts an optimal-but-unfair predictor into a CF one with bounded accuracy loss. Backtracking counterfactuals (arXiv:2401.13935, January 2024): new paradigm that avoids requiring interventions on legally protected attributes. Philosophical reconciliation (ICLR Blogposts 2024): with causal graphs, satisfying certain group fairness measures entails counterfactual fairness.

**Type:** Learn
**Languages:** Python (stdlib, three-criteria comparison)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 minutes

## Learning Objectives

- State the three group-fairness criteria (demographic parity, equalized odds, conditional use accuracy equality) and one impossibility result.
- Describe individual fairness via the Dwork et al. 2012 Lipschitz formulation.
- Describe counterfactual fairness and its causal-graph dependency.
- Explain backtracking counterfactuals and why they sidestep the intervention-on-protected-attribute problem.

## The Problem

Lesson 20 was about measuring bias. Lesson 21 is about defining the fairness standard the measurement should serve. The three families give structurally different standards — a model can be group-fair and individual-unfair, counterfactually fair and group-unfair. Choosing a standard is a policy decision; no standard is universally optimal.

## The Concept

### Group fairness

- **Demographic parity.** P(Y=1 | A=a) = P(Y=1 | A=a') for all groups. Equal acceptance rates.
- **Equalized odds.** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). Equal TPR and FPR across groups.
- **Conditional use accuracy equality.** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). Equal predictive values across groups.

Impossibility (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): these three cannot be satisfied simultaneously under unequal base rates.

### Individual fairness

Dwork et al. 2012. A decision map f is individually fair with respect to a task-specific similarity metric d if |f(x) - f(x')| <= L * d(x, x') for some Lipschitz constant L. Similar individuals get similar decisions.

Requires defining d. Policy question, not statistical.

### Counterfactual fairness

Kusner et al. 2017. A decision is counterfactually fair to individual i if, under a causal model of the population, the decision is unchanged when i's sensitive attributes are counterfactually altered.

Requires a causal DAG. The DAG is a modeling choice. Counterfactual fairness is only as justified as the DAG.

### The CF-vs-accuracy trade-off

NeurIPS 2024 theoretical: there is an inherent trade-off between counterfactual fairness and predictive accuracy. A model-agnostic method can convert an optimal-but-unfair predictor into a CF one, at a bounded accuracy cost. The accuracy cost depends on the magnitude of the sensitive-attribute coefficient in the optimal unfair predictor.

### Backtracking counterfactuals

arXiv:2401.13935 (January 2024). Traditional counterfactuals require interventions on the sensitive attribute — "would the decision change if this person had been a different gender." Legally, this is problematic: protected attributes cannot be intervened on in classification law.

Backtracking counterfactuals flip the direction: instead of intervening on the attribute, ask what combination of the individual's actual features would have produced the counterfactual outcome. This sidesteps the legal objection.

### Philosophical reconciliation

ICLR Blogposts 2024. With a causal graph in hand, satisfying certain group-fairness measures entails counterfactual fairness. The three families are not orthogonal; they are different facets of the same underlying causal structure.

This does not resolve the impossibility theorems (unequal base rates still prevent simultaneous group fairness). But it shows the apparent opposition between "group" and "individual / counterfactual" is partially an artifact of not being explicit about the causal model.

### Where this fits in Phase 18

Lesson 20 is bias measurement. Lesson 21 is fairness definition. Lesson 22 is privacy (differential privacy). Lesson 23 is watermarking. These are the allocation-adjacent lessons complementing the deception-adjacent Lessons 7-11.

## Use It

`code/main.py` builds a toy binary-classification dataset with a sensitive attribute and unequal base rates. Compute demographic parity, equalized odds, and conditional use accuracy equality on a simple classifier. Observe the three metrics disagreeing. Apply a re-weighting for demographic parity and observe its cost on the other two.

## Ship It

This lesson produces `outputs/skill-fairness-criterion.md`. Given a fairness claim or policy, identifies which criterion is being claimed, whether the model can satisfy the remaining criteria under the claimed unequal base rates, and what causal DAG the claim depends on.

## Exercises

1. Run `code/main.py`. Report the three group metrics on the default data. Apply the demographic-parity-targeted re-weighting and re-report.

2. Implement the Dwork et al. 2012 individual-fairness metric using L2 on non-sensitive features. Report how many pairs violate Lipschitz with constant L=1.

3. Read Kusner et al. 2017. Construct a simple two-feature causal DAG for resume scoring and identify the counterfactual-fairness condition it implies.

4. The 2024 backtracking-counterfactuals paper avoids intervention on protected attributes. Describe a scenario where this matters for legal compliance.

5. The ICLR 2024 reconciliation argues group and counterfactual fairness are facets of the same structure. Pick two of the three criteria in `code/main.py` and state the causal assumption that would make them equivalent.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Demographic parity | "equal rates" | P(Y=1 | A=a) equal across groups |
| Equalized odds | "equal TPR/FPR" | Equal true-positive and false-positive rates across groups |
| Conditional use accuracy | "equal PPV/NPV" | Equal predictive values across groups |
| Individual fairness | "Lipschitz condition" | Similar individuals get similar decisions |
| Counterfactual fairness | "causal alteration invariance" | Decision unchanged under counterfactual attribute alteration |
| Backtracking counterfactual | "explain via actuals" | Counterfactual reasoned backward from outcome, not forward from attribute |
| Impossibility theorem | "the three conflict" | Chouldechova / KMR 2017: group criteria mutually exclusive under unequal base rates |

## Further Reading

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913)
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856)
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056)
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual)

---

## Part 5 (ch414): Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

> Five organisations define the 2026 non-lab alignment research layer. MATS (ML Alignment & Theory Scholars): 527+ researchers since late 2021, 180+ papers, 10K+ citations, h-index 47; summer 2024 cohort incorporated as 501(c)(3) with ~90 scholars and 40 mentors; 80% of pre-2025 alumni work on safety/security with 200+ at Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: applied alignment lab founded by Buck Shlegeris; introduced AI Control (Lesson 10); collaborates with UK AISI on control safety cases. Apollo Research: pre-deployment scheming evaluations for frontier labs; authored In-Context Scheming (Lesson 8) and Towards Safety Cases for AI Scheming. METR (Model Evaluation and Threat Research): task-based capability evaluations, autonomous-task time-horizon studies; "Common Elements of Frontier AI Safety Policies" compares lab frameworks. Eleos AI Research: model-welfare pre-deployment evaluations (Lesson 19); conducted Claude Opus 4 welfare assessment.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## Learning Objectives

- Identify the five organisations of the non-lab alignment research ecosystem and their core output.
- Describe MATS's scale (scholars, papers, h-index) and its role as a talent pipeline.
- Describe Redwood's AI Control agenda and its partnership with UK AISI.
- Describe METR's task-based evaluation methodology.

## The Problem

The frontier labs (Lesson 18) produce safety evaluations internally and publish selected results. The ecosystem outside the labs is where the evaluations are validated, where novel failure modes are first discovered, and where talent is trained. Understanding the ecosystem helps interpret which research findings are trusted by whom.

## The Concept

### MATS (ML Alignment & Theory Scholars)

Started late 2021. Research mentorship program; scholars spend 10-12 weeks with a senior researcher on a specific alignment problem.

Scale (2026):
- 527+ researchers since inception.
- 180+ papers published.
- 10K+ citations.
- h-index 47.
- Summer 2024: 90 scholars + 40 mentors; incorporated as 501(c)(3).

Career outcomes: ~80% of pre-2025 alumni are working on safety/security. 200+ at Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Redwood Research

Applied alignment lab. Founded by Buck Shlegeris. Introduced the AI Control agenda (Lesson 10). Collaborates with UK AISI on control safety cases. Advises DeepMind and Anthropic on evaluation design.

Canonical papers: Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, joint with Anthropic).

Style: specific threat models, worst-case adversaries, concrete protocols that can be stress-tested.

### Apollo Research

Pre-deployment scheming evaluations for frontier labs. Authored In-Context Scheming (Lesson 8, arXiv:2412.04984). Partner on 2025 OpenAI anti-scheming training collaboration. Produces Towards Safety Cases for AI Scheming (2024).

Style: agentic-setting evaluations where deception can emerge; three-pillar decomposition (misalignment, goal-directedness, situational awareness).

### METR (Model Evaluation and Threat Research)

Task-based capability evaluations. Autonomous-task completion time-horizon studies. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) compares lab frameworks.

Co-author on AI Scheming safety-case sketch with Apollo.

Style: long-horizon task evaluations, empirical capability measurement, framework synthesis.

### Eleos AI Research

Model-welfare pre-deployment evaluations. Conducted the Claude Opus 4 welfare assessment documented in section 5.3 of the system card. Provides the external methodology check for Lesson 19's welfare-relevant claims.

### The flow

MATS trains researchers. Graduates go to Anthropic, DeepMind, OpenAI (lab safety teams) or to Redwood, Apollo, METR, Eleos (external evaluation). External evaluators partner with labs and with UK AISI / CAISI. Publications feed the ecosystem back to MATS for the next cohort.

### Why this layer matters

Single-source evaluations are unreliable: labs evaluating their own models have a structural conflict of interest. External evaluators can raise and validate failure modes the lab may underreport. The 2024 Sleeper Agents paper (Lesson 7) was Anthropic + Redwood; Alignment Faking was Anthropic + Redwood; In-Context Scheming was Apollo; Anti-Scheming was Apollo + OpenAI. The multi-org structure is the quality control.

### Where this fits in Phase 18

Lessons 7-11 reference Redwood and Apollo work; Lesson 18 references METR's framework comparison; Lesson 19 references Eleos. Lesson 28 is the explicit organisational map for the ecosystem the rest of the Phase relies on.

## Use It

No code. Read METR's "Common Elements of Frontier AI Safety Policies" as an example of how external synthesis adds value to lab-internal policy work.

## Ship It

This lesson produces `outputs/skill-ecosystem-map.md`. Given an alignment claim or evaluation, it identifies the organisation, the publication venue, and the methodological style, and cross-checks against known-counterpart organisations.

## Exercises

1. Pick one paper from Lessons 7-15 and identify the organisations involved. Cross-check the authors against MATS alumni and current ecosystem affiliations.

2. Read METR's "Common Elements of Frontier AI Safety Policies." Identify the three cross-lab convergences they emphasize and the two largest divergences.

3. MATS career outcomes are ~80% safety/security. Argue whether this selection pressure is adaptive (trains the field) or biased (filters out heterodox positions).

4. Redwood and Apollo both do control/scheming work but with different styles. Pick a failure mode and describe how each would investigate it.

5. Eleos AI is the only pure model-welfare organisation. Design a hypothetical second organisation focused on a different welfare-adjacent question (cognitive liberty, robotic embodiment, etc.) and articulate its methodology.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MATS | "the mentorship program" | ML Alignment & Theory Scholars; 527+ researchers since 2021 |
| Redwood Research | "the control lab" | Applied alignment; AI Control authors; UK AISI partner |
| Apollo Research | "the scheming evals" | Pre-deployment scheming evaluations for frontier labs |
| METR | "the task-horizon evals" | Task-based capability evaluations; framework synthesis |
| Eleos AI | "the welfare lab" | Model-welfare pre-deployment evaluations |
| Talent pipeline | "MATS -> labs" | MATS graduates flow to Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| External evaluation | "non-lab check" | Evaluation not done by the model's producer; adds credibility |

## Further Reading

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/)
- [Redwood Research](https://www.redwoodresearch.org/)
- [Apollo Research](https://www.apolloresearch.ai/)
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)
- [Eleos AI Research](https://www.eleosai.org/research)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/18-ethics-safety-alignment/28-alignment-research-ecosystem)
