# JudgeCal — Revised Project Plan

> LLM-as-judge calibration harness: a chance-corrected reliability profile for
> several LLM judges, measured against an apples-to-apples human ceiling, with
> a calibration map that turns a judge's confidence into a number that means
> what it says.

This is the working plan. The original idea is kept, unchanged, in
[`docs/original-idea.md`](docs/original-idea.md). The section
[Changes from the original plan](#changes-from-the-original-plan) records what
changed and why.

**Project type:** portfolio piece. The aim is one sharp, credible finding, a
small reusable tool, and a write-up a hiring manager can skim in five minutes.
It is not trying to cover every known judge bias.

---

## 1. The claim this project tests

The most cited validation of LLM-as-judge (Zheng et al., 2023, *Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena*) reports that GPT-4 agrees
with expert humans about 85% of the time on MT-Bench pairwise comparisons,
against roughly 81% human–human agreement. That pair of numbers is widely read
as "judges are at human level."

Two problems with that reading:

1. **Raw agreement doesn't correct for chance.** On a near-binary decision,
   much of the agreement would happen by chance anyway. A re-analysis reports
   Krippendorff's α ≈ 0.478 for human–human agreement on the same data, against
   raw agreement of 0.827. *(TODO: pin the exact citation, and reproduce this
   number in Week 0 before relying on it.)*
2. **The headline number depends on how ties are handled.** As far as I recall
   the paper, the 85%/81% figures are for the setting *without ties*. With ties
   counted as a third option, both numbers drop sharply (to roughly the
   mid-60s). Verify this against the paper in Week 0. If it holds, part of the
   "inflation" is a tie-handling choice, not only a chance-correction issue,
   and the write-up has to separate the two.

**Research question:** once chance correction, tie policy, and a fair
comparison against a human ceiling are all handled, how reliable are current
LLM judges, and can their confidence be calibrated into something a
practitioner can use?

---

## 2. Core design principles

### 2.1 Always report all three metrics, side by side
Report raw agreement, Cohen's κ, and Krippendorff's α together, with bootstrap
95% CIs, in every table. The gap between them is the finding. Never report
only the flattering number, and never report only the corrected one.

### 2.2 Compare like with like (the most important fix)
Scoring a judge against the **human majority vote** while scoring humans
against **each other** is unfair in the judge's favour. A majority vote is less
noisy than any single annotator, so a judge can appear to "beat" the human
ceiling just because its target is cleaner.

Rules:
- **Primary comparison:** judge vs. a single held-out human, computed in the
  same leave-one-annotator-out way as human vs. human. This is the number that
  goes next to the human ceiling.
- **Secondary comparison:** judge vs. the majority of the *remaining* humans,
  reported separately and labelled clearly. Never put it beside a human–human
  number.

### 2.3 Fix the tie policy before computing anything
Tie handling changes κ and α a lot. It gets decided in Week 0 and written into
the pre-registration:
- **Primary:** two-class (ties dropped), to match the setting the headline
  claim was made in.
- **Secondary:** three-class (A / B / tie), reported in full.
- Judge outputs that pick "tie" in the two-class setting are counted and
  reported, not silently thrown away.

### 2.4 Split by question, not by comparison
MT-Bench has only **80 questions**, and each question appears in many model
pairings. If you split at the comparison level, the same question lands in both
the calibration split and the test split, which leaks information. All splits
are **grouped by question ID**, and the test split is locked and opened once.

### 2.5 Self-consistency sets the judge's own ceiling
Run each judge several times per item (k samples at non-zero temperature, plus
one run at temperature 0, which is what production setups usually use). A
judge that disagrees with itself 20% of the time can't agree with anyone else
more often than that. This splits judge error into:
- **stochastic noise**, which sampling plus aggregation can fix, and
- **systematic bias**, which no amount of resampling fixes.

### 2.6 Every mitigation comes with its cost
Report each mitigation as *agreement gained per unit of added cost*, using
API calls, tokens, dollars and latency. Order-averaging doubles calls, and a
three-judge panel triples them. A +0.02 α gain for 3× cost is reported as
exactly that.

### 2.7 Pre-register, then run
Before any judge is called, commit `docs/preregistration.md` containing the
hypotheses, metrics, tie policy, split procedure, primary vs. secondary
analyses, and the minimum effect size the sample can detect. It costs about
half a day, and it makes "I locked the test split" a credible claim.

### 2.8 Reproducibility is infrastructure, not cleanup
From day one:
- Every API response is cached on disk, keyed by a hash of
  (model ID + pinned version, prompt, parameters, sample index).
- Model versions are pinned and recorded in every results file.
- Judges are forced into structured output (JSON schema or tool call). The
  **parse-failure and refusal rate** is reported as a metric in its own right.
- One command regenerates every table and figure from the cache.

---

## 3. Data

### 3.1 Primary: `lmsys/mt_bench_human_judgments`
- About 3.3K expert pairwise votes over 80 questions and 6 models (GPT-4,
  GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B), plus the original
  GPT-4 judge verdicts.
- **Week 0 gate:** count how many comparisons have **two or more independent
  human votes**. Only those support a human–human ceiling. If that set is too
  small for useful CIs (see the power analysis), switch the primary dataset
  before going any further.

### 3.2 Known validity threats (stated up front in the write-up)
- **Age:** the responses come from 2023-era models, and their failure modes
  differ from today's.
- **Contamination:** MT-Bench, and possibly its judgments, are widely
  published, so modern judges may have seen them in training. Mitigation:
  replicate on a second dataset (stretch goal), and report any difference.

### 3.3 Second dataset (stretch, for generalisation)
Candidate: **HelpSteer2-Preference** (NVIDIA), pairwise preferences with
several annotators per pair plus preference strength. Verify licensing and
annotator counts in Week 0. Even a smaller replication on a few hundred items
turns "on MT-Bench" into "on MT-Bench, and it replicates."

---

## 4. Judge panel

The panel should cover **open-weight local models and API models across at
least two size tiers**, so it can answer the practical question: *can a small
local judge replace an expensive API judge without losing reliability?*

| Slot | Example (to be decided) | Why |
|---|---|---|
| Large API judge | a frontier closed model | the current default in industry |
| Small API judge | a cheap or fast closed model | the cost-sensitive option |
| Large open-weight | a ~70B-class model (quantised, Kaggle or rented GPU) | gives full logprobs for calibration |
| Small open-weight | a ~7–8B-class model | the cheapest option that runs locally |

Open-weight models matter for a second reason: they expose **token logprobs**.
Some API models do not (Anthropic's API, for example), which limits which
confidence signals are available for those models (see §5.4).

**Open decision:** the exact models, once the budget is known (§7).

---

## 5. Experiments

Experiments are tiered so the project ships even if time runs short.

### Core (must ship)

#### 5.1 The ruler: human ceiling
- Build the evaluation set from multi-annotated items only.
- Compute the human–human ceiling (leave-one-annotator-out) under all three
  metrics, both tie policies, with bootstrap CIs (resampling at the question
  level, so the CIs respect the grouping).
- **Reproduce the 0.827 vs. 0.478 gap** and break it down: how much comes
  from chance correction and how much from tie handling.
- Power analysis: given n items, the smallest difference in κ/α between two
  judges that can be detected.

#### 5.2 Baseline agreement and self-consistency
- Every judge, every item, once at temperature 0, giving the agreement table
  with the human ceiling as a reference row.
- k samples per item at temperature > 0, giving self-agreement (Fleiss' κ or α
  across samples) and the stochastic/systematic breakdown.

#### 5.3 Position bias
- Every item is shown to every judge in both orders (a paired, within-item
  design).
- Report the flip rate, the direction of any asymmetry (does the judge prefer
  slot A?), and a McNemar test.
- Mitigation measured right here: **order-averaging** (only count a verdict
  when both orders agree, or average the two), with its 2× cost.

#### 5.4 Calibration (the main deliverable)
- **Confidence signals**, compared head to head:
  1. self-consistency vote share across the k samples (works for every judge),
  2. token logprob of the verdict (open-weight judges, and APIs that expose it),
  3. verbalised confidence ("how sure are you, 0–100?"). Expected to be poorly
     calibrated, and included to show it.
- **Target:** the soft human label (the share of annotators who preferred A),
  with the human majority as a secondary target.
- **Methods:** isotonic regression and Platt scaling, fit on the calibration
  split only.
- **Evaluation:** Brier score, ECE (adaptive binning, since n is small), and
  reliability diagrams, before and after calibration, on the **locked test
  split, opened once**.
- **Practitioner output:** a per-judge lookup table that maps stated
  confidence to expected agreement with a human.

### Stretch (only after the core is done)

#### 5.5 Verbosity bias, run two ways
- **Causal:** pad one response with content that adds no information, then
  measure how often the verdict shifts. Use **at least three padding styles**
  (restating the answer, extra formatting or headers, harmless elaboration),
  because judges may *penalise* obvious filler, which would hide a real
  verbosity bias.
- **Observational:** regress the judge's preference on the length difference,
  controlling for human preference.
- The result counts as credible only if the two designs agree in direction.

#### 5.6 Second-dataset replication (§3.3)

#### 5.7 Panel and rubric mitigations
Rubric-anchored prompting, and a majority vote of three judges, each reported
as a cost-benefit point on the same chart as order-averaging.

### Dropped
- **Self-preference.** MT-Bench responses come from 2023 models (GPT-4,
  GPT-3.5, Claude-v1), not from the judges being tested today, so "same model
  family" is too weak a link for a credible result, and the sample would be
  tiny. It gets one paragraph in the write-up explaining why it was excluded,
  which also shows judgement.

---

## 6. Timeline (6 weeks)

| Week | Focus | Exit criteria |
|---|---|---|
| **0** | Data audit, sanity checks, budget, pre-registration | Multi-annotator count known; 85/81 and tie numbers checked against the paper; 0.478 reproduced or explained; judges chosen; preregistration committed |
| **1** | Harness skeleton + the ruler (§5.1) | Caching, structured output and the metrics module tested; human-ceiling table with CIs; question-grouped splits frozen |
| **2** | Baseline + self-consistency (§5.2) | Agreement table for all judges; stochastic/systematic breakdown |
| **3** | Position bias + order-averaging (§5.3) | Flip-rate table; first point on the cost-benefit chart |
| **4** | Calibration (§5.4) | Calibrators fit; test split opened once; reliability diagrams before and after |
| **5** | Stretch: verbosity and/or replication (§5.5–5.7) | Whatever fits; nothing in the core depends on it |
| **6** | Write-up, figures, README, demo | Lead figure, blog-style write-up, `judgecal` runs end to end from a clean clone |

**Risk buffer:** judge-output parsing (hedging, refusals, malformed verdicts)
is the most likely time sink. Structured output reduces it but won't remove
it. If the core slips, Week 5 is the buffer. Stretch work is cut before any
core work is.

---

## 7. Budget

API cost scales as

```
calls = items × judges × orders(2) × (1 + k samples) × conditions
```

For example, 1,000 comparisons × 2 orders × (1 + 5) samples = **12,000 calls
per API judge** for the core experiments alone. At about 1.5K input + 200
output tokens per call, that is roughly 18M input / 2.4M output tokens per
judge.

Week 0 tasks:
- Plug in current prices for the chosen API judges and set a hard budget cap.
- If it's over budget: lower k (3 is often enough to estimate vote share),
  use the cheaper model for self-consistency, or reduce the item count to what
  the power analysis says is needed.
- Measure local-judge throughput on Kaggle GPUs on a 50-item pilot before
  committing to the full run.

**Open decision:** the API budget, and which GPU access is available beyond
Kaggle.

---

## 8. Deliverables

1. **Lead figure: the agreement gap.** For each judge and for humans: raw
   agreement, then κ, then α, with the human ceiling as a band. One image that
   tells the whole story.
2. **Reliability diagrams** before and after calibration, one panel per judge.
3. **Cost-benefit chart:** Δα against added cost for each mitigation.
4. **Write-up** (`docs/writeup.md`, also suitable as a blog post): the finding
   up front, method second, limitations stated honestly.
5. **The `judgecal` harness:** a small CLI and library that produces the
   reliability profile for *any* judge plus any set of multi-annotated human
   labels:
   ```bash
   judgecal run --judge <model> --labels data/labels.jsonl --out results/
   judgecal report results/
   ```
   This turns the project from a one-off analysis into a tool. Scope it as thin
   as possible; it should be a by-product of the analysis code, not a separate
   project.

### Proposed repo layout

```
judgecal/
  data/          # loaders, multi-annotator filtering, grouped splits
  judges/        # API + local judge adapters, structured output, response cache
  metrics/       # raw agreement, κ, α, bootstrap CIs, leave-one-out ceilings
  calibration/   # confidence signals, isotonic/Platt, ECE/Brier, reliability plots
  experiments/   # one module per experiment, config-driven
  cli.py
configs/         # judge panel, prompts, k, tie policy — all versioned
docs/            # preregistration, writeup, original idea
results/         # generated tables/figures (regenerable from cache)
tests/           # metric correctness tests against known values
```

---

## 9. Changes from the original plan

| # | Change | Why |
|---|---|---|
| 1 | Judge vs. single held-out human is the primary comparison; judge vs. majority is secondary and kept separate | Comparing against the majority favours the judge and can create fake "above-human" results |
| 2 | Tie policy fixed up front; both two-class and three-class reported | Tie handling may explain a large part of the headline gap |
| 3 | Reproduce the 0.478 figure and pin its citation in Week 0 | The project's framing rests on it |
| 4 | Splits grouped by question ID | Only 80 questions, so comparison-level splits leak into the test set |
| 5 | Confidence signals defined explicitly (vote share / logprob / verbalised), soft human labels as the target | "Stated confidence" was undefined, and verbalised confidence is known to be poor |
| 6 | Contamination and dataset age named as threats; second-dataset replication added as stretch | Generalisation from a single 2023 dataset is weak |
| 7 | Self-preference experiment dropped | Today's judges aren't the models that wrote the responses, so the result would be confounded |
| 8 | Verbosity padding uses three or more styles | A single "zero-information" padding may be penalised, which would hide the bias |
| 9 | Core and stretch tiers; position bias and order-averaging merged into one week | Six weeks wasn't enough for the original scope |
| 10 | Pre-registration added | Cheap, and makes the locked-split discipline credible |
| 11 | Caching, version pinning, structured output and parse-failure metric from day one | Reproducibility, and the original plan named parsing as the biggest time risk |
| 12 | Reusable `judgecal` CLI as a deliverable | A tool is a stronger portfolio artifact than a one-off analysis |
| 13 | Personal narrative moved out of the technical plan (see below) | Keeps the technical document focused |

---

## 10. Open decisions

- [ ] API budget and GPU access (§7)
- [ ] Final judge panel (§4)
- [ ] Primary dataset confirmed after the multi-annotator count (§3.1)
- [ ] Second dataset chosen and licence checked (§3.3)
- [ ] Exact citation for the α = 0.478 re-analysis (§1)

---

## Appendix: Portfolio framing (for README / applications, not the write-up)

The author has two years of paid professional preference annotation and
adversarial red-teaming at DataAnnotation and Outlier AI. That gives first-hand
intuition for where human labels come from, why annotators disagree, and what
separates a genuinely ambiguous comparison from a badly specified one. Use it
concretely in the write-up, for example in the qualitative analysis of the
items where humans disagree most. Don't use it as a general credential claim.
