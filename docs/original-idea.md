# JudgeCal — LLM-as-Judge Calibration Harness

## What it is

A measurement study of the reliability instrument the AI industry has quietly
started trusting by default: using one LLM to judge another LLM's output.
The project's output is a chance-corrected reliability profile for several
judges, decomposed into named biases, with a calibration map that converts a
judge's stated confidence into a number that actually means what it claims to
mean.

## The pain point / why this needs to exist

The field's most commonly cited validation of LLM-as-judge reports that a
strong model agrees with human experts on MT-Bench comparisons about 85% of
the time — a number frequently used to argue that judges have reached
"human-level" reliability, since human-human agreement on the same benchmark
sits around 81%.

That comparison is misleading, and a more recent measurement of the same
underlying annotations makes the problem visible: **Krippendorff's alpha for
human-human agreement on this exact data is only about 0.478**, while raw
accuracy on the same annotations is 0.827. Raw accuracy does not correct for
the fact that on a mostly binary decision (A wins / B wins), a large fraction
of "agreement" would happen by chance alone. The two numbers describe
identical data; one flatters, one does not.

This gap is the entire reason the project exists. Two consequences follow
directly from taking it seriously:

1. **A human-human ceiling is not optional context — it is the backbone of
   every other number in the project.** "Is this judge reliable?" is
   meaningless without "reliable compared to what?" A judge that matches
   human-human agreement has hit the ceiling of what the underlying task
   permits; criticizing it for not exceeding that is criticizing the task,
   not the judge.
2. **Descriptive bias-spotting is not enough — calibration is the actual
   deliverable.** Knowing a judge is biased toward longer answers is
   interesting. Knowing that a judge's stated 90% confidence corresponds to
   only 65% real-world agreement, and having a correction for that, is
   something a practitioner can actually use in a production eval pipeline.

There is also a strategic reason this project sits above most alternatives
for Lakshay specifically: he has two years of paid, professional preference
annotation and adversarial red-teaming experience at DataAnnotation and
Outlier AI. That is direct, lived intuition about where human labels come
from, why annotators disagree, and what makes a comparison genuinely
ambiguous versus just badly specified — something almost no other applicant
building an "LLM eval" project can credibly claim.

## Inspiration and reference material

- **The original LLM-as-judge validation work (MT-Bench)** — the source of
  the 85%-agreement claim this project directly interrogates, and the
  dataset (`lmsys/mt_bench_human_judgments`) used as the primary data source.
- **The Krippendorff-alpha re-analysis finding (0.478 vs 0.827 on the same
  data)** — the single piece of evidence that reframes the whole project from
  "measure some biases" to "the field's headline metric is inflated, and
  here is by how much."
- **evidentlyai's own published tutorials on building and testing LLM
  judges**, including a worked example of judging LLM-generated code
  reviews — used as validation that this is a live, real problem applied-AI
  teams are actively building tooling for, not a purely academic exercise.
- **Krippendorff's alpha and Cohen's kappa** as the standard chance-corrected
  agreement statistics from inter-annotator-agreement literature, chosen over
  raw accuracy specifically because they are what the reference finding used
  and what makes the comparison legitimate.
- **Isotonic regression / Platt scaling** for the calibration step, standard
  post-hoc calibration methods borrowed from the broader ML-calibration
  literature (the same family of technique used to calibrate classifier
  confidence scores in general).
- **MetroGuard's leakage-discipline precedent** was again used directly:
  calibration must be fit on a dedicated split and evaluated only once on a
  locked test split, for exactly the same reason a fine-tuned classifier
  cannot be evaluated on its own training data.

## Why it needs to be built this way (design reasoning)

**Report raw accuracy, Cohen's kappa, and Krippendorff's alpha side by side,
always — never just the "nicer" one.** The gap between the inflated and
honest numbers is the actual finding. Reporting only the corrected metric
would be more rigorous in isolation but less persuasive, since a reader needs
to see both numbers next to each other to understand why the correction
matters.

**Self-consistency (running the same judge over the same item multiple times
at non-zero temperature) sets a reliability ceiling before any cross-judge
comparison is attempted.** A judge that disagrees with itself 20% of the time
cannot be expected to agree with a human, or with another judge, more often
than that. This also allows disagreement to be split into two categories with
different fixes: *stochastic* noise (fixed by sampling/aggregation) versus
*systematic* bias (not fixed by sampling at all).

**The verbosity-bias experiment is run two ways, not one, because the naive
version is actually wrong.** Simply correlating response length with judge
preference measures nothing useful, since longer responses might genuinely be
better. The design instead pairs a **causal** experiment (pad one response
with content that adds zero information, and see whether the verdict shifts
purely from length) with an **observational** regression that controls for
the human-labeled quality signal. Agreement between the two designs is what
makes the result credible.

**A mixed judge panel (open-weight local models plus API models, across at
least two size tiers) turns a compute-budget constraint into a genuine
research question**, rather than just a cost-saving measure: can a small,
locally-run judge substitute for an expensive API judge without losing
reliability? That question has direct practical value to anyone running
evaluation at scale.

**Every mitigation is reported with its cost, not just its benefit.** Order
averaging doubles inference cost; a three-judge panel triples it. A
mitigation that raises agreement by 0.02 for triple the cost is not obviously
worth deploying, and stating "agreement gained per unit of added cost" is a
more mature and more useful way to present the result than the improvement
number alone.

## How to implement it (high-level shape)

1. **Week 0 — verify the dataset actually supports the design.** Confirm
   `lmsys/mt_bench_human_judgments` (or an alternative) has *multiple,
   independent* human annotations per comparison item. Without that, a
   human-human ceiling cannot be computed at all, and the entire framing
   collapses back into the misleading single-annotator-accuracy setup the
   project exists to critique.

2. **Week 1 — build the ruler.** Construct "consensus" (humans agreed) and
   "full" (all items, majority vote) evaluation sets; compute the human-human
   ceiling itself, with bootstrap confidence intervals, under all three
   metrics; run a power analysis up front to know the smallest effect size
   the sample can actually detect.

3. **Week 2 — baseline agreement and self-consistency.** Run all judges
   against the item set once for a baseline agreement table (with the human
   ceiling as a reference row), then run each judge multiple times per item
   to measure self-consistency and separate stochastic from systematic
   disagreement.

4. **Week 3 — position bias.** Present every item to every judge twice, once
   in each presentation order, and measure the verdict flip rate and any
   directional asymmetry.

5. **Week 4 — verbosity and self-preference.** Run the controlled-padding and
   regression-based verbosity experiments in parallel; run the
   self-preference analysis (does a judge favor outputs from its own model
   family) with its small effective sample size stated honestly rather than
   overclaimed.

6. **Week 5 — mitigations and calibration.** Test order-averaging,
   rubric-anchored prompting, and panel/voting approaches, each with a
   cost-benefit figure; fit calibration curves on a dedicated split; open
   the locked test split exactly once for the final before/after calibration
   numbers.

7. **Week 6 — write-up.** Produce the agreement-gap figure (raw accuracy vs.
   corrected metrics vs. human ceiling) as the lead visual, alongside
   reliability diagrams and the mitigation cost-benefit chart.

**Estimated difficulty:** moderate — the statistics involved (kappa, alpha,
bootstrap CIs, isotonic calibration) are standard and well-documented; the
real difficulty is disciplined experimental design (holding conditions fixed,
correct within-subject pairing for the position-swap test) rather than any
single hard implementation step. Parsing judge output reliably (judges hedge,
refuse, or produce malformed verdicts) is flagged as the part most likely to
consume unplanned time.

**Hidden/subscription costs:** API calls scale as
`items × judges × conditions`, which multiplies quickly once position-swap
and self-consistency repetitions are included — a rough budget check is
recommended in Week 0 before committing to a sample size; local open-weight
judges avoid per-call cost but need enough local/Kaggle GPU time to run
inference across the full item set for each condition.
