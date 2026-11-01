# Week 0 Data Audit: `lmsys/mt_bench_human_judgments`

**Status:** gate passed. MT-Bench is usable as the primary dataset.
**Reproduce:** `python scripts/download_data.py && python scripts/week0_audit.py`
(writes every number below to `results/week0/audit.json`; takes about 30 s).

---

## TL;DR

1. **Gate passed.** 960 comparisons have two or more independent human votes
   (699 if ties are dropped), covering all 80 questions. That gives a
   human–human α with a 95% CI of about ±0.05.
2. **Both headline papers reproduce exactly from the public data:**
   - Zheng et al. (2023), Table 5: human–human 63% / 81% and GPT-4–human
     66% / 85% (first turn, with / without ties). Vote and pair counts match
     the paper to the unit.
   - Haldar & Hockenmaier (2025), "Rating Roulette": human–human
     **α = 0.4784** on the same 761 items.
3. **The "0.478 vs 0.827" contrast compares two different tie policies.**
   On identical data:

   | Human–human, all annotators | Raw agreement | Krippendorff α (nominal) |
   |---|---|---|
   | **With ties** (3-class) | 0.671 | **0.485** |
   | **Without ties** (2-class) | **0.828** | 0.653 |

   0.478 is a with-ties number. 0.827 matches the *without-ties* raw agreement
   (0.828) to three decimals. (Rating Roulette describes its 0.827 as accuracy
   against the human majority, so the match may be a coincidence, but the two
   numbers are still not on the same tie policy.) Like for like, chance
   correction costs **about 0.17–0.19** (0.671 → 0.485, or 0.828 → 0.653), not
   0.35. Tie policy moves α by another 0.17. **Both effects are real and of
   similar size, so the write-up must separate them.**
4. **"GPT-4 beats humans, 85% vs 81%" is partly a selection effect.** The
   no-ties setting drops GPT-4's tie and position-inconsistent verdicts, so the
   judge effectively *abstains* on 17% of the human signal. Those items are the
   hard ones: humans agree only 70% on items where GPT-4 abstained, against 85%
   where it committed. **On the items where GPT-4 committed, humans agree 85.0%
   and GPT-4 agrees 85.6%**, essentially the same. The headline gap comes from
   comparing GPT-4 on easy items with humans on all items.
5. **Scoring against the majority vote does inflate judge agreement, though
   only modestly here.** GPT-4 agrees with a single human 63.5% of the time but
   with the majority 68.1% (with ties, items with ≥3 votes). The like-for-like
   human figures are 64.4% and 66.5%.

---

## 1. Dataset structure

| | |
|---|---|
| Human votes | 3,355 (2,668 expert, 687 author) |
| Annotators | 65 |
| Questions × model pairs × turns | 80 × 15 × 2 = 2,400 possible comparisons |
| Comparisons with ≥1 human vote | 1,814 |
| GPT-4 pairwise verdicts | 2,400 (all comparisons, both orders) |
| GPT-4 position-inconsistent verdicts | 15.8% |
| Duplicate votes (same annotator, same comparison) | 1 (dropped) |
| Licence | CC-BY-4.0 |

**Data gotcha:** in **582 comparisons**, different annotators saw the two
models in *opposite presentation orders*, so the raw `model_a` / `model_b` /
`winner` columns are **not consistent within a comparison**. Grouping by the
raw `(question, model_a, model_b, turn)` key, as the public 0.478 reproduction
does, splits these comparisons apart and leaves 761 multi-vote items instead of
960. The audit script puts every vote into a canonical (alphabetical) model
order before grouping. Doing so barely changes α (0.478 → 0.485), so the
published figure holds up, but it recovers 199 usable items.

## 2. The gate: multi-annotator coverage

| Annotator pool / tie policy | Items with ≥2 votes | ≥3 votes | Questions covered |
|---|---|---|---|
| All humans, with ties | **960** | 400 | 80 |
| All humans, no ties | **699** | 254 | 79 |
| Experts only, with ties | 734 | 227 | 80 |
| Experts only, no ties | 539 | 144 | 79 |

Votes per item (all humans, with ties): 1 → 854, 2 → 560, 3 → 271, 4 → 92,
5 → 25, 6 → 10, 7 → 2.

## 3. Reproducing Zheng et al. (2023), Table 5

Expert annotators only (the paper's "Human"), GPT-4 pairwise, with
position-inconsistent verdicts counted as ties.

| | Paper | Ours | Count check (paper / ours) |
|---|---|---|---|
| Turn 1, S1: human–human | 63% | 63.0% | 721 / 721 pairs |
| Turn 1, S1: GPT-4–human | 66% | 66.2% | 1343 / 1343 votes |
| Turn 1, S2: human–human | 81% | 81.0% | 479 / 479 pairs |
| Turn 1, S2: GPT-4–human | 85% | 84.6% | 859 / 859 votes |
| Turn 2, S1: human–human | 67% | 66.6% | 707 / 707 pairs |
| Turn 2, S1: GPT-4–human | 66% | 67.0% | 1325 / 1325 votes |
| Turn 2, S2: human–human | 82% | 81.9% | 474 / 474 pairs |
| Turn 2, S2: GPT-4–human | 85% | 85.4% | 864 / 864 votes |

Human–human agreement reproduces exactly when **all annotator pairs are pooled
across items**. Averaging per item first gives 64.1% / 80.1%. GPT-4–human
agreement is averaged per item and lands within one point. The harness will
implement the pooled definition as "paper-compatible" and say so in the
documentation.

## 4. Reproducing α = 0.478 and how sensitive it is

| Variant | Items | α nominal [95% CI] | α ordinal | Raw agreement [95% CI] |
|---|---|---|---|---|
| Raw grouping, all humans, ties (Rating Roulette setup) | 761 | **0.478** | 0.571 | — |
| Canonical grouping, all humans, ties | 960 | 0.485 [0.435, 0.530] | 0.576 | 0.671 [0.639, 0.701] |
| Canonical grouping, all humans, no ties | 699 | 0.653 [0.595, 0.708] | — | 0.828 [0.799, 0.856] |
| Canonical grouping, experts, ties | 734 | 0.467 [0.418, 0.515] | 0.556 | 0.663 [0.630, 0.696] |
| Canonical grouping, experts, no ties | 539 | 0.628 [0.567, 0.689] | — | 0.815 [0.787, 0.845] |

CIs come from a bootstrap that resamples **questions** (1,000 resamples),
which respects the grouping described in PLAN.md §2.4.

Notes:
- The nominal α reproduces 0.478. Rating Roulette says it uses ordinal
  distance, which gives 0.571 on the same items. Either the paper used nominal
  in practice, or it encoded labels differently. This is worth a footnote, not
  a correction.
- **Label orientation doesn't matter for the human data.** Counting every
  comparison in both orientations gives α = 0.485 / 0.654, the same as above,
  because human labels are nearly balanced between the two positions. It *will*
  matter for any analysis that uses the raw GPT-4 columns: in the raw data
  `model_b` wins 1,610 of 2,400 times, because the dataset puts the stronger
  model in slot B. Always canonicalise first.
- Experts-only α is about 0.02 lower than all-humans. That is well inside the
  CI.

## 5. Judge vs. single human vs. majority (PLAN.md §2.2)

Items with ≥3 human votes, so the "other humans" in the leave-one-out majority
are at least two people.

| | With ties (400 items) | No ties (216 items) |
|---|---|---|
| GPT-4 vs. single human | 0.635 | 0.844 |
| GPT-4 vs. majority | **0.681** | **0.861** |
| Human vs. single other human | 0.644 | 0.829 |
| Human vs. leave-one-out majority | 0.665 | 0.840 |
| Cohen's κ, GPT-4 vs. one random human (mean of 200 draws, all items) | 0.492 | 0.715 |

Scoring against the majority adds 2–5 points for GPT-4. Even the
leave-one-out human figure is slightly unfair *to humans*, because their
majority is built from one fewer vote. The like-for-like comparisons are row 1
against row 3, and row 2 against row 4.

## 6. The S2 selection effect

| | |
|---|---|
| Non-tie human votes on comparisons GPT-4 judged | 2,575 |
| Share of those where GPT-4 also gave a non-tie verdict | **82.7%** |
| GPT-4 agreement on that covered share | 85.6% |
| Human–human agreement on items where GPT-4 **committed** | **85.0%** |
| Human–human agreement on items where GPT-4 **abstained** | **69.9%** |
| Forced-choice lower bound (each abstention = coin flip) | 79.5% |

GPT-4's non-tie verdicts fall disproportionately on easy comparisons. On the
items where it commits, it matches human–human agreement, but it doesn't beat
it. Its true forced-choice agreement lies somewhere between 79.5% (if it
guessed at random on abstentions) and 85.6%. Our own judges will be run in
forced-choice mode as well as tie-allowed mode, so this can be measured
directly rather than bounded.

## 7. Precision / power

- Human–human α (with ties, 960 items): 95% CI half-width **≈ ±0.05**.
  Raw agreement: ≈ ±0.03.
- A locked test split of about 30 of the 80 questions leaves roughly 360
  multi-vote items. Expect α half-widths of **≈ ±0.08** on that split.
  Differences between judges smaller than about 0.1 α won't be resolvable
  on the test split alone, so judge comparisons should use paired bootstraps
  (same resampled questions for both judges), which are tighter.
- **Calibration** can also use the 854 single-vote comparisons (target = that
  one human). Soft-label targets need the 960 multi-vote comparisons.

---

## Decisions this audit proposes (for the pre-registration)

| Decision | Proposal | Reason |
|---|---|---|
| Primary dataset | MT-Bench human judgments | Gate passed |
| Item definition | (question, model pair in canonical order, turn) | Raw grouping silently splits 199 comparisons |
| Annotator pool | **All humans (experts + authors)** as primary; experts-only as sensitivity check. **Accepted.** | +226 multi-vote items; α differs by only 0.02 |
| Tie policy | Report **both**. Primary for the "inflation" story: with ties, since it's the stricter and more realistic setting | The 0.478/0.827 contrast mixes the two policies |
| Human–human agreement definition | Pooled pairs (paper-compatible) as primary; per-item mean reported alongside | Reproduces Zheng et al. exactly |
| Judge modes | Run every judge **forced-choice and tie-allowed** | Measures the S2 selection effect directly |
| CIs | Question-level bootstrap; paired bootstrap for judge-vs-judge differences | Respects the grouping |

## Still open for Week 0

- [ ] API budget and final judge panel (needs your input)
- [ ] Check HelpSteer2-Preference annotator counts and licence (second dataset)
- [ ] Commit `docs/preregistration.md` using the decisions above
- [ ] 50-item pilot to measure local-judge throughput on Kaggle

## References

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena.* NeurIPS Datasets & Benchmarks. arXiv:2306.05685 (Table 5, Table 13,
  Appendix D.3).
- Haldar & Hockenmaier (2025). *Rating Roulette: Self-Inconsistency in
  LLM-As-A-Judge Frameworks.* Findings of EMNLP 2025. arXiv:2510.27106
  (Table 3).
- `gondiam/llm-judge-stats`, `examples/mt_bench.py`: public reproduction of
  α = 0.478, which identified the 761-item raw-grouping setup.
