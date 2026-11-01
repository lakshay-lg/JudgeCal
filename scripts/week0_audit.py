"""Week 0 data audit for lmsys/mt_bench_human_judgments.

Answers the Week 0 gate questions in PLAN.md:
  1. How many comparisons have >= 2 independent human votes?
  2. Do the paper's headline numbers (85% / 81% w/o ties; 66% / 63% with ties)
     reproduce from the public data?
  3. Does the alpha = 0.478 re-analysis reproduce, and how sensitive is it to
     tie policy, item grouping and annotator pool?
  4. How large is the "judge vs majority" advantage over "judge vs single human"?
  5. How precise can estimates be with this sample (bootstrap CIs by question)?

Usage:
  python scripts/download_data.py
  python scripts/week0_audit.py            # writes results/week0/audit.json
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "week0"
N_BOOT = 1000
SEED = 0

FLIP = {"model_a": "model_b", "model_b": "model_a", "tie": "tie"}
FLIP_C = {"m1": "m2", "m2": "m1", "tie": "tie"}


# --------------------------------------------------------------------------
# Loading & normalisation
# --------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
    h = pd.read_parquet(RAW / "human.parquet", columns=cols)
    g = pd.read_parquet(RAW / "gpt4_pair.parquet", columns=cols)
    return h, g


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Orient every row so (m1, m2) is alphabetical; label = which model won.

    Human annotators saw pairs in different presentation orders, so the raw
    model_a/model_b columns are NOT consistent within a comparison. Labels are
    re-expressed as 'm1' / 'm2' / 'tie' relative to the canonical order.
    """
    df = df.copy()
    swap = df.model_a > df.model_b
    df["m1"] = np.where(swap, df.model_b, df.model_a)
    df["m2"] = np.where(swap, df.model_a, df.model_b)
    w = df.winner.replace({"tie (inconsistent)": "tie"})
    w = np.where(swap, w.map(FLIP), w)
    df["label"] = pd.Series(w, index=df.index).map({"model_a": "m1", "model_b": "m2", "tie": "tie"})
    df["item"] = list(zip(df.question_id, df.m1, df.m2, df.turn))
    return df


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def kripp_alpha_nominal(units: list[list[str]]) -> float:
    """Krippendorff's alpha (nominal) from a list of per-unit label lists.

    Units with < 2 labels are ignored (they carry no pairable information).
    """
    units = [u for u in units if len(u) >= 2]
    coinc: Counter = Counter()
    for u in units:
        m = len(u)
        for a, b in combinations(range(m), 2):
            coinc[(u[a], u[b])] += 1 / (m - 1)
            coinc[(u[b], u[a])] += 1 / (m - 1)
    n_c: Counter = Counter()
    for (c, _), v in coinc.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    d_o = sum(v for (c, k), v in coinc.items() if c != k) / n
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n * (n - 1))
    return float("nan") if d_e == 0 else 1 - d_o / d_e


def kripp_alpha_ordinal(units: list[list[str]], order: list[str]) -> float:
    """Krippendorff's alpha with ordinal distance over the given label order."""
    units = [u for u in units if len(u) >= 2]
    coinc: Counter = Counter()
    for u in units:
        m = len(u)
        for a, b in combinations(range(m), 2):
            coinc[(u[a], u[b])] += 1 / (m - 1)
            coinc[(u[b], u[a])] += 1 / (m - 1)
    n_c = Counter()
    for (c, _), v in coinc.items():
        n_c[c] += v
    n = sum(n_c.values())
    cats = [c for c in order if n_c[c] > 0]
    idx = {c: i for i, c in enumerate(cats)}

    def delta(c: str, k: str) -> float:
        i, j = sorted((idx[c], idx[k]))
        s = sum(n_c[cats[g]] for g in range(i, j + 1)) - (n_c[cats[i]] + n_c[cats[j]]) / 2
        return s**2

    d_o = sum(v * delta(c, k) for (c, k), v in coinc.items()) / n
    d_e = sum(n_c[c] * n_c[k] * delta(c, k) for c in cats for k in cats) / (n * (n - 1))
    return 1 - d_o / d_e


def pairwise_agreement(units: list[list[str]]) -> float:
    """Paper (Zheng et al. 2023, App. D.3): P(two random distinct humans agree),
    averaged over pairs within each item, then over items."""
    per_item = []
    for u in units:
        if len(u) < 2:
            continue
        pairs = list(combinations(u, 2))
        per_item.append(np.mean([a == b for a, b in pairs]))
    return float(np.mean(per_item))


def pairwise_agreement_pooled(units: list[list[str]]) -> float:
    """Same, but pooling all pairs across items (alternative reading)."""
    agree = total = 0
    for u in units:
        for a, b in combinations(u, 2):
            agree += a == b
            total += 1
    return agree / total


def judge_vs_humans(judge: dict, units: dict) -> float:
    """P(judge agrees with a randomly selected human), averaged over items."""
    per_item = [np.mean([judge[i] == v for v in vs]) for i, vs in units.items() if i in judge and vs]
    return float(np.mean(per_item))


def majority(vs: list[str]) -> list[str]:
    """Return the modal label(s); several if tied (paper: 'even split')."""
    c = Counter(vs)
    top = max(c.values())
    return [k for k, v in c.items() if v == top]


def judge_vs_majority(judge: dict, units: dict, min_votes: int = 1) -> float:
    per_item = []
    for i, vs in units.items():
        if i not in judge or len(vs) < min_votes:
            continue
        maj = majority(vs)
        per_item.append(np.mean([judge[i] == m for m in maj]))
    return float(np.mean(per_item))


def human_vs_loo_majority(units: dict) -> float:
    """Each human vs majority of the *other* humans on the item (needs >= 3 votes
    so 'others' has >= 2). The like-for-like ceiling for a judge-vs-majority number."""
    per_item = []
    for vs in units.values():
        if len(vs) < 3:
            continue
        scores = []
        for k, v in enumerate(vs):
            maj = majority(vs[:k] + vs[k + 1 :])
            scores.append(np.mean([v == m for m in maj]))
        per_item.append(np.mean(scores))
    return float(np.mean(per_item))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    cats = np.union1d(a, b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe))


def judge_vs_single_human_kappa(judge: dict, units: dict, rng: np.random.Generator) -> float:
    """Cohen's kappa between the judge and ONE randomly drawn human per item."""
    a, b = [], []
    for i, vs in units.items():
        if i in judge and vs:
            a.append(judge[i])
            b.append(vs[rng.integers(len(vs))])
    return cohen_kappa(a, b)


# --------------------------------------------------------------------------
# Bootstrap by question (respects grouping; see PLAN.md §2.4)
# --------------------------------------------------------------------------
def bootstrap(fn, units: dict, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    by_q: dict[int, list] = {}
    for item in units:
        by_q.setdefault(item[0], []).append(item)
    qs = list(by_q)
    rng = np.random.default_rng(seed)
    est = fn(units)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(qs, size=len(qs), replace=True)
        boot: dict = {}
        for r, q in enumerate(sample):
            for item in by_q[q]:
                boot[(r,) + item] = units[item]
        stats.append(fn(boot))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return round(est, 4), round(float(lo), 4), round(float(hi), 4)


def fmt(t):
    return {"est": t[0], "ci95": [t[1], t[2]]}


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h_raw, g_raw = load()
    h, g = canonicalise(h_raw), canonicalise(g_raw)
    h["pool"] = np.where(h.judge.str.startswith("author"), "author", "expert")
    report: dict = {}

    # ---- 1. Structure & coverage -------------------------------------------
    dup = h.duplicated(["item", "judge"], keep="first")
    report["structure"] = {
        "human_votes": len(h),
        "gpt4_pair_verdicts": len(g),
        "questions": int(h.question_id.nunique()),
        "model_pairs": int(h[["m1", "m2"]].drop_duplicates().shape[0]),
        "possible_items(q x pair x turn)": 80 * 15 * 2,
        "items_with_any_human_vote": int(h["item"].nunique()),
        "annotators": int(h.judge.nunique()),
        "votes_by_pool": h.pool.value_counts().to_dict(),
        "label_dist_human": h.label.value_counts().to_dict(),
        "label_dist_gpt4_raw": g_raw.winner.value_counts().to_dict(),
        "duplicate_votes_same_judge_same_item": int(dup.sum()),
        "items_with_mixed_presentation_order": int(
            (h.groupby("item").model_a.nunique() > 1).sum()
        ),
    }
    h = h[~dup]

    def units_from(df: pd.DataFrame, drop_ties: bool) -> dict:
        d = df[df.label != "tie"] if drop_ties else df
        return d.groupby("item").label.agg(list).to_dict()

    # ---- 2. Multi-annotator counts (the Week 0 gate) -----------------------
    gate = {}
    for pool_name, pool_df in [("all_humans", h), ("experts_only", h[h.pool == "expert"])]:
        for tie_name, drop in [("with_ties", False), ("no_ties", True)]:
            u = units_from(pool_df, drop)
            counts = Counter(len(v) for v in u.values())
            multi = {k: v for k, v in u.items() if len(v) >= 2}
            gate[f"{pool_name}/{tie_name}"] = {
                "items_with_>=2_votes": len(multi),
                "items_with_>=3_votes": sum(len(v) >= 3 for v in u.values()),
                "questions_covered_by_>=2": len({i[0] for i in multi}),
                "votes_per_item_hist": dict(sorted(counts.items())),
            }
    report["gate_multi_annotator"] = gate

    # ---- 3. Reproduce Zheng et al. Table 5 (expert humans, per turn) -------
    gpt4 = dict(zip(g["item"], g["label"]))
    experts = h[h.pool == "expert"]
    t5 = {}
    for turn in (1, 2):
        e = experts[experts.turn == turn]
        for setup, drop in [("S1_with_ties", False), ("S2_no_ties", True)]:
            u = units_from(e, drop)
            judge = {k: v for k, v in gpt4.items() if k[3] == turn and (not drop or v != "tie")}
            u_j = {k: v for k, v in u.items() if k in judge}
            t5[f"turn{turn}/{setup}"] = {
                "human_human": round(pairwise_agreement(list(u.values())), 4),
                "human_human_pooled_pairs": round(pairwise_agreement_pooled(list(u.values())), 4),
                "human_human_n_pairs": sum(len(v) * (len(v) - 1) // 2 for v in u.values()),
                "gpt4_human": round(judge_vs_humans(judge, u_j), 4),
                "gpt4_human_n_votes": sum(len(v) for v in u_j.values()),
                "gpt4_majority": round(judge_vs_majority(judge, u_j), 4),
            }
    report["zheng2023_table5_repro"] = {
        "paper_values": {
            "turn1": {"S1": {"gpt4_human": 0.66, "human_human": 0.63},
                      "S2": {"gpt4_human": 0.85, "human_human": 0.81}},
            "turn2": {"S1": {"gpt4_human": 0.66, "human_human": 0.67},
                      "S2": {"gpt4_human": 0.85, "human_human": 0.82}},
        },
        "ours": t5,
    }

    # ---- 4. Reproduce alpha = 0.478 and sensitivity ------------------------
    # The public reproduction (gondiam/llm-judge-stats) groups by the RAW
    # (question, model_a, model_b, turn) key, i.e. without merging annotators
    # who saw the pair in the opposite order, and keeps author votes and ties.
    raw_units = (
        h_raw.groupby(["question_id", "model_a", "model_b", "turn"]).winner.agg(list)
    )
    raw_units = [v for v in raw_units if len(v) >= 2]
    order_raw = ["model_a", "tie", "model_b"]
    alpha = {
        "raw_grouping/all/with_ties": {
            "n_items": len(raw_units),
            "nominal": round(kripp_alpha_nominal(raw_units), 4),
            "ordinal": round(kripp_alpha_ordinal(raw_units, order_raw), 4),
        }
    }
    order_c = ["m1", "tie", "m2"]
    for pool_name, pool_df in [("all", h), ("experts", h[h.pool == "expert"])]:
        for tie_name, drop in [("with_ties", False), ("no_ties", True)]:
            u = units_from(pool_df, drop)
            u = {k: v for k, v in u.items() if len(v) >= 2}
            key = f"canonical_grouping/{pool_name}/{tie_name}"
            alpha[key] = {
                "n_items": len(u),
                "nominal": fmt(bootstrap(lambda x: kripp_alpha_nominal(list(x.values())), u)),
                "raw_agreement": fmt(bootstrap(lambda x: pairwise_agreement(list(x.values())), u)),
            }
            if not drop:
                alpha[key]["ordinal"] = round(kripp_alpha_ordinal(list(u.values()), order_c), 4)
    # Orientation sensitivity: chance agreement depends on label marginals, and
    # which model is called "A" is arbitrary. Symmetrising (each unit counted in
    # both orientations) gives an orientation-invariant alpha.
    u_all = {k: v for k, v in units_from(h, False).items() if len(v) >= 2}
    sym = list(u_all.values()) + [[FLIP_C[x] for x in v] for v in u_all.values()]
    alpha["orientation_invariant/all/with_ties"] = {"nominal": round(kripp_alpha_nominal(sym), 4)}
    u_nt = {k: v for k, v in units_from(h, True).items() if len(v) >= 2}
    sym_nt = list(u_nt.values()) + [[FLIP_C[x] for x in v] for v in u_nt.values()]
    alpha["orientation_invariant/all/no_ties"] = {"nominal": round(kripp_alpha_nominal(sym_nt), 4)}
    report["alpha_repro_and_sensitivity"] = alpha

    # ---- 5. Judge vs majority vs single human (PLAN.md §2.2) ---------------
    rng = np.random.default_rng(SEED)
    comp = {}
    for tie_name, drop in [("with_ties", False), ("no_ties", True)]:
        u = units_from(h, drop)
        judge = {k: v for k, v in gpt4.items() if not drop or v != "tie"}
        u3 = {k: v for k, v in u.items() if len(v) >= 3 and k in judge}
        comp[tie_name] = {
            "n_items_>=3_votes": len(u3),
            "gpt4_vs_single_human": round(judge_vs_humans(judge, u3), 4),
            "gpt4_vs_majority": round(judge_vs_majority(judge, u3), 4),
            "human_vs_single_other_human": round(pairwise_agreement(list(u3.values())), 4),
            "human_vs_loo_majority": round(human_vs_loo_majority(u3), 4),
        }
        u1 = {k: v for k, v in u.items() if k in judge}
        kappas = [judge_vs_single_human_kappa(judge, u1, rng) for _ in range(200)]
        comp[tie_name]["gpt4_vs_single_human_cohen_kappa_mean_200_draws"] = round(float(np.mean(kappas)), 4)
    report["judge_vs_majority_vs_single"] = comp

    # ---- 6. S2 selection effect: GPT-4 abstentions ------------------------
    # S2 drops GPT-4's tie / position-inconsistent verdicts, i.e. the judge
    # abstains on the items it finds hardest. How much of the human non-tie
    # signal does it cover, and what would agreement be if each abstention
    # were replaced by a coin flip (forced choice)?
    u_nt_all = units_from(h, True)
    votes = [(i, v) for i, vs in u_nt_all.items() for v in vs if i in gpt4]
    covered = [(i, v) for i, v in votes if gpt4[i] != "tie"]
    acc_cov = np.mean([gpt4[i] == v for i, v in covered])
    cov = len(covered) / len(votes)
    abst = [i for i in {i for i, _ in votes} if gpt4[i] == "tie"]
    report["s2_selection_effect"] = {
        "nontie_human_votes_on_gpt4_items": len(votes),
        "gpt4_coverage_of_those_votes": round(cov, 4),
        "gpt4_agreement_on_covered_votes(vote-level)": round(float(acc_cov), 4),
        "forced_choice_estimate(abstain=coin_flip)": round(float(cov * acc_cov + (1 - cov) * 0.5), 4),
        "human_human_agreement_on_items_gpt4_abstained": round(
            pairwise_agreement([u_nt_all[i] for i in abst if len(u_nt_all[i]) >= 2]), 4
        ),
        "human_human_agreement_on_items_gpt4_committed": round(
            pairwise_agreement([vs for i, vs in u_nt_all.items()
                                if i in gpt4 and gpt4[i] != "tie" and len(vs) >= 2]), 4
        ),
    }

    # ---- 7. GPT-4 position consistency (free preview of Week 3) ------------
    report["gpt4_position_inconsistency_rate"] = round(
        float((g_raw.winner == "tie (inconsistent)").mean()), 4
    )

    (OUT / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
