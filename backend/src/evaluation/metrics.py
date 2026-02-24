"""
Stage 8 — Evaluation

Rigorous, multi-dimensional evaluation against the gold set.

Metrics computed:
  • Precision / Recall / F1 for span extraction (exact and partial match)
  • Cohen's Kappa for categorical fields (hazard_type, impact_domain, …)
  • Exact match + year-match for location and time fields
  • Hallucination confusion matrix:
        correct | missed | hallucinated | wrong_label
    per field, answering four different research questions.
  • Intra-annotator Kappa (two annotation rounds)
  • Inter-annotator Kappa (you vs. second annotator)

Usage:
    python -m src.evaluation.metrics \
        --gold  data/annotated/gold_set.json \
        --pred  data/weak_labels/labeled_abstracts.json \
        --out   outputs/evaluation_report.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)


# ── Span metrics ──────────────────────────────────────────────────────────────

def span_f1_exact(
    gold_spans: list[tuple],
    pred_spans: list[tuple],
) -> dict[str, float]:
    """
    Exact-match span F1.

    Each span is a tuple (start_char, end_char, entity_type).
    """
    g = set(gold_spans)
    p = set(pred_spans)
    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def span_f1_partial(
    gold_spans: list[tuple],
    pred_spans: list[tuple],
) -> dict[str, float]:
    """
    Partial-overlap span F1.

    Two spans match if their entity type is identical AND their char ranges overlap.
    """

    def overlaps(a: tuple, b: tuple) -> bool:
        return a[2] == b[2] and a[0] < b[1] and b[0] < a[1]

    tp = sum(1 for g in gold_spans if any(overlaps(g, p) for p in pred_spans))
    fp = sum(1 for p in pred_spans if not any(overlaps(p, g) for g in gold_spans))
    fn = sum(1 for g in gold_spans if not any(overlaps(g, p) for p in pred_spans))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Categorical field metrics ─────────────────────────────────────────────────

def field_kappa(gold: list, pred: list) -> float:
    """Cohen's Kappa for a categorical field; skips None gold values."""
    pairs = [(g, p) for g, p in zip(gold, pred) if g is not None]
    if len(pairs) < 2:
        return float("nan")
    g_list, p_list = zip(*pairs)
    try:
        return float(cohen_kappa_score(g_list, p_list))
    except ValueError:
        return float("nan")


def field_f1(
    gold: list,
    pred: list,
    average: str = "macro",
) -> dict[str, float]:
    """Precision / Recall / F1 for a categorical field; skips None pairs."""
    pairs = [(g, p) for g, p in zip(gold, pred) if g is not None and p is not None]
    if not pairs:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}
    g_list, p_list = zip(*pairs)
    return {
        "f1": float(f1_score(g_list, p_list, average=average, zero_division=0)),
        "precision": float(precision_score(g_list, p_list, average=average, zero_division=0)),
        "recall": float(recall_score(g_list, p_list, average=average, zero_division=0)),
    }


# ── Location and time match ───────────────────────────────────────────────────

def location_match(gold: Optional[str], pred: Optional[str]) -> dict[str, bool]:
    if gold is None and pred is None:
        return {"exact": True, "partial": True}
    if gold is None or pred is None:
        return {"exact": False, "partial": False}
    g, p = gold.strip().lower(), pred.strip().lower()
    exact = g == p
    partial = g in p or p in g
    return {"exact": exact, "partial": partial}


def time_match(gold: Optional[str], pred: Optional[str]) -> dict[str, bool]:
    """Match ISO 8601 intervals; also check year-level agreement."""
    if gold is None and pred is None:
        return {"exact": True, "year_match": True}
    if gold is None or pred is None:
        return {"exact": False, "year_match": False}
    exact = gold.strip() == pred.strip()
    gold_year = gold[:4] if len(gold) >= 4 else gold
    pred_year = pred[:4] if len(pred) >= 4 else pred
    return {"exact": exact, "year_match": gold_year == pred_year}


# ── Hallucination confusion matrix ────────────────────────────────────────────

HALLUCINATION_CATS = ["correct", "missed", "hallucinated", "wrong_label"]

HALLUCINATION_FIELDS = [
    "hazard_type",
    "impact_domain",
    "causal_relation",
    "uncertainty_level",
]


def hallucination_matrix(
    gold_records: list[dict],
    pred_records: list[dict],
    fields: Optional[list[str]] = None,
) -> dict[str, dict[str, int]]:
    """
    Produce a per-field hallucination confusion matrix.

    Categories:
      correct       — both match (or both null)
      missed        — gold has value, prediction is null
      hallucinated  — gold is null, prediction has value
      wrong_label   — both non-null but differ
    """
    if fields is None:
        fields = HALLUCINATION_FIELDS

    counts: dict[str, dict[str, int]] = {f: defaultdict(int) for f in fields}

    gold_by_id = {r["abstract_id"]: r for r in gold_records}
    pred_by_id = {r["abstract_id"]: r for r in pred_records}
    common = set(gold_by_id) & set(pred_by_id)

    for abs_id in common:
        g = gold_by_id[abs_id]
        p = pred_by_id[abs_id]
        for field in fields:
            gv = _scalar_value(g.get(field))
            pv = _scalar_value(p.get(field))
            if gv and pv:
                counts[field]["correct" if gv == pv else "wrong_label"] += 1
            elif gv and not pv:
                counts[field]["missed"] += 1
            elif not gv and pv:
                counts[field]["hallucinated"] += 1
            else:
                counts[field]["correct"] += 1  # both null

    return {f: dict(d) for f, d in counts.items()}


def _scalar_value(val) -> Optional[str]:
    """Flatten nested dicts (e.g. causal_relation) to a scalar string for comparison."""
    if val is None:
        return None
    if isinstance(val, dict):
        # Use the 'predicate' for causal_relation, 'normalized' for location/time
        for key in ("predicate", "normalized", "raw"):
            if val.get(key):
                return str(val[key]).lower().strip()
        return None
    return str(val).lower().strip() or None


# ── Intra- and inter-annotator agreement ─────────────────────────────────────

def intra_annotator_kappa(
    round1: list[dict],
    round2: list[dict],
    fields: Optional[list[str]] = None,
) -> dict[str, float]:
    """Cohen's Kappa between two annotation rounds by the same annotator."""
    if fields is None:
        fields = ["hazard_type", "impact_domain", "impact_type", "uncertainty_level"]
    result: dict[str, float] = {}
    for field in fields:
        v1 = [r.get(field) for r in round1]
        v2 = [r.get(field) for r in round2]
        result[field] = field_kappa(v1, v2)
    return result


def inter_annotator_kappa(
    annotator_a: list[dict],
    annotator_b: list[dict],
    fields: Optional[list[str]] = None,
) -> dict[str, float]:
    """Cohen's Kappa between two different annotators (matched by abstract_id)."""
    if fields is None:
        fields = ["hazard_type", "impact_domain", "impact_type", "uncertainty_level"]

    a_by_id = {r["abstract_id"]: r for r in annotator_a}
    b_by_id = {r["abstract_id"]: r for r in annotator_b}
    common = sorted(set(a_by_id) & set(b_by_id))

    result: dict[str, float] = {}
    for field in fields:
        va = [a_by_id[i].get(field) for i in common]
        vb = [b_by_id[i].get(field) for i in common]
        result[field] = field_kappa(va, vb)
    return result


# ── Full evaluation report ────────────────────────────────────────────────────

CATEGORICAL_FIELDS = ["hazard_type", "impact_domain", "impact_type", "uncertainty_level"]


def full_report(
    gold_records: list[dict],
    pred_records: list[dict],
) -> dict:
    """
    Run all evaluation metrics and return a nested report dict.
    """
    report: dict = {}

    # Align by abstract_id
    gold_by_id = {r["abstract_id"]: r for r in gold_records}
    pred_by_id = {r["abstract_id"]: r for r in pred_records}
    common = sorted(set(gold_by_id) & set(pred_by_id))

    if not common:
        return {"error": "No matching abstract IDs between gold and pred sets"}

    g_aligned = [gold_by_id[i] for i in common]
    p_aligned = [pred_by_id[i] for i in common]

    # 1. Per-field categorical metrics
    report["categorical"] = {}
    for field in CATEGORICAL_FIELDS:
        gv = [r.get(field) for r in g_aligned]
        pv = [r.get(field) for r in p_aligned]
        report["categorical"][field] = {
            "f1": field_f1(gv, pv),
            "kappa": field_kappa(gv, pv),
        }

    # 2. Location metrics
    g_locs = [(r.get("location") or {}).get("normalized") for r in g_aligned]
    p_locs = [(r.get("location") or {}).get("normalized") for r in p_aligned]
    loc_results = [location_match(g, p) for g, p in zip(g_locs, p_locs)]
    report["location"] = {
        "exact_match": float(np.mean([r["exact"] for r in loc_results])),
        "partial_match": float(np.mean([r["partial"] for r in loc_results])),
    }

    # 3. Time metrics
    g_times = [(r.get("time_period") or {}).get("normalized") for r in g_aligned]
    p_times = [(r.get("time_period") or {}).get("normalized") for r in p_aligned]
    time_results = [time_match(g, p) for g, p in zip(g_times, p_times)]
    report["time_period"] = {
        "exact_match": float(np.mean([r["exact"] for r in time_results])),
        "year_match": float(np.mean([r["year_match"] for r in time_results])),
    }

    # 4. Hallucination matrix
    report["hallucination_matrix"] = hallucination_matrix(g_aligned, p_aligned)

    # 5. Summary counts
    report["summary"] = {
        "n_gold": len(gold_records),
        "n_pred": len(pred_records),
        "n_matched": len(common),
    }

    return report


def save_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved → {path}")


def print_summary(report: dict) -> None:
    print("\n=== Indra Evaluation Summary ===")
    summary = report.get("summary", {})
    print(f"Gold: {summary.get('n_gold')}  |  Pred: {summary.get('n_pred')}  |  Matched: {summary.get('n_matched')}")

    print("\n── Categorical fields ──")
    for field, metrics in report.get("categorical", {}).items():
        f1 = metrics["f1"].get("f1", float("nan"))
        kappa = metrics.get("kappa", float("nan"))
        print(f"  {field:<22}  F1={f1:.3f}  κ={kappa:.3f}")

    print("\n── Location ──")
    loc = report.get("location", {})
    print(f"  exact={loc.get('exact_match', 0):.3f}  partial={loc.get('partial_match', 0):.3f}")

    print("\n── Time ──")
    t = report.get("time_period", {})
    print(f"  exact={t.get('exact_match', 0):.3f}  year_match={t.get('year_match', 0):.3f}")

    print("\n── Hallucination matrix ──")
    hm = report.get("hallucination_matrix", {})
    header = f"  {'field':<22}  {'correct':>8}  {'missed':>8}  {'hallucinated':>12}  {'wrong_label':>11}"
    print(header)
    for field, cats in hm.items():
        row = (
            f"  {field:<22}  {cats.get('correct', 0):>8}  {cats.get('missed', 0):>8}"
            f"  {cats.get('hallucinated', 0):>12}  {cats.get('wrong_label', 0):>11}"
        )
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold set")
    parser.add_argument("--gold", required=True, help="Gold set JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON")
    parser.add_argument("--out", required=True, help="Output report JSON")
    parser.add_argument("--round2", default=None, help="Second annotation round (intra-kappa)")
    parser.add_argument("--annotator2", default=None, help="Second annotator set (inter-kappa)")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    with open(args.pred, encoding="utf-8") as f:
        pred = json.load(f)

    report = full_report(gold, pred)

    if args.round2:
        with open(args.round2, encoding="utf-8") as f:
            round2 = json.load(f)
        report["intra_annotator_kappa"] = intra_annotator_kappa(gold, round2)

    if args.annotator2:
        with open(args.annotator2, encoding="utf-8") as f:
            ann2 = json.load(f)
        report["inter_annotator_kappa"] = inter_annotator_kappa(gold, ann2)

    save_report(report, Path(args.out))
    print_summary(report)
