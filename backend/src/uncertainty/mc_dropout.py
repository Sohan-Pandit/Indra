"""
Stage 7B — MC Dropout Epistemic Uncertainty

Runs the trained JER model N times with dropout kept active to estimate
epistemic (model) uncertainty per abstract.

The key research question:
  Do abstracts with high hedge-term density produce higher MC Dropout variance?
  → Compute Spearman correlation between hedge_score and mc_variance.
  → Where they diverge is the most interesting finding.

Usage:
    python -m src.uncertainty.mc_dropout \
        --model     models/jer \
        --abstracts data/raw/abstracts.json \
        --hedges    data/raw/hedge_scores.json \
        --output    data/raw/uncertainty_report.json \
        --n_passes  30
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Core MC Dropout function ──────────────────────────────────────────────────

def mc_dropout_predict(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_passes: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run Monte Carlo Dropout inference over *n_passes* stochastic forward passes.

    model.train() is called to keep dropout active; gradients are disabled.

    Returns:
        mean_pred : mean softmax probabilities  (B, L, C) or (B, C)
        variance  : variance across passes      same shape as mean_pred
    """
    model.train()  # activate dropout layers
    predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_passes):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.get("span_logits", next(iter(out.values())))
            probs = torch.softmax(logits, dim=-1)
            predictions.append(probs)

    stacked = torch.stack(predictions, dim=0)  # (n_passes, B, L, C)
    return stacked.mean(0), stacked.var(0)


def predictive_entropy(mean_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Total uncertainty = entropy of mean prediction."""
    return -(mean_pred * (mean_pred + eps).log()).sum(-1)


def epistemic_uncertainty(variance: torch.Tensor) -> torch.Tensor:
    """Summarise variance tensor to a scalar per position."""
    return variance.mean(-1)


# ── Batch uncertainty computation ─────────────────────────────────────────────

def compute_batch_uncertainty(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    n_passes: int = 30,
) -> list[dict]:
    """
    Compute abstract-level uncertainty scores over an entire dataloader.

    Returns a list of dicts with keys:
        abstract_idx, epistemic_uncertainty, predictive_entropy.
    """
    results: list[dict] = []
    abs_idx = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        mean_pred, variance = mc_dropout_predict(
            model, input_ids, attention_mask, n_passes=n_passes
        )

        mask = attention_mask.bool()
        for i in range(input_ids.size(0)):
            valid_mask = mask[i]
            tok_var = variance[i][valid_mask]           # (valid_L, C)
            tok_ent = predictive_entropy(mean_pred[i][valid_mask])  # (valid_L,)

            results.append(
                {
                    "abstract_idx": abs_idx,
                    "epistemic_uncertainty": float(tok_var.mean().item()),
                    "predictive_entropy": float(tok_ent.mean().item()),
                }
            )
            abs_idx += 1

    return results


# ── Correlation analysis ──────────────────────────────────────────────────────

def correlate_hedge_with_mc(
    hedge_scores: list[float],
    mc_variances: list[float],
) -> dict:
    """
    Compute Spearman rank correlation between linguistic hedge scores and
    MC Dropout variance.

    Returns a dict with spearman_r, p_value, n, and a human-readable note.
    """
    from scipy.stats import spearmanr

    if len(hedge_scores) != len(mc_variances):
        raise ValueError("hedge_scores and mc_variances must have the same length")

    r, p = spearmanr(hedge_scores, mc_variances)
    return {
        "spearman_r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": len(hedge_scores),
        "significant": bool(p < 0.05),
        "interpretation": _interpret(r, p),
    }


def _interpret(r: float, p: float) -> str:
    if p >= 0.05:
        return (
            f"No significant correlation (r={r:.3f}, p={p:.4f}). "
            "Hedge language and model uncertainty are independent here."
        )
    strength = (
        "weak" if abs(r) < 0.3
        else "moderate" if abs(r) < 0.6
        else "strong"
    )
    direction = "positive" if r > 0 else "negative"
    note = (
        "Abstracts with more hedging also confuse the model more."
        if r > 0
        else "High hedge density correlates with lower model uncertainty — "
        "possibly because hedged abstracts use more domain-specific language."
    )
    return f"{strength.capitalize()} {direction} correlation (r={r:.3f}, p={p:.4f}). {note}"


def find_divergent_abstracts(
    hedge_scores: list[float],
    mc_variances: list[float],
    ids: list[str],
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """
    Find abstracts where hedge score and MC variance diverge the most.

    Returns:
        "high_hedge_low_mc"  — certain model, uncertain language
        "low_hedge_high_mc"  — uncertain model, certain language
    """
    h = np.array(hedge_scores)
    v = np.array(mc_variances)

    # Normalise both to [0, 1]
    h_n = (h - h.min()) / (h.max() - h.min() + 1e-8)
    v_n = (v - v.min()) / (v.max() - v.min() + 1e-8)

    divergence = h_n - v_n

    top_hh_lm = np.argsort(-divergence)[:top_k]   # high hedge, low MC
    top_lh_hm = np.argsort(divergence)[:top_k]    # low hedge, high MC

    def entry(i: int) -> dict:
        return {
            "id": ids[i],
            "hedge_score": round(float(h[i]), 4),
            "mc_variance": round(float(v[i]), 6),
            "divergence": round(float(divergence[i]), 4),
        }

    return {
        "high_hedge_low_mc": [entry(i) for i in top_hh_lm],
        "low_hedge_high_mc": [entry(i) for i in top_lh_hm],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty analysis")
    parser.add_argument("--model", required=True, help="Directory containing jer_model.pt")
    parser.add_argument("--abstracts", required=True, help="Abstracts JSON")
    parser.add_argument("--hedges", required=True, help="Hedge scores JSON")
    parser.add_argument("--output", required=True, help="Output report JSON")
    parser.add_argument("--n_passes", type=int, default=30, help="MC Dropout passes")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from src.extraction.jer_model import JERModel, JERDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = JERModel()
    model.load_state_dict(torch.load(model_dir / "jer_model.pt", map_location=device))
    model.to(device)

    with open(args.abstracts, encoding="utf-8") as f:
        abstracts = json.load(f)
    with open(args.hedges, encoding="utf-8") as f:
        hedge_data = json.load(f)

    hedge_by_id = {h["id"]: h["hedge_score"] for h in hedge_data}

    dataset = JERDataset(abstracts, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch)

    print(f"Running {args.n_passes} MC Dropout passes on {len(dataset)} abstracts…")
    mc_results = compute_batch_uncertainty(model, loader, device, n_passes=args.n_passes)

    ids = [a["id"] for a in abstracts]
    hedge_scores = [hedge_by_id.get(aid, 0.0) for aid in ids]
    mc_variances = [r["epistemic_uncertainty"] for r in mc_results]

    correlation = correlate_hedge_with_mc(hedge_scores, mc_variances)
    divergent = find_divergent_abstracts(hedge_scores, mc_variances, ids)

    report = {
        "n_abstracts": len(ids),
        "n_mc_passes": args.n_passes,
        "correlation": correlation,
        "divergent_cases": divergent,
        "per_abstract": [
            {
                "id": ids[i],
                "hedge_score": hedge_scores[i],
                **mc_results[i],
            }
            for i in range(len(ids))
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nCorrelation: {correlation['interpretation']}")
    print(f"Report saved → {out}")
