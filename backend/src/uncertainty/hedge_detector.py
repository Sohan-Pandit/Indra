"""
Stage 7A — Linguistic Hedge Detection

Assigns an uncertainty score to each abstract based on the density and
weight of hedge terms.  The lexicon covers three tiers:

  high_uncertainty   (weight 1.0): may, might, could, possibly, …
  medium_uncertainty (weight 0.6): likely, probably, suggests, …
  low_uncertainty    (weight 0.2): estimated, approximately, reported, …

Score = (weighted_count / word_count) × 100, capped and normalised to [0, 1].
Level: low (<0.1), medium (0.1–0.4), high (>0.4).

Usage:
    python -m src.uncertainty.hedge_detector \
        --input  data/raw/abstracts.json \
        --output data/raw/hedge_scores.json
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import NamedTuple

# ── Lexicon ───────────────────────────────────────────────────────────────────

HEDGE_TERMS: dict[str, list[str]] = {
    "high_uncertainty": [
        "may", "might", "could", "possibly", "uncertain", "unclear",
        "unknown", "speculative", "hypothetical", "unconfirmed", "tentative",
        "perhaps", "conceivably", "questionable", "debatable", "ambiguous",
        "equivocal", "inconclusive",
    ],
    "medium_uncertainty": [
        "likely", "probably", "suggests", "suggest", "indicates", "indicate",
        "appears", "seem", "seems", "tends", "generally", "often", "typically",
        "roughly", "presumably", "apparently", "supposedly", "infer", "inferred",
        "consistent with", "can be",
    ],
    "low_uncertainty": [
        "estimated", "approximately", "around", "reported", "observed",
        "measured", "calculated", "documented", "recorded", "noted",
        "found", "detected", "identified", "confirmed",
    ],
}

HEDGE_WEIGHTS: dict[str, float] = {
    "high_uncertainty": 1.0,
    "medium_uncertainty": 0.6,
    "low_uncertainty": 0.2,
}

# Normalisation cap: density ≥ 5.0 weighted-terms-per-100-words → score = 1.0
_DENSITY_CAP = 5.0


# ── Result type ───────────────────────────────────────────────────────────────

class HedgeScore(NamedTuple):
    score: float           # normalised [0, 1]
    level: str             # "low" | "medium" | "high" | "none"
    found_terms: dict      # category → list[str] (with repetitions)
    term_count: int        # raw matched hedge-term count
    word_count: int        # total word count of the abstract


# ── Core detector ─────────────────────────────────────────────────────────────

def detect_hedges(text: str) -> HedgeScore:
    """
    Score a single text for linguistic hedging.

    Multi-word phrases (e.g. "consistent with") are matched before
    individual tokens are split.
    """
    if not text:
        return HedgeScore(0.0, "none", {c: [] for c in HEDGE_TERMS}, 0, 0)

    lower = text.lower()

    found: dict[str, list[str]] = {cat: [] for cat in HEDGE_TERMS}

    # Multi-word phrases first (greedy, order matters)
    _multi = ["consistent with", "can be"]
    _consumed_spans: list[tuple[int, int]] = []
    for phrase in _multi:
        start = 0
        while True:
            idx = lower.find(phrase, start)
            if idx == -1:
                break
            _consumed_spans.append((idx, idx + len(phrase)))
            for cat, terms in HEDGE_TERMS.items():
                if phrase in terms:
                    found[cat].append(phrase)
            start = idx + 1

    # Tokenise — skip chars that belong to already-consumed spans
    words = re.findall(r"\b\w+\b", lower)
    word_count = len(words)

    word_set = words  # preserves repetitions for counting
    for cat, terms in HEDGE_TERMS.items():
        for term in terms:
            if " " in term:
                continue  # already handled above
            count = word_set.count(term)
            if count:
                found[cat].extend([term] * count)

    total_count = sum(len(v) for v in found.values())
    weighted = sum(len(v) * HEDGE_WEIGHTS[cat] for cat, v in found.items())

    if word_count == 0:
        return HedgeScore(0.0, "none", found, 0, 0)

    density = (weighted / word_count) * 100
    score = min(density / _DENSITY_CAP, 1.0)

    if score == 0.0:
        level = "none"
    elif score < 0.1:
        level = "low"
    elif score < 0.4:
        level = "medium"
    else:
        level = "high"

    return HedgeScore(score, level, found, total_count, word_count)


def get_unique_hedge_terms(text: str) -> list[str]:
    """Return a sorted, deduplicated list of all hedge terms found in text."""
    hs = detect_hedges(text)
    all_terms: set[str] = set()
    for terms in hs.found_terms.values():
        all_terms.update(terms)
    return sorted(all_terms)


# ── Corpus scorer ─────────────────────────────────────────────────────────────

def score_corpus(abstracts: list[dict]) -> list[dict]:
    """
    Score a list of abstract dicts (must have 'id' and 'abstract' keys).

    Returns a list of score dicts suitable for JSON serialisation.
    """
    results: list[dict] = []
    for ab in abstracts:
        hs = detect_hedges(ab.get("abstract", ""))
        results.append(
            {
                "id": ab.get("id", ""),
                "hedge_score": round(hs.score, 4),
                "hedge_level": hs.level,
                "hedge_terms": get_unique_hedge_terms(ab.get("abstract", "")),
                "hedge_count": hs.term_count,
                "word_count": hs.word_count,
            }
        )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score abstracts for linguistic hedging")
    parser.add_argument("--input", required=True, help="abstracts JSON")
    parser.add_argument("--output", required=True, help="hedge scores JSON output")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        abstracts = json.load(f)

    scores = score_corpus(abstracts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    avg = sum(s["hedge_score"] for s in scores) / max(len(scores), 1)
    print(f"Scored {len(scores)} abstracts  |  mean hedge score = {avg:.4f}")
    print(f"Saved → {out}")
