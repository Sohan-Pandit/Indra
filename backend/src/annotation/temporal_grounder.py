"""
Stage 6 — Temporal Grounding
Normalizes raw time expressions to ISO 8601 intervals.

Strategy (in priority order):
  1. Detect projected/future keywords → time_type=projected, no normalization.
  2. Regex for explicit 4-digit years → "YYYY-01-01/YYYY-12-31".
  3. Relative expressions ("last decade", "recent years") → anchored to pub year.
  4. "post-YYYY" / "since YYYY" → open-ended interval.
  5. dateparser library for complex natural-language dates.
  6. Fall through → normalized=None, time_type=unknown.

Usage:
    python -m src.annotation.temporal_grounder \
        --input  data/weak_labels/labeled_abstracts.json \
        --output data/weak_labels/labeled_abstracts_time.json
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import Optional

try:
    import dateparser
    _HAS_DATEPARSER = True
except ImportError:
    _HAS_DATEPARSER = False


# ── Keyword sets ──────────────────────────────────────────────────────────────

_PROJECTED_KEYWORDS = {
    "project", "projection", "future", "forecast", "scenario",
    "predict", "prediction", "simulation", "by 2050", "by 2100",
    "by the end of the century", "rcp", "ssp",
}

_RELATIVE_YEAR_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "last decade" / "past 10 years" → 10-year window before pub year
    (re.compile(r"\b(last|past)\s+decade\b", re.I), "decade"),
    (re.compile(r"\b(last|past)\s+(\d+)\s+years?\b", re.I), "n_years"),
    # "recent years" → 5-year window
    (re.compile(r"\brecent\s+years?\b", re.I), "recent_years"),
    # "recent decades" → 20-year window
    (re.compile(r"\brecent\s+decades?\b", re.I), "recent_decades"),
    # "in the 1990s" → 1990-01-01/1999-12-31
    (re.compile(r"\bin\s+the\s+(1\d{2}|20[0-2])0s\b", re.I), "decade_ref"),
]


# ── Core normalizer ───────────────────────────────────────────────────────────

def normalize_time(
    raw_time: str,
    publication_year: Optional[int] = None,
) -> dict:
    """
    Normalize a raw time string to an ISO 8601 interval dict.

    Returns:
        {
          "normalized": "YYYY-MM-DD/YYYY-MM-DD" or None,
          "time_type":  "historical" | "projected" | "relative" | "unknown",
        }
    """
    if not raw_time or not raw_time.strip():
        return {"normalized": None, "time_type": "unknown"}

    raw = raw_time.strip()
    lower = raw.lower()

    # 1. Projected / future
    if any(kw in lower for kw in _PROJECTED_KEYWORDS):
        return {"normalized": None, "time_type": "projected"}

    # 2. Explicit year(s): "2013", "the 1998 event", "1990-2000"
    year_range = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\s*[-–]\s*(1[5-9]\d{2}|20[0-2]\d)\b", raw)
    if year_range:
        y1, y2 = int(year_range.group(1)), int(year_range.group(2))
        return {
            "normalized": f"{min(y1, y2)}-01-01/{max(y1, y2)}-12-31",
            "time_type": "historical",
        }

    single_year = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", raw)
    if single_year:
        y = int(single_year.group(1))
        return {"normalized": f"{y}-01-01/{y}-12-31", "time_type": "historical"}

    # 3. Relative expressions (require publication_year)
    for pattern, kind in _RELATIVE_YEAR_PATTERNS:
        m = pattern.search(lower)
        if not m:
            continue

        # decade_ref ("in the 1990s") doesn't need publication year
        if kind == "decade_ref":
            prefix = m.group(1)          # e.g. "199" from "1990s"
            decade_start = int(prefix + "0")
            return {
                "normalized": f"{decade_start}-01-01/{decade_start + 9}-12-31",
                "time_type": "historical",
            }

        pub = publication_year
        if pub is None:
            continue  # skip anchor-based patterns when pub year is unknown

        if kind == "decade":
            return {"normalized": f"{pub - 10}-01-01/{pub}-12-31", "time_type": "relative"}
        if kind == "n_years":
            n = int(m.group(2))
            return {"normalized": f"{pub - n}-01-01/{pub}-12-31", "time_type": "relative"}
        if kind == "recent_years":
            return {"normalized": f"{pub - 5}-01-01/{pub}-12-31", "time_type": "relative"}
        if kind == "recent_decades":
            return {"normalized": f"{pub - 20}-01-01/{pub}-12-31", "time_type": "relative"}

    # 4. "post-2000" / "since 2000"
    post = re.search(r"\b(?:post[- ]|since\s+)(1[5-9]\d{2}|20[0-2]\d)\b", lower)
    if post:
        start = int(post.group(1))
        end = publication_year or 2024
        return {"normalized": f"{start}-01-01/{end}-12-31", "time_type": "historical"}

    # 5. dateparser fallback
    if _HAS_DATEPARSER:
        try:
            parsed = dateparser.parse(raw, settings={"PREFER_DAY_OF_MONTH": "first"})
            if parsed:
                date_str = parsed.strftime("%Y-%m-%d")
                return {"normalized": f"{date_str}/{date_str}", "time_type": "historical"}
        except Exception:
            pass

    return {"normalized": None, "time_type": "unknown"}


# ── Batch normalizer ──────────────────────────────────────────────────────────

def normalize_records(records: list[dict]) -> list[dict]:
    """
    Apply temporal normalization in-place to each record's time_period field.
    Returns the modified list.
    """
    updated = 0
    for rec in records:
        tp = rec.get("time_period") or {}
        raw = tp.get("raw")
        if not raw:
            continue

        pub_year = rec.get("year") or rec.get("publication_year")
        result = normalize_time(raw, publication_year=pub_year)

        rec["time_period"] = {
            "raw": raw,
            "normalized": result["normalized"],
            "time_type": result["time_type"],
        }
        updated += 1

    print(f"Normalized time in {updated}/{len(records)} records")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize time expressions in labeled records")
    parser.add_argument("--input", required=True, help="Input labeled JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    records = normalize_records(records)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out}")
