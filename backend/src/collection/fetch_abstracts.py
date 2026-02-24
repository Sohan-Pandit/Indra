"""
Stage 1 — Data Collection
Fetches climate impact abstracts from the OpenAlex API (no key required).

Usage:
    python -m src.collection.fetch_abstracts
    python -m src.collection.fetch_abstracts --multilingual
"""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Query configuration ───────────────────────────────────────────────────────

EN_QUERIES = [
    "flood impacts mortality",
    "drought socioeconomic damage",
    "heatwave health effects",
    "extreme rainfall infrastructure",
    "tropical cyclone displacement",
]

MULTILINGUAL_QUERIES: dict[str, list[str]] = {
    "fr": [
        "inondation mortalité impacts",
        "sécheresse dommages socioéconomiques",
        "vague de chaleur santé effets",
        "cyclone tropical déplacement",
    ],
    "es": [
        "inundación mortalidad impactos",
        "sequía daños socioeconómicos",
        "ola de calor salud efectos",
        "ciclón tropical desplazamiento",
    ],
    "de": [
        "Überschwemmung Sterblichkeit Auswirkungen",
        "Dürre sozioökonomische Schäden",
        "Hitzewelle Gesundheit Auswirkungen",
        "tropischer Wirbelsturm Vertreibung",
    ],
}

OPENALEX_URL = "https://api.openalex.org/works"
# Polite pool: identifies your application to OpenAlex
MAILTO = "research@climpact.org"


# ── Abstract reconstruction ───────────────────────────────────────────────────

def reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """
    Convert OpenAlex inverted-index format back to plain text.

    The inverted index maps word → [position, ...].  We invert it to
    position → word, then join in order.
    """
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


# ── Core fetch function ───────────────────────────────────────────────────────

def fetch_abstracts(
    query: str,
    n: int = 200,
    language: str = "en",
) -> list[dict]:
    """
    Fetch up to *n* abstracts from OpenAlex matching *query*.

    Returns a list of dicts with keys: id, title, abstract, year, doi,
    language, query.
    """
    params: dict = {
        "search": query,
        "filter": f"language:{language}",
        "per_page": min(n, 200),
        "select": "id,title,abstract_inverted_index,publication_year,doi",
        "mailto": MAILTO,
    }

    all_results: list[dict] = []
    cursor = "*"

    with tqdm(total=n, desc=f"[{language}] {query[:40]}", unit="abs", leave=False) as pbar:
        while len(all_results) < n:
            params["cursor"] = cursor
            try:
                resp = requests.get(OPENALEX_URL, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"  Request error: {exc}. Stopping early.")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                abstract = reconstruct_abstract(r.get("abstract_inverted_index"))
                if not abstract:
                    continue
                work_id = r["id"].split("/")[-1]  # "https://openalex.org/W123" → "W123"
                all_results.append(
                    {
                        "id": work_id,
                        "title": r.get("title") or "",
                        "abstract": abstract,
                        "year": r.get("publication_year"),
                        "doi": r.get("doi"),
                        "language": language,
                        "query": query,
                    }
                )
                pbar.update(1)
                if len(all_results) >= n:
                    break

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

            time.sleep(0.1)  # respect OpenAlex rate limit

    return all_results[:n]


# ── Collection pipelines ──────────────────────────────────────────────────────

def collect_english(
    output_path: Path,
    n_per_query: int = 200,
) -> list[dict]:
    """Collect English abstracts for all queries, deduplicate, and save."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_abstracts: list[dict] = []
    seen_ids: set[str] = set()

    for query in EN_QUERIES:
        print(f"\nFetching: {query!r}")
        results = fetch_abstracts(query, n=n_per_query, language="en")
        added = 0
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_abstracts.append(r)
                added += 1
        print(f"  → {added} new  |  {len(all_abstracts)} total unique")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_abstracts, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_abstracts)} English abstracts → {output_path}")
    return all_abstracts


def collect_multilingual(
    output_path: Path,
    n_per_query: int = 30,
) -> list[dict]:
    """Collect ~100 abstracts per language (fr, es, de) and save."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_abstracts: list[dict] = []
    seen_ids: set[str] = set()

    for lang, queries in MULTILINGUAL_QUERIES.items():
        print(f"\nLanguage: {lang.upper()}")
        for query in queries:
            results = fetch_abstracts(query, n=n_per_query, language=lang)
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_abstracts.append(r)
        lang_count = sum(1 for a in all_abstracts if a["language"] == lang)
        print(f"  → {lang_count} unique [{lang}] so far")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_abstracts, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_abstracts)} multilingual abstracts → {output_path}")
    return all_abstracts


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch climate abstracts from OpenAlex")
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Also collect French, Spanish, German abstracts",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Abstracts per query (default 200)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: data/raw relative to project root)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2] / "data" / "raw"
    if args.outdir:
        base = Path(args.outdir)

    collect_english(base / "abstracts.json", n_per_query=args.n)

    if args.multilingual:
        collect_multilingual(base / "abstracts_multilingual.json")
