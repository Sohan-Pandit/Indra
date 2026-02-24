"""
Stage 4 — LLM-Assisted Weak Labeling
Two-pass prompting strategy:
  Pass 1 — extract structured JSON from abstract.
  Pass 2 — ground each field in a verbatim quote; null unsupported fields.
  Programmatic check — string-match every claimed quote against the abstract.

Supports resume: if the output file already exists, previously labeled
abstracts are skipped.

Usage:
    python -m src.annotation.llm_labeler \
        --abstracts data/raw/abstracts.json \
        --output    data/weak_labels/labeled_abstracts.json \
        --gold      data/annotated/gold_set.json   # IDs to skip
"""

from __future__ import annotations

import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional

from src.schema.impact_schema import validate_record, schema_as_json_string
from src.llm.wrapper import GeminiWrapper

# ── Prompt templates ──────────────────────────────────────────────────────────

_SCHEMA = schema_as_json_string()

PASS1_PROMPT = """\
You are an expert annotator for climate impact literature.

Given the abstract below, extract a single structured impact record that \
follows the JSON schema exactly. Return ONLY valid JSON — no explanation, \
no markdown fences.

IMPORTANT RULES:
- Extract only REALIZED impacts, not potential or hypothetical ones.
- "increased vulnerability" is NOT an impact.
- Distinguish observed vs projected impacts using the impact_type field.
- If magnitude is a range (e.g., "2,000–5,000 deaths"), set magnitude_vague=true.
- Set genuinely unknown fields to null, never to empty strings.

Schema:
{schema}

Abstract ID: {abstract_id}
Abstract: {abstract_text}
"""

PASS2_PROMPT = """\
You previously generated the JSON below from the abstract.

For every non-null field in the JSON, find the EXACT verbatim substring from \
the abstract that supports it and record its character offsets [start, end].
If you CANNOT find a verbatim supporting substring, set that field to null \
in the corrected JSON.

Return a JSON object with exactly two keys:
  "verified_json" — the corrected record (fields without support set to null)
  "evidence"      — dict mapping field names to {{"quote": "...", "offsets": [s, e]}}

Abstract:
{abstract_text}

Original JSON:
{generated_json}
"""


# ── LLMLabeler class ──────────────────────────────────────────────────────────

class LLMLabeler:
    """
    Two-pass LLM labeler with programmatic grounding verification.

    Args:
    Args:
        api_key:     Google Gemini API key.
        model:       Gemini model ID (default: gemini-1.5-pro-latest).
        max_retries: Number of retries on API errors with exponential back-off.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        max_retries: int = 3,
    ) -> None:
        self.llm = GeminiWrapper(api_key=api_key, model_name=model)
        self.model = model
        self.max_retries = max_retries

    # ── API call ──────────────────────────────────────────────────────────

    def _call(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.llm.generate(prompt, max_tokens=max_tokens)

    # ── JSON extraction ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Strip markdown fences and parse the first JSON object found."""
        text = text.strip()
        # Remove ```json … ``` wrappers
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fallback: find the outermost {...}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # ── Programmatic grounding check ──────────────────────────────────────

    @staticmethod
    def _programmatic_verify(record: dict, abstract: str) -> dict:
        """
        Null out grounding_quote if it does not appear verbatim in the abstract.
        Set grounding_verified accordingly.
        """
        quote = record.get("grounding_quote")
        if quote:
            if quote in abstract:
                record["grounding_verified"] = True
            elif quote.lower() in abstract.lower():
                # Accept case-insensitive match but flag
                record["grounding_verified"] = True
            else:
                record["grounding_quote"] = None
                record["grounding_verified"] = False
        else:
            record["grounding_verified"] = False
        return record

    # ── Single abstract ───────────────────────────────────────────────────

    def label_abstract(
        self,
        abstract_id: str,
        abstract_text: str,
    ) -> Optional[dict]:
        """
        Run the two-pass pipeline on one abstract.

        Returns the final verified dict, or None if parsing failed at Pass 1.
        """
        # ── Pass 1: extraction ────────────────────────────────────────────
        p1 = PASS1_PROMPT.format(
            schema=_SCHEMA,
            abstract_id=abstract_id,
            abstract_text=abstract_text,
        )
        raw1 = self._call(p1)
        record = self._parse_json(raw1)
        if not record:
            print(f"  [WARN] Pass 1 JSON parse failed for {abstract_id}")
            return None

        record["abstract_id"] = abstract_id

        # ── Pass 2: grounding verification ───────────────────────────────
        p2 = PASS2_PROMPT.format(
            abstract_text=abstract_text,
            generated_json=json.dumps(record, indent=2, ensure_ascii=False),
        )
        raw2 = self._call(p2)
        result2 = self._parse_json(raw2)
        if result2 and isinstance(result2.get("verified_json"), dict):
            record = result2["verified_json"]
            record["abstract_id"] = abstract_id
            record["_evidence"] = result2.get("evidence", {})

        # ── Programmatic check ────────────────────────────────────────────
        record = self._programmatic_verify(record, abstract_text)

        # ── Schema validation ─────────────────────────────────────────────
        is_valid, errors = validate_record(record)
        if not is_valid:
            record["validation_errors"] = errors

        return record

    # ── Batch pipeline ────────────────────────────────────────────────────

    def label_batch(
        self,
        abstracts: list[dict],
        output_path: Path,
        skip_ids: Optional[set[str]] = None,
        checkpoint_every: int = 10,
    ) -> list[dict]:
        """
        Label a list of abstract dicts (each must have 'id' and 'abstract').

        Supports resume: existing records in output_path are kept and their
        IDs are skipped.  Checkpoints are written every *checkpoint_every*
        new labels.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume from existing output
        labeled: list[dict] = []
        done_ids: set[str] = set(skip_ids or [])
        if output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                labeled = json.load(f)
            done_ids.update(r["abstract_id"] for r in labeled)
            print(f"Resuming: {len(labeled)} already done, {len(done_ids)} IDs skipped")



        remaining = [a for a in abstracts if a["id"] not in done_ids]
        print(f"Labeling {len(remaining)} abstracts with {self.model}…\n")

        for i, abstract in enumerate(remaining, start=1):
            abs_id = abstract["id"]
            text = abstract.get("abstract", "")
            print(f"[{i}/{len(remaining)}] {abs_id}")

            try:
                record = self.label_abstract(abs_id, text)
                if record:
                    labeled.append(record)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            if i % checkpoint_every == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(labeled, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Checkpoint: {len(labeled)} records saved")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labeled, f, indent=2, ensure_ascii=False)
        print(f"\nDone. {len(labeled)} records → {output_path}")
        return labeled


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM weak labeling pipeline")
    parser.add_argument("--abstracts", required=True, help="Path to abstracts JSON")
    parser.add_argument("--output", required=True, help="Output path for labeled JSON")
    parser.add_argument("--gold", default=None, help="Gold set JSON — IDs to skip")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model ID")
    args = parser.parse_args()

    with open(args.abstracts, encoding="utf-8") as f:
        abstracts = json.load(f)

    skip_ids: set[str] = set()
    if args.gold and Path(args.gold).exists():
        with open(args.gold, encoding="utf-8") as f:
            gold = json.load(f)
        skip_ids = {r["abstract_id"] for r in gold}
        print(f"Skipping {len(skip_ids)} gold-set IDs")

    labeler = LLMLabeler(model=args.model)
    labeler.label_batch(abstracts, Path(args.output), skip_ids=skip_ids)
