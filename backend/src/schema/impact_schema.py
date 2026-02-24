"""
Stage 2 — Schema Design
Defines the canonical ImpactRecord schema using Pydantic v2.

Design decisions (documented):
- "increased vulnerability" is NOT an impact; only realized impacts qualify.
- Projected mortality differs from observed: use impact_type field.
- Range magnitudes ("2,000-5,000 deaths") are stored as strings with
  magnitude_vague=True.
- Null unknown fields rather than empty strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ── Controlled vocabularies ───────────────────────────────────────────────────

HazardType = Literal[
    "flood",
    "drought",
    "heatwave",
    "extreme_rainfall",
    "tropical_cyclone",
    "wildfire",
    "storm",
    "landslide",
    "sea_level_rise",
    "other",
]

ImpactDomain = Literal[
    "mortality",
    "morbidity",
    "displacement",
    "economic_loss",
    "infrastructure",
    "agriculture",
    "ecosystem",
    "mental_health",
    "food_security",
    "water_security",
    "other",
]

ImpactType = Literal["observed", "projected", "modeled", "unknown"]

RelationPredicate = Literal[
    "caused",
    "contributed_to",
    "associated_with",
    "mitigated",
    "none",
]

UncertaintyLevel = Literal["low", "medium", "high", "unknown"]

TimeType = Literal["historical", "projected", "relative", "unknown"]


# ── Sub-models ────────────────────────────────────────────────────────────────

class Location(BaseModel):
    """Geographic location, raw and ISO-normalized."""

    raw: Optional[str] = None
    # Normalized format: "City, CC" or just "CC" (ISO 3166-1 alpha-2)
    normalized: Optional[str] = None


class TimePeriod(BaseModel):
    """Time expression, raw and ISO 8601 interval-normalized."""

    raw: Optional[str] = None
    # ISO 8601 interval: "YYYY-MM-DD/YYYY-MM-DD"
    normalized: Optional[str] = None
    time_type: TimeType = "unknown"


class CausalRelation(BaseModel):
    """Subject–predicate–object triple with optional dependency path."""

    subject: Optional[str] = None          # hazard mention
    predicate: Optional[RelationPredicate] = None
    object: Optional[str] = None           # impact mention
    # Dependency path between head tokens, e.g. "nsubj → resulted → in"
    dependency_path: Optional[str] = None


# ── Core record ───────────────────────────────────────────────────────────────

class ImpactRecord(BaseModel):
    """
    One structured climate impact record extracted from an abstract.

    All optional fields default to None; callers should leave genuinely
    unknown information as None rather than guessing.
    """

    abstract_id: str

    # Hazard
    hazard_type: Optional[HazardType] = None
    hazard_intensity: Optional[str] = None   # e.g. ">200 mm/day"

    # Geography and time
    location: Optional[Location] = None
    time_period: Optional[TimePeriod] = None

    # Impact
    impact_domain: Optional[ImpactDomain] = None
    impact_type: Optional[ImpactType] = None
    impact_magnitude: Optional[str] = None   # e.g. "5 700 deaths"
    magnitude_vague: bool = False            # True when a range or approximate

    # Population
    affected_group: Optional[str] = None

    # Causality
    causal_relation: Optional[CausalRelation] = None

    # Uncertainty
    uncertainty_level: Optional[UncertaintyLevel] = None
    uncertainty_source: Optional[str] = None  # e.g. "observational", "modeled"
    hedge_terms: list[str] = Field(default_factory=list)

    # Grounding
    grounding_quote: Optional[str] = None
    grounding_verified: bool = False

    # Internal — populated by validation pipeline; stored as regular field
    validation_errors: list[str] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def _auto_flag_vague_magnitude(self) -> "ImpactRecord":
        """Mark magnitude as vague when it contains range indicators."""
        if self.impact_magnitude:
            vague_signals = ["–", "-", " to ", "~", ">", "<", "approx", "around", "about"]
            if any(s in self.impact_magnitude for s in vague_signals):
                self.magnitude_vague = True
        return self

    # ── Serialisation helpers ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "ImpactRecord":
        return cls.model_validate(data)

    # ── Pretty display ─────────────────────────────────────────────────────

    def summary(self) -> str:
        parts = [f"[{self.abstract_id}]"]
        if self.hazard_type:
            parts.append(f"hazard={self.hazard_type}")
        if self.impact_domain:
            parts.append(f"impact={self.impact_domain}")
        if self.location and self.location.normalized:
            parts.append(f"loc={self.location.normalized}")
        if self.uncertainty_level:
            parts.append(f"uncertainty={self.uncertainty_level}")
        return "  ".join(parts)


# ── Validation utility ────────────────────────────────────────────────────────

def validate_record(data: dict) -> tuple[bool, list[str]]:
    """
    Validate a raw dict against ImpactRecord.

    Returns:
        (True, []) on success.
        (False, [error_strings]) on failure.
    """
    from pydantic import ValidationError

    try:
        ImpactRecord.model_validate(data)
        return True, []
    except ValidationError as exc:
        return False, [f"{e['loc']}: {e['msg']}" for e in exc.errors()]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[ImpactRecord]:
    """Load a JSON list of impact records from disk."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [ImpactRecord.from_dict(r) for r in raw]


def save_records(records: list[ImpactRecord], path: Path) -> None:
    """Serialise a list of ImpactRecord objects to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)


def save_records_as_python(records: list[ImpactRecord], path: Path) -> None:
    """Serialise a list of ImpactRecord objects to a Python file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\"\"\"Generated by Indra LLM Labeler.\"\"\"\n\n")
        f.write("from src.schema.impact_schema import ImpactRecord, Location, TimePeriod, CausalRelation\n\n")
        f.write("records = [\n")
        for record in records:
            # repr() of a pydantic model isn't always copy-pasteable constructor code,
            # but model_dump() gives us a dict we can pass to the constructor.
            # A cleaner way is to dump the dict and wrap it in the class constructor.
            # However, for 'ImpactRecord', we can just write the dict literals if the recipient
            # iterates them and does ImpactRecord.model_validate().
            # 
            # User asked for "everything in python", suggesting they want an importable list of objects.
            # Let's write them as direct object instantiations for best IDE support.
            
            # Pydantic v2 repr is usually quite good. Let's try to use the repr of the object 
            # but we need to ensure all sub-models are imported.
            
            # Actually, standard repr() might not be fully reproducible if it contains 'None' 
            # for optional fields, which is fine, but let's stick to a robust dict-based approach
            # wrapped in the constructor for clarity and safety.
            
            data = record.model_dump(exclude_none=True)
            f.write(f"    ImpactRecord.model_validate({data!r}),\n")
        f.write("]\n")


# ── Schema JSON export (for prompts) ─────────────────────────────────────────

def schema_as_json_string() -> str:
    """Return a compact JSON schema representation for use in prompts."""
    example = {
        "abstract_id": "OA_12345",
        "hazard_type": "flood | drought | heatwave | extreme_rainfall | tropical_cyclone | wildfire | storm | landslide | sea_level_rise | other",
        "hazard_intensity": "string or null",
        "location": {"raw": "string or null", "normalized": "City, CC or CC or null"},
        "time_period": {
            "raw": "string or null",
            "normalized": "YYYY-MM-DD/YYYY-MM-DD or null",
            "time_type": "historical | projected | relative | unknown",
        },
        "impact_domain": "mortality | morbidity | displacement | economic_loss | infrastructure | agriculture | ecosystem | mental_health | food_security | water_security | other",
        "impact_type": "observed | projected | modeled | unknown",
        "impact_magnitude": "string with units or null",
        "magnitude_vague": "boolean — true if range or approximate",
        "affected_group": "string or null",
        "causal_relation": {
            "subject": "hazard span string or null",
            "predicate": "caused | contributed_to | associated_with | mitigated | none",
            "object": "impact span string or null",
            "dependency_path": "e.g. 'nsubj → resulted → in' or null",
        },
        "uncertainty_level": "low | medium | high | unknown",
        "uncertainty_source": "observational | modeled | survey | expert_judgment or null",
        "hedge_terms": ["list of hedge words found"],
        "grounding_quote": "verbatim sentence from abstract or null",
        "grounding_verified": False,
    }
    return json.dumps(example, indent=2)
