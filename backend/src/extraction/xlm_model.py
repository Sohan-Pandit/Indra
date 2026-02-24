"""
Stage 9 — Cross-lingual Zero-Shot Transfer

Swaps the backbone from DeBERTa-v3-small to XLM-RoBERTa-base while keeping
all output heads and training logic identical.  The same JERTrainer class is
reused; only the model instantiation changes.

Usage:
    python -m src.extraction.xlm_model \
        --weak   data/weak_labels/labeled_abstracts.json \
        --gold   data/annotated/gold_set.json \
        --multi  data/raw/abstracts_multilingual.json \
        --out    models/xlm_jer

    # After training, evaluate zero-shot on each language:
    python -m src.extraction.xlm_model --eval \
        --model  models/xlm_jer \
        --data   data/raw/abstracts_multilingual.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from src.extraction.jer_model import (
    JERModel,
    JERDataset,
    JERTrainer,
    BIO_LABELS,
    RELATION_TYPES,
)

XLM_BACKBONE = "xlm-roberta-base"


def build_xlm_model(dropout: float = 0.1) -> JERModel:
    """Return a JERModel using XLM-RoBERTa as the backbone."""
    return JERModel(
        backbone=XLM_BACKBONE,
        num_bio_labels=len(BIO_LABELS),
        num_rel_labels=len(RELATION_TYPES),
        dropout=dropout,
    )


def evaluate_zero_shot(
    trainer: JERTrainer,
    multilingual_records: list[dict],
    languages: list[str] = None,
    n_eval: int = 20,
) -> dict[str, list[dict]]:
    """
    Run inference on multilingual records and collect span predictions per language.

    Returns a dict mapping language code → list of prediction dicts for manual evaluation.
    """
    if languages is None:
        languages = ["fr", "es", "de"]

    results: dict[str, list[dict]] = {}

    for lang in languages:
        lang_records = [r for r in multilingual_records if r.get("language") == lang]
        sample = lang_records[:n_eval]
        preds: list[dict] = []

        for rec in sample:
            text = rec.get("abstract", "")
            if not text:
                continue
            pred = trainer.predict(text)
            preds.append(
                {
                    "id": rec.get("id"),
                    "language": lang,
                    "abstract": text[:300],  # truncated for readability
                    "predicted_spans": pred["spans"],
                }
            )

        results[lang] = preds
        print(f"[{lang}] Predicted {len(preds)} abstracts")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLM-RoBERTa JER — cross-lingual transfer")
    parser.add_argument("--weak", default=None, help="Weak labels JSON (English training)")
    parser.add_argument("--gold", default=None, help="Gold set JSON (val)")
    parser.add_argument("--multi", default=None, help="Multilingual abstracts JSON")
    parser.add_argument("--out", default="models/xlm_jer", help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Skip training; run zero-shot eval")
    parser.add_argument("--model", default="models/xlm_jer", help="Trained model dir (eval mode)")
    parser.add_argument("--data", default=None, help="Multilingual JSON for eval mode")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(XLM_BACKBONE)
    model = build_xlm_model()

    if args.eval:
        # Evaluation mode: load trained weights, run zero-shot on multilingual data
        model_path = Path(args.model) / "jer_model.pt"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        trainer = JERTrainer(model, tokenizer)

        data_path = args.data or args.multi
        with open(data_path, encoding="utf-8") as f:
            multi_records = json.load(f)

        results = evaluate_zero_shot(trainer, multi_records)

        out = Path(args.model) / "zero_shot_predictions.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Zero-shot predictions saved → {out}")

    else:
        # Training mode
        with open(args.weak, encoding="utf-8") as f:
            weak_records = json.load(f)
        with open(args.gold, encoding="utf-8") as f:
            gold_records = json.load(f)

        train_ds = JERDataset(weak_records, tokenizer)
        val_ds = JERDataset(gold_records, tokenizer)

        trainer = JERTrainer(model, tokenizer)
        trainer.train(
            train_ds,
            val_ds,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            output_dir=args.out,
        )
        print(f"XLM-RoBERTa JER model saved → {args.out}/")

        if args.multi:
            with open(args.multi, encoding="utf-8") as f:
                multi_records = json.load(f)
            results = evaluate_zero_shot(trainer, multi_records)
            out = Path(args.out) / "zero_shot_predictions.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Zero-shot predictions saved → {out}")
