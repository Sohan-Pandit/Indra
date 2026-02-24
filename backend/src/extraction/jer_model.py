"""
Stage 5 — Joint Entity and Relation Extraction (JER) Model

Architecture: DeBERTa-v3-small backbone with two output heads.
  Head 1 — BIO span tagging: HAZARD, IMPACT, LOCATION, TIME, AFFECTED_GROUP.
  Head 2 — Relation classification over (span1, span2) pairs:
            caused | contributed_to | associated_with | mitigated | none.

SpaCy dependency paths are encoded as token-ID sequences and embedded via a
small learnable embedding table, then concatenated with span representations
before the relation head.  This operationalizes the insight that
"nsubj → resulted → in" is a strong signal for the `caused` predicate.

Usage (training):
    python -m src.extraction.jer_model \
        --weak  data/weak_labels/labeled_abstracts.json \
        --gold  data/annotated/gold_set.json \
        --out   models/jer
"""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ── Label sets ────────────────────────────────────────────────────────────────

ENTITY_TYPES = ["HAZARD", "IMPACT", "LOCATION", "TIME", "AFFECTED_GROUP"]
RELATION_TYPES = ["caused", "contributed_to", "associated_with", "mitigated", "none"]

BIO_LABELS = ["O"] + [
    f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ["B", "I"]
]
BIO2IDX: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
IDX2BIO: dict[int, str] = {i: label for label, i in BIO2IDX.items()}

REL2IDX: dict[str, int] = {rel: i for i, rel in enumerate(RELATION_TYPES)}
IDX2REL: dict[int, str] = {i: rel for rel, i in REL2IDX.items()}


# ── Dependency path encoder ───────────────────────────────────────────────────

class DepPathEncoder:
    """Incrementally builds a vocabulary of dependency relation+POS tokens."""

    def __init__(self, max_vocab: int = 512) -> None:
        self.vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.max_vocab = max_vocab

    def _key(self, token) -> str:
        return f"{token.dep_}:{token.pos_}"

    def encode(self, path_tokens: list) -> list[int]:
        ids: list[int] = []
        for tok in path_tokens:
            key = self._key(tok)
            if key not in self.vocab and len(self.vocab) < self.max_vocab:
                self.vocab[key] = len(self.vocab)
            ids.append(self.vocab.get(key, 1))
        return ids

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: Path) -> "DepPathEncoder":
        enc = cls()
        with open(path) as f:
            enc.vocab = json.load(f)
        return enc


def get_dependency_path(doc, span1_tokens: list, span2_tokens: list) -> list:
    """
    Return the shortest dependency-tree path between the head tokens of two spans.
    Falls back to an empty list if no path exists.
    """
    try:
        import networkx as nx
    except ImportError:
        return []

    G = nx.Graph()
    for tok in doc:
        G.add_node(tok.i)
        if tok.head.i != tok.i:
            G.add_edge(tok.i, tok.head.i)

    # Use the syntactic head of each span (root token)
    def span_head(tokens: list):
        return min(tokens, key=lambda t: t.head.i if t.head != t else -1)

    if not span1_tokens or not span2_tokens:
        return []

    h1 = span_head(span1_tokens)
    h2 = span_head(span2_tokens)
    try:
        path_indices = nx.shortest_path(G, h1.i, h2.i)
        return [doc[i] for i in path_indices]
    except Exception:
        return []


# ── Model definition ──────────────────────────────────────────────────────────

class JERModel(nn.Module):
    """
    Joint Entity and Relation Extraction model.

    Forward inputs:
        input_ids, attention_mask     — tokenized text (batch)
        span1_mask, span2_mask        — boolean masks over token positions for
                                        candidate span 1 and 2 (relation head only)
        dep_path_ids                  — (B, path_len) token IDs for dep path

    Forward outputs (dict):
        span_logits  — (B, L, num_bio_labels)
        rel_logits   — (B, num_rel_labels)   [only if span masks provided]
    """

    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-small",
        num_bio_labels: int = len(BIO_LABELS),
        num_rel_labels: int = len(RELATION_TYPES),
        dep_path_vocab_size: int = 512,
        dep_path_embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Head 1 — BIO span tagging
        self.span_head = nn.Linear(hidden, num_bio_labels)

        # Head 2 — Relation classification
        self.dep_embed = nn.Embedding(
            dep_path_vocab_size, dep_path_embed_dim, padding_idx=0
        )
        rel_in = hidden * 2 + dep_path_embed_dim
        self.rel_head = nn.Sequential(
            nn.Linear(rel_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_rel_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span1_mask: Optional[torch.Tensor] = None,
        span2_mask: Optional[torch.Tensor] = None,
        dep_path_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)  # (B, L, H)

        result: dict[str, torch.Tensor] = {
            "span_logits": self.span_head(seq)  # (B, L, num_bio)
        }

        if span1_mask is not None and span2_mask is not None:
            s1 = self._mean_pool(seq, span1_mask)  # (B, H)
            s2 = self._mean_pool(seq, span2_mask)  # (B, H)

            if dep_path_ids is not None:
                dep = self.dep_embed(dep_path_ids).mean(dim=1)  # (B, dep_dim)
            else:
                dep = torch.zeros(
                    s1.size(0), self.dep_embed.embedding_dim, device=s1.device
                )

            rel_in = torch.cat([s1, s2, dep], dim=-1)
            result["rel_logits"] = self.rel_head(rel_in)  # (B, num_rel)

        return result

    @staticmethod
    def _mean_pool(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pool: (B, L, H) × (B, L) → (B, H)."""
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(1) / m.sum(1).clamp(min=1e-9)


# ── Dataset ───────────────────────────────────────────────────────────────────

class JERDataset(torch.utils.data.Dataset):
    """
    Converts weak-labeled or gold-set records to model inputs.

    BIO labels are derived from grounding_quote / causal_relation fields
    by string-matching entity mentions back into the abstract.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        abstract = rec.get("abstract", "")

        enc = self.tokenizer(
            abstract,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offset_mapping = enc["offset_mapping"].squeeze(0)

        bio_labels = self._build_bio(rec, abstract, offset_mapping, len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
        }

    def _build_bio(
        self,
        rec: dict,
        abstract: str,
        offset_mapping: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Build token-level BIO label tensor by locating entity spans in text.
        """
        labels = torch.zeros(seq_len, dtype=torch.long)  # default O

        causal = rec.get("causal_relation") or {}
        loc_dict = rec.get("location") or {}

        entity_map = {
            "HAZARD": causal.get("subject"),
            "IMPACT": causal.get("object"),
            "LOCATION": loc_dict.get("raw"),
            "TIME": (rec.get("time_period") or {}).get("raw"),
            "AFFECTED_GROUP": rec.get("affected_group"),
        }

        offsets = offset_mapping.tolist()

        for etype, mention in entity_map.items():
            if not mention:
                continue
            start_char = abstract.find(mention)
            if start_char == -1:
                continue
            end_char = start_char + len(mention)

            in_span = False
            for i, (tok_s, tok_e) in enumerate(offsets):
                if tok_s == 0 and tok_e == 0:
                    in_span = False
                    continue
                if tok_s >= start_char and tok_e <= end_char:
                    if not in_span:
                        labels[i] = BIO2IDX[f"B-{etype}"]
                        in_span = True
                    else:
                        labels[i] = BIO2IDX[f"I-{etype}"]
                elif tok_s >= end_char:
                    in_span = False

        return labels


# ── Trainer ───────────────────────────────────────────────────────────────────

class JERTrainer:
    def __init__(
        self,
        model: JERModel,
        tokenizer,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def train(
        self,
        train_dataset: JERDataset,
        val_dataset: Optional[JERDataset] = None,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        output_dir: str = "models/jer",
    ) -> None:
        from torch.utils.data import DataLoader

        os.makedirs(output_dir, exist_ok=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bio_labels = batch["bio_labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["span_logits"]  # (B, L, C)

                loss = loss_fn(logits.view(-1, logits.size(-1)), bio_labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg = total_loss / len(train_loader)
            print(f"Epoch {epoch}/{epochs}  train_loss={avg:.4f}", end="")

            if val_dataset:
                val_loss = self._eval_loss(val_dataset, batch_size)
                print(f"  val_loss={val_loss:.4f}", end="")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save(output_dir)
                    print("  ✓ saved", end="")
            print()

        if not val_dataset:
            self._save(output_dir)

    def _eval_loss(self, dataset: JERDataset, batch_size: int) -> float:
        from torch.utils.data import DataLoader

        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        total = 0.0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bio_labels = batch["bio_labels"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["span_logits"]
                total += loss_fn(logits.view(-1, logits.size(-1)), bio_labels.view(-1)).item()

        return total / len(loader)

    def _save(self, output_dir: str) -> None:
        torch.save(self.model.state_dict(), f"{output_dir}/jer_model.pt")
        self.tokenizer.save_pretrained(output_dir)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """Run span extraction on a single text string."""
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        bio_pred = out["span_logits"][0].argmax(-1).cpu().tolist()

        spans = self._decode_bio(text, bio_pred, offsets)
        return {
            "spans": spans,
            "bio_sequence": [IDX2BIO[i] for i in bio_pred],
        }

    @staticmethod
    def _decode_bio(text: str, bio_pred: list[int], offsets: list) -> list[dict]:
        """Convert a BIO label sequence back to character-level span dicts."""
        spans: list[dict] = []
        current: Optional[dict] = None

        for label_id, (s, e) in zip(bio_pred, offsets):
            if s == 0 and e == 0:
                if current:
                    spans.append(current)
                    current = None
                continue

            label = IDX2BIO[label_id]
            if label.startswith("B-"):
                if current:
                    spans.append(current)
                current = {"type": label[2:], "start": s, "end": e, "text": text[s:e]}
            elif label.startswith("I-") and current and current["type"] == label[2:]:
                current["end"] = e
                current["text"] = text[current["start"]:e]
            else:
                if current:
                    spans.append(current)
                    current = None

        if current:
            spans.append(current)

        return spans


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JER model")
    parser.add_argument("--weak", required=True, help="Weak labels JSON")
    parser.add_argument("--gold", required=True, help="Gold set JSON (used as val)")
    parser.add_argument("--out", default="models/jer", help="Output directory")
    parser.add_argument("--backbone", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    with open(args.weak, encoding="utf-8") as f:
        weak_records = json.load(f)
    with open(args.gold, encoding="utf-8") as f:
        gold_records = json.load(f)

    # Attach abstract text to gold records (requires matching to raw abstracts)
    train_ds = JERDataset(weak_records, tokenizer)
    val_ds = JERDataset(gold_records, tokenizer)

    model = JERModel(backbone=args.backbone)
    trainer = JERTrainer(model, tokenizer)
    trainer.train(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        output_dir=args.out,
    )
    print(f"Training complete. Model saved to {args.out}/")
