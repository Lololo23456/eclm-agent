"""Fine-tuning CamemBERT sur le dataset d'intentions accumulé.

Architecture :
- CamemBERT base (110M) comme encodeur
- Deux têtes de classification indépendantes :
    action_head      → 14 classes (MODIFY, CREATE, DELETE, ...)
    target_type_head → 6 classes (function, class, file, ...)
- Multi-task loss = CrossEntropy(action) + CrossEntropy(target_type)
- Confiance = max(softmax(action)) — utilisée dans IntentExtractor.extract()

Usage:
    python -m src.intent.train
    python -m src.intent.train --epochs 5 --batch-size 16

Le modèle sauvegardé remplace Ollama pour C0 (classification instantanée < 50ms).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.config import Config
from src.shared.types import VALID_ACTIONS, VALID_TARGET_TYPES

logger = logging.getLogger(__name__)

MIN_EXAMPLES_FOR_TRAINING = 500

# Ordre déterministe des classes (doit correspondre à l'inférence)
ACTION_CLASSES: list[str] = sorted(VALID_ACTIONS)
TARGET_TYPE_CLASSES: list[str] = sorted(VALID_TARGET_TYPES)


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class IntentExample:
    command: str
    action: str
    target_type: str


def load_intent_examples(data_path: Path) -> list[IntentExample]:
    examples: list[IntentExample] = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                intent = d["intent"]
                action = str(intent["action"])
                ttype = str(intent["target_type"])
                if action not in VALID_ACTIONS or ttype not in VALID_TARGET_TYPES:
                    continue
                examples.append(IntentExample(
                    command=str(d["command"]),
                    action=action,
                    target_type=ttype,
                ))
            except (KeyError, ValueError) as exc:
                logger.debug("Ligne %d ignorée: %s", i + 1, exc)
    return examples


# ── Model architecture ────────────────────────────────────────────────────────

def _build_model(base_model_name: str, n_actions: int, n_target_types: int) -> Any:
    """Construit CamemBERT + deux têtes de classification."""
    import torch
    import torch.nn as nn
    from transformers import CamembertModel

    class IntentClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = CamembertModel.from_pretrained(base_model_name)
            hidden = self.encoder.config.hidden_size  # 768
            self.dropout = nn.Dropout(0.1)
            self.action_head = nn.Linear(hidden, n_actions)
            self.target_type_head = nn.Linear(hidden, n_target_types)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            action_labels: Any = None,
            target_type_labels: Any = None,
        ) -> dict[str, Any]:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.last_hidden_state[:, 0])  # [CLS]
            action_logits = self.action_head(cls)
            tt_logits = self.target_type_head(cls)

            loss = None
            if action_labels is not None and target_type_labels is not None:
                ce = nn.CrossEntropyLoss()
                loss = ce(action_logits, action_labels) + ce(tt_logits, target_type_labels)

            return {"loss": loss, "action_logits": action_logits, "target_type_logits": tt_logits}

    return IntentClassifier()


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    config: Config,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    base_model: str = "camembert-base",
) -> None:
    """Fine-tune CamemBERT sur le dataset d'intentions.

    Args:
        config: Configuration du projet.
        epochs: Nombre d'époques d'entraînement.
        batch_size: Taille des batchs.
        lr: Learning rate.
        base_model: Modèle de base HuggingFace (camembert-base recommandé).

    Raises:
        ValueError: Si dataset insuffisant (< MIN_EXAMPLES_FOR_TRAINING).
        FileNotFoundError: Si intent_raw.jsonl est introuvable.
        ImportError: Si torch/transformers non installés.
    """
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset, random_split
        from transformers import CamembertTokenizerFast
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("Installez: pip install torch transformers") from exc

    data_path = config.data_dir / "training" / "intent" / "intent_raw.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {data_path}\n"
            f"Lancez d'abord: python scripts/bootstrap_dataset.py"
        )

    examples = load_intent_examples(data_path)
    if len(examples) < MIN_EXAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Seulement {len(examples)} exemples — minimum {MIN_EXAMPLES_FOR_TRAINING} requis.\n"
            f"Continuez à utiliser l'agent ou lancez scripts/bootstrap_dataset.py"
        )

    logger.info("Fine-tuning %s sur %d exemples...", base_model, len(examples))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = CamembertTokenizerFast.from_pretrained(base_model)

    # Tokenization
    action_to_idx = {a: i for i, a in enumerate(ACTION_CLASSES)}
    tt_to_idx = {t: i for i, t in enumerate(TARGET_TYPE_CLASSES)}

    encodings = tokenizer(
        [ex.command for ex in examples],
        truncation=True, padding=True, max_length=128, return_tensors="pt",
    )
    action_labels = torch.tensor([action_to_idx[ex.action] for ex in examples])
    tt_labels = torch.tensor([tt_to_idx[ex.target_type] for ex in examples])

    dataset = TensorDataset(
        encodings["input_ids"], encodings["attention_mask"],
        action_labels, tt_labels,
    )
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = _build_model(base_model, len(ACTION_CLASSES), len(TARGET_TYPE_CLASSES))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    out_dir = config.intent_model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids, mask, act_lbl, tt_lbl = (t.to(device) for t in batch)
            optimizer.zero_grad()
            out = model(ids, mask, action_labels=act_lbl, target_type_labels=tt_lbl)
            loss: Any = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # ── Eval ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        action_correct = 0
        tt_correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, act_lbl, tt_lbl = (t.to(device) for t in batch)
                out = model(ids, mask, action_labels=act_lbl, target_type_labels=tt_lbl)
                val_loss += out["loss"].item()
                action_correct += (out["action_logits"].argmax(1) == act_lbl).sum().item()
                tt_correct += (out["target_type_logits"].argmax(1) == tt_lbl).sum().item()
                total += act_lbl.size(0)

        action_acc = action_correct / total if total else 0
        tt_acc = tt_correct / total if total else 0
        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f action_acc=%.1f%% tt_acc=%.1f%%",
            epoch + 1, epochs,
            train_loss / len(train_loader),
            val_loss / len(val_loader),
            action_acc * 100, tt_acc * 100,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "model.pt")
            logger.info("  ↳ Meilleur modèle sauvegardé (val_loss=%.4f)", best_val_loss)

    # Sauvegarder les métadonnées de classe
    import json as _json
    (out_dir / "label_map.json").write_text(
        _json.dumps({
            "action_classes": ACTION_CLASSES,
            "target_type_classes": TARGET_TYPE_CLASSES,
            "base_model": base_model,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Modèle final sauvegardé: %s", out_dir)
    logger.info("Pour utiliser: ECLM_INTENT_MODEL_DIR=%s", out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune CamemBERT pour l'extraction d'intentions")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--base-model", default="camembert-base")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(
        Config(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_model=args.base_model,
    )
