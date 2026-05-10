"""Entraînement du C1 ASTPlanner — Flan-T5 seq2seq (~250M params).

Architecture :
- Modèle de base : google/flan-t5-base (250M, déjà instruit sur du code)
- Fine-tuning seq2seq : IntentJSON sérialisé → ASTOperationPlan JSON
- Input  : "plan: action=CREATE target=function name=add desc=Add two integers"
- Output : '[{"op_type":"CREATE_FUNCTION","target":"add","params":{...}}]'

Usage:
    python -m src.planner.train
    python -m src.planner.train --epochs 5 --base-model google/flan-t5-base

Le modèle remplace l'approche Ollama actuelle (< 200ms au lieu de 2-4s).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.planner.dataset import PlannerExample, load_planner_dataset
from src.shared.config import Config
from src.shared.types import ASTOperation

logger = logging.getLogger(__name__)

MIN_EXAMPLES = 500
BASE_MODEL = "google/flan-t5-base"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 512


# ── Sérialisation ─────────────────────────────────────────────────────────────

def intent_to_text(example: PlannerExample) -> str:
    """Sérialise un IntentJSON en texte pour le modèle seq2seq."""
    intent = example.intent
    constraints = "; ".join(intent.constraints) if intent.constraints else "none"
    return (
        f"plan: action={intent.action} target_type={intent.target_type} "
        f"name={intent.target_name} description={intent.description} "
        f"constraints={constraints}"
    )


def plan_to_text(example: PlannerExample) -> str:
    """Sérialise un ASTOperationPlan en JSON compact pour le modèle."""
    ops = [
        {"op_type": op.op_type, "target": op.target, "params": dict(op.params)}
        for op in example.plan.operations
    ]
    return json.dumps(ops, ensure_ascii=False, separators=(",", ":"))


def text_to_ops(text: str) -> list[ASTOperation]:
    """Désérialise le JSON produit par le modèle en liste d'ASTOperation."""
    try:
        data = json.loads(text)
        return [ASTOperation(op_type=d["op_type"], target=d["target"], params=d.get("params", {}))
                for d in data]
    except (json.JSONDecodeError, KeyError, ValueError):
        return []


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    config: Config,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    base_model: str = BASE_MODEL,
) -> None:
    """Fine-tune Flan-T5 sur les paires (IntentJSON → ASTOperationPlan).

    Args:
        config: Configuration du projet.
        epochs: Nombre d'époques.
        batch_size: Taille des batchs.
        lr: Learning rate (plus élevé que BERT car T5 est déjà instruit).
        base_model: Modèle HuggingFace de base.

    Raises:
        ValueError: Si dataset insuffisant.
        FileNotFoundError: Si planner_train.jsonl introuvable.
        ImportError: Si transformers non installé.
    """
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from transformers import DataCollatorForSeq2Seq
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    except ImportError as exc:
        raise ImportError("Installez: pip install torch transformers datasets") from exc

    examples = load_planner_dataset(config.data_dir)
    if len(examples) < MIN_EXAMPLES:
        raise ValueError(
            f"Seulement {len(examples)} exemples — minimum {MIN_EXAMPLES} requis.\n"
            f"Lancez: python scripts/bootstrap_dataset.py"
        )

    logger.info("Fine-tuning %s sur %d exemples (planner)...", base_model, len(examples))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    inputs = [intent_to_text(ex) for ex in examples]
    targets = [plan_to_text(ex) for ex in examples]

    # ── Dataset HuggingFace ──────────────────────────────────────────────────
    from datasets import Dataset as HFDataset  # type: ignore[import-untyped]

    raw = HFDataset.from_dict({"input": inputs, "target": targets})

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LEN, truncation=True, padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=MAX_TARGET_LEN, truncation=True, padding="max_length",
            )
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in labels["input_ids"]
        ]
        return model_inputs

    tokenized = raw.map(tokenize, batched=True, remove_columns=["input", "target"])
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    out_dir = config.planner_model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=__import__("torch").cuda.is_bf16_supported() if hasattr(__import__("torch").cuda, "is_bf16_supported") else False,
        report_to="none",
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    logger.info("Planner sauvegardé: %s", out_dir / "final")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraîne le ASTPlanner (Flan-T5 seq2seq)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(
        Config(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_model=args.base_model,
    )
