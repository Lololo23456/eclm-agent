"""Entraînement du TestGenerator — Flan-T5 seq2seq (code → tests pytest).

Architecture :
- Modèle de base : google/flan-t5-base (250M)
- Fine-tuning seq2seq : code Python → fonctions pytest standalone
- Input  : "generate tests: def add(a, b): return a + b"
- Output : "def test_add(): assert add(1, 2) == 3\n\ndef test_add_zero(): ..."

Format du dataset testgen_*.jsonl :
{
  "code": "def add(a, b):\\n    return a + b",
  "tests": "def test_add():\\n    assert add(1, 2) == 3"
}

Usage:
    python -m src.verifier.test_generator.train
    python -m src.verifier.test_generator.train --epochs 5

ISOLÉ de l'ECLM Core : ce modèle ne voit jamais les candidats ECLM.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_EXAMPLES = 500
BASE_MODEL = "google/flan-t5-base"
MAX_INPUT_LEN = 512   # code peut être long
MAX_TARGET_LEN = 256  # tests sont plus courts
_PROMPT_PREFIX = "generate pytest tests: "


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_dataset(data_path: Path) -> tuple[list[str], list[str]]:
    """Charge un fichier testgen_*.jsonl → (inputs, targets)."""
    inputs: list[str] = []
    targets: list[str] = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                code = str(d["code"]).strip()
                tests = str(d["tests"]).strip()
                if not code or not tests:
                    continue
                inputs.append(_PROMPT_PREFIX + code)
                targets.append(tests)
            except (KeyError, ValueError) as exc:
                logger.debug("Ligne %d ignorée: %s", i + 1, exc)
    return inputs, targets


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    config: Config,
    train_data_path: Path | None = None,
    val_data_path: Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    base_model: str = BASE_MODEL,
) -> None:
    """Fine-tune Flan-T5 pour la génération de tests pytest.

    Args:
        config: Configuration du projet.
        train_data_path: Chemin vers testgen_train.jsonl (None → défaut config).
        val_data_path: Chemin vers testgen_val.jsonl (None → split auto 90/10).
        epochs: Nombre d'époques.
        batch_size: Taille des batchs.
        lr: Learning rate.
        base_model: Modèle HuggingFace de base.

    Raises:
        FileNotFoundError: Si les données d'entraînement sont introuvables.
        ValueError: Si dataset insuffisant.
        ImportError: Si transformers non installé.
    """
    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        import torch
    except ImportError as exc:
        raise ImportError("Installez: pip install torch transformers datasets") from exc

    testgen_dir = config.data_dir / "training" / "test_generator"

    if train_data_path is None:
        train_data_path = testgen_dir / "testgen_train.jsonl"
    if val_data_path is None:
        val_data_path = testgen_dir / "testgen_val.jsonl"

    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Données introuvables: {train_data_path}\n"
            f"Générez-les avec: python scripts/bootstrap_dataset.py --mode testgen"
        )

    train_inputs, train_targets = _load_dataset(train_data_path)
    if len(train_inputs) < MIN_EXAMPLES:
        raise ValueError(
            f"Seulement {len(train_inputs)} exemples — minimum {MIN_EXAMPLES} requis."
        )

    logger.info("Fine-tuning %s (TestGenerator) sur %d exemples...", base_model, len(train_inputs))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    # ── Dataset HuggingFace ──────────────────────────────────────────────────
    from datasets import Dataset as HFDataset  # type: ignore[import-untyped]

    # Merge train + val (ou split auto si val absent)
    if val_data_path.exists():
        val_inputs, val_targets = _load_dataset(val_data_path)
    else:
        n_val = max(1, int(0.1 * len(train_inputs)))
        val_inputs = train_inputs[-n_val:]
        val_targets = train_targets[-n_val:]
        train_inputs = train_inputs[:-n_val]
        train_targets = train_targets[:-n_val]

    def _make_hf_dataset(inp: list[str], tgt: list[str]) -> Any:
        return HFDataset.from_dict({"input": inp, "target": tgt})

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

    train_ds = _make_hf_dataset(train_inputs, train_targets).map(
        tokenize, batched=True, remove_columns=["input", "target"]
    )
    val_ds = _make_hf_dataset(val_inputs, val_targets).map(
        tokenize, batched=True, remove_columns=["input", "target"]
    )

    out_dir = config.testgen_model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
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
        bf16=bf16_supported,
        report_to="none",
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    logger.info("TestGenerator sauvegardé: %s", out_dir / "final")

    # Note : le modèle sauvegardé peut être chargé dans TestGenerator
    # en remplaçant self._call_ollama() par une inférence locale Flan-T5.
    logger.info(
        "Pour activer le modèle local, setez ECLM_TESTGEN_MODEL_DIR=%s",
        out_dir / "final",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraîne le TestGenerator (Flan-T5 seq2seq)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config()
    train(
        cfg,
        train_data_path=cfg.data_dir / "training" / "test_generator" / "testgen_train.jsonl",
        val_data_path=cfg.data_dir / "training" / "test_generator" / "testgen_val.jsonl",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_model=args.base_model,
    )
