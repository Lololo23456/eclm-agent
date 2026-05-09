"""Fine-tuning CamemBERT sur le dataset d'intentions accumulé."""
from __future__ import annotations

import logging
from pathlib import Path

from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_EXAMPLES_FOR_TRAINING = 500


def train(config: Config) -> None:
    """Lance le fine-tuning CamemBERT quand assez d'exemples sont disponibles.

    Args:
        config: Configuration du projet.

    Raises:
        ValueError: Si le dataset est insuffisant (< MIN_EXAMPLES_FOR_TRAINING).
        FileNotFoundError: Si intent_raw.jsonl est introuvable.
    """
    data_path = config.data_dir / "training" / "intent" / "intent_raw.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {data_path}")

    n = sum(1 for _ in data_path.open(encoding="utf-8"))
    if n < MIN_EXAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Seulement {n} exemples — minimum {MIN_EXAMPLES_FOR_TRAINING} requis. "
            f"Continuez à utiliser l'agent pour accumuler des données."
        )

    logger.info("Fine-tuning CamemBERT sur %d exemples...", n)
    # TODO: charger camembert-base, tokenizer, DataLoader, Trainer HuggingFace
    raise NotImplementedError


if __name__ == "__main__":
    train(Config())
