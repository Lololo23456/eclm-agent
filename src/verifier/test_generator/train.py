"""Script d'entraînement du TestGenerator (~150M params)."""
from __future__ import annotations

import logging
from pathlib import Path

from src.shared.config import Config

logger = logging.getLogger(__name__)


def train(config: Config, train_data_path: Path, val_data_path: Path) -> None:
    """Lance l'entraînement du TestGenerator sur paires (code → tests).

    Args:
        config: Configuration du projet.
        train_data_path: Chemin vers testgen_train.jsonl.
        val_data_path: Chemin vers testgen_val.jsonl.

    Raises:
        FileNotFoundError: Si les fichiers de données sont introuvables.
    """
    if not train_data_path.exists():
        raise FileNotFoundError(f"Données d'entraînement introuvables: {train_data_path}")
    if not val_data_path.exists():
        raise FileNotFoundError(f"Données de validation introuvables: {val_data_path}")
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    cfg = Config()
    train(
        config=cfg,
        train_data_path=cfg.data_dir / "training" / "test_generator" / "testgen_train.jsonl",
        val_data_path=cfg.data_dir / "training" / "test_generator" / "testgen_val.jsonl",
    )
