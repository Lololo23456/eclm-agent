"""Entraînement du C1 ASTPlanner (~200M params seq2seq)."""
from __future__ import annotations

import logging

from src.planner.dataset import load_planner_dataset
from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_EXAMPLES = 500


def train(config: Config) -> None:
    """Lance l'entraînement du ASTPlanner.

    Args:
        config: Configuration du projet.

    Raises:
        ValueError: Si pas assez d'exemples.
        FileNotFoundError: Si le dataset est introuvable.
    """
    examples = load_planner_dataset(config.data_dir)
    if len(examples) < MIN_EXAMPLES:
        raise ValueError(
            f"Seulement {len(examples)} exemples — minimum {MIN_EXAMPLES} requis."
        )
    # TODO: tokenizer, DataLoader, T5/BART seq2seq, Trainer HuggingFace
    raise NotImplementedError


if __name__ == "__main__":
    train(Config())
