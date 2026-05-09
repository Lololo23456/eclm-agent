"""DPO fine-tuning mensuel de l'ECLMCore."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_PAIRS_FOR_FINETUNE = 100


def load_dpo_pairs(config: Config) -> list[dict[str, str]]:
    """Charge toutes les paires DPO disponibles.

    Args:
        config: Configuration du projet.

    Returns:
        Liste de dicts {prompt, chosen, rejected}.
    """
    pairs: list[dict[str, str]] = []
    for path in sorted(config.dpo_pairs_dir.glob("dpo_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    return pairs


def run_finetune(config: Config) -> None:
    """Lance le DPO fine-tuning sur le modèle courant.

    Args:
        config: Configuration du projet.

    Raises:
        ValueError: Si pas assez de paires DPO disponibles.
        FileNotFoundError: Si le modèle de base est introuvable.
    """
    pairs = load_dpo_pairs(config)
    n = len(pairs)

    if n < MIN_PAIRS_FOR_FINETUNE:
        raise ValueError(
            f"Seulement {n} paires DPO — minimum {MIN_PAIRS_FOR_FINETUNE} requis. "
            f"Continuez à utiliser l'agent pour accumuler du signal."
        )

    model_dir = config.eclm_model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Modèle ECLM introuvable: {model_dir}")

    logger.info("DPO fine-tuning sur %d paires...", n)
    # TODO: charger le modèle via transformers/peft, instancier trl.DPOTrainer,
    #       lancer l'entraînement, sauvegarder dans model_dir / "dpo_latest"
    raise NotImplementedError


if __name__ == "__main__":
    run_finetune(Config())
