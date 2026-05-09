"""Entraînement RL de l'ECLMCore — objectif maximize(execution_reward)."""
from __future__ import annotations

import logging

from src.eclm.dataset import load_curriculum
from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_EXAMPLES = 1000


def train(config: Config, max_complexity: int = 5) -> None:
    """Lance l'entraînement RL de l'ECLMCore par curriculum.

    Le signal de reward provient exclusivement de l'exécution :
    - reward=1.0 si tous les tests passent
    - reward=0.5 si tests partiels (self-reflection avec trace d'erreur)
    - reward=0.0 si erreur avant les tests

    Args:
        config: Configuration du projet.
        max_complexity: Complexité max du curriculum à cette étape.

    Raises:
        ValueError: Si pas assez d'exemples de curriculum.
        FileNotFoundError: Si le curriculum est introuvable.
    """
    examples = load_curriculum(config.data_dir, max_complexity=max_complexity)
    if len(examples) < MIN_EXAMPLES:
        raise ValueError(
            f"Seulement {len(examples)} exemples — minimum {MIN_EXAMPLES} requis. "
            f"Générez le curriculum avec scripts/build_curriculum.py."
        )

    logger.info("Entraînement ECLM sur %d exemples (max_complexity=%d)...", len(examples), max_complexity)
    # TODO:
    # 1. Charger le modèle de base (CodeT5+ ou similaire ~500M)
    # 2. Pour chaque batch : générer k candidats, exécuter dans sandbox, calculer reward
    # 3. PPO/GRPO update sur les candidats avec reward signal
    # 4. Sauvegarder checkpoint dans config.eclm_model_dir
    raise NotImplementedError


if __name__ == "__main__":
    train(Config())
