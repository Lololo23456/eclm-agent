"""Curriculum dataset pour l'entraînement RL de l'ECLMCore."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.shared.types import ASTOperation

logger = logging.getLogger(__name__)


@dataclass
class CurriculumExample:
    """Exemple d'entraînement : opération → code target + reward signal."""

    operation: ASTOperation
    current_code: str
    target_code: str
    reward: float
    complexity: int  # 1=simple, 5=complexe — curriculum ordonné par complexité


def load_curriculum(data_dir: Path, max_complexity: int = 5) -> list[CurriculumExample]:
    """Charge les exemples du curriculum filtrés par complexité max.

    Args:
        data_dir: Répertoire data/ du projet.
        max_complexity: Complexité maximale à inclure (curriculum learning).

    Returns:
        Exemples triés par complexité croissante.

    Raises:
        FileNotFoundError: Si aucun fichier curriculum n'est trouvé.
    """
    curriculum_dir = data_dir / "training" / "eclm"
    files = sorted(curriculum_dir.glob("curriculum_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier curriculum dans {curriculum_dir}")

    examples: list[CurriculumExample] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    complexity = int(data.get("complexity", 1))
                    if complexity > max_complexity:
                        continue
                    op = ASTOperation(**data["operation"])
                    examples.append(CurriculumExample(
                        operation=op,
                        current_code=str(data["current_code"]),
                        target_code=str(data["target_code"]),
                        reward=float(data["reward"]),
                        complexity=complexity,
                    ))
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning("%s ligne %d ignorée: %s", path.name, i + 1, exc)

    examples.sort(key=lambda e: e.complexity)
    logger.info("Curriculum: %d exemples (max_complexity=%d)", len(examples), max_complexity)
    return examples
