"""Curriculum dataset pour l'entraînement GRPO de l'ECLMCore."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
    tests: list[str] = field(default_factory=list)  # pytest bodies for reward fn


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
                        tests=list(data.get("tests", [])),
                    ))
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning("%s ligne %d ignorée: %s", path.name, i + 1, exc)

    examples.sort(key=lambda e: e.complexity)
    logger.info("Curriculum: %d exemples (max_complexity=%d)", len(examples), max_complexity)
    return examples


def load_from_dpo_pairs(dpo_dir: Path) -> list[CurriculumExample]:
    """Importe les paires DPO comme exemples de curriculum (reward=1.0 pour chosen).

    Args:
        dpo_dir: Répertoire contenant les fichiers dpo_*.jsonl.

    Returns:
        Exemples avec reward=1.0 (chosen) convertis en CREATE_MODULE ops.
    """
    examples: list[CurriculumExample] = []
    for path in sorted(dpo_dir.glob("dpo_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    op = ASTOperation(
                        op_type="CREATE_MODULE",
                        target=str(data.get("prompt", "unknown")[:50]),
                        params={"description": str(data.get("prompt", ""))},
                    )
                    examples.append(CurriculumExample(
                        operation=op,
                        current_code="",
                        target_code=str(data["chosen"]),
                        reward=float(data.get("chosen_score", 1.0)),
                        complexity=2,
                    ))
                except (KeyError, ValueError, TypeError) as exc:
                    logger.debug("Paire DPO ignorée: %s", exc)
    return examples
