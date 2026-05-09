"""Dataset pour l'entraînement du C1 ASTPlanner."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.shared.types import ASTOperation, ASTOperationPlan, IntentJSON

logger = logging.getLogger(__name__)


@dataclass
class PlannerExample:
    """Exemple d'entraînement : (intent, plan)."""

    intent: IntentJSON
    plan: ASTOperationPlan
    source_file: str


def load_planner_dataset(data_dir: Path) -> list[PlannerExample]:
    """Charge le dataset du planner depuis planner_train.jsonl.

    Args:
        data_dir: Répertoire data/ du projet.

    Returns:
        Liste d'exemples d'entraînement.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
    """
    path = data_dir / "training" / "planner" / "planner_train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset planner introuvable: {path}")

    examples: list[PlannerExample] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                intent = IntentJSON(**data["intent"])
                ops = tuple(ASTOperation(**op) for op in data["plan"]["operations"])
                plan = ASTOperationPlan(
                    operations=ops,
                    intent=intent,
                    estimated_complexity=data["plan"].get("estimated_complexity", 1),
                )
                examples.append(PlannerExample(intent=intent, plan=plan, source_file=str(path)))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Ligne %d ignorée: %s", i + 1, exc)

    logger.info("Dataset planner: %d exemples", len(examples))
    return examples
