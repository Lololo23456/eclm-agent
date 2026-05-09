"""Logging des extractions pour constituer le dataset CamemBERT."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)


class IntentDataLogger:
    """Enregistre chaque paire (commande, IntentJSON) dans un JSONL.

    Ces données serviront au fine-tuning CamemBERT une fois ~2000 exemples
    accumulés. Objectif : remplacer Ollama par un modèle 110M 100% local.
    """

    def __init__(self, config: Config) -> None:
        log_dir = config.data_dir / "training" / "intent"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / "intent_raw.jsonl"

    def log(self, command: str, intent: IntentJSON, validated: bool = False) -> None:
        """Enregistre une paire (commande → intent) dans le JSONL.

        Args:
            command: Commande originale en français.
            intent: IntentJSON extrait (validé ou non).
            validated: True si l'utilisateur a confirmé le résultat.
        """
        record = {
            "command": command,
            "intent": {
                "action": intent.action,
                "target_type": intent.target_type,
                "target_name": intent.target_name,
                "target_file": intent.target_file,
                "description": intent.description,
                "constraints": list(intent.constraints),
                "confidence": intent.confidence,
            },
            "validated": validated,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def count(self) -> int:
        """Retourne le nombre d'exemples enregistrés."""
        if not self._path.exists():
            return 0
        return sum(1 for _ in self._path.open(encoding="utf-8"))
