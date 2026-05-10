"""Collecte de paires DPO depuis les runs de l'agent."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.shared.config import Config
from src.shared.types import DPOPair, IntentJSON

logger = logging.getLogger(__name__)


@dataclass
class _Attempt:
    code: str
    score: float


@dataclass
class RunRecord:
    """Historique d'une requête : commande + tentatives successives."""

    command: str
    intent: IntentJSON
    attempts: list[_Attempt] = field(default_factory=list)

    def add(self, code: str, score: float) -> None:
        self.attempts.append(_Attempt(code=code, score=score))

    def make_dpo_pair(self) -> DPOPair | None:
        """Crée une paire DPO si la séquence contient un échec puis un succès.

        Returns:
            DPOPair ou None si la séquence n'est pas exploitable.
        """
        passing = [a for a in self.attempts if a.score >= 0.8]
        failing = [a for a in self.attempts if a.score < 0.8]

        if not passing or not failing:
            return None

        chosen = max(passing, key=lambda a: a.score)
        rejected = min(failing, key=lambda a: a.score)

        try:
            return DPOPair(
                prompt=f"{self.intent.action} {self.intent.target_name}: {self.intent.description}",
                chosen=chosen.code,
                rejected=rejected.code,
                source="agent_retry",
                chosen_score=chosen.score,
                rejected_score=rejected.score,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except ValueError:
            return None


class DPOCollector:
    """Accumule les paires DPO dans un fichier JSONL mensuel.

    Appelé par l'agent après chaque run. Les paires sont créées automatiquement
    quand une tentative échoue puis réussit (self-reflection produit du signal).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()
        config.dpo_pairs_dir.mkdir(parents=True, exist_ok=True)

    def _current_path(self) -> str:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return str(self.config.dpo_pairs_dir / f"dpo_{month}.jsonl")

    def collect(self, record: RunRecord) -> bool:
        """Tente de créer une paire DPO depuis un RunRecord.

        Args:
            record: Historique de la requête avec toutes ses tentatives.

        Returns:
            True si une paire a été enregistrée.
        """
        pair = record.make_dpo_pair()
        if pair is None:
            return False

        path = self._current_path()
        entry = json.dumps({
            "prompt": pair.prompt, "chosen": pair.chosen, "rejected": pair.rejected,
            "source": pair.source, "chosen_score": pair.chosen_score,
            "rejected_score": pair.rejected_score, "timestamp": pair.timestamp,
        }, ensure_ascii=False)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

        logger.info("Paire DPO enregistrée (chosen=%.2f rejected=%.2f)", pair.chosen_score, pair.rejected_score)
        return True

    def collect_manual(self, prompt: str, chosen: str, rejected: str) -> None:
        """Enregistre une correction manuelle de l'utilisateur comme paire DPO.

        Args:
            prompt: Description de la tâche.
            chosen: Code correct (fourni ou corrigé par l'utilisateur).
            rejected: Code incorrect généré par l'agent.
        """
        pair = DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source="manual_correction",
            chosen_score=1.0,
            rejected_score=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        path = self._current_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "source": pair.source,
                "chosen_score": pair.chosen_score,
                "rejected_score": pair.rejected_score,
                "timestamp": pair.timestamp,
            }, ensure_ascii=False) + "\n")

    def count(self) -> int:
        """Compte le nombre total de paires dans tous les fichiers mensuels."""
        total = 0
        for path in self.config.dpo_pairs_dir.glob("dpo_*.jsonl"):
            total += sum(1 for _ in open(path, encoding="utf-8"))
        return total
