"""Self-play adversarial — Generator vs Critic pour DPO automatique."""
from __future__ import annotations

import logging

from src.improvement.dpo_collector import DPOCollector, RunRecord
from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)


class AdversarialLoop:
    """Generator vs Critic : génère des paires DPO sans intervention humaine.

    Stratégie :
    1. Generator produit plusieurs candidats pour une tâche.
    2. Critic (verifier) les score.
    3. Les paires (meilleur, pire) alimentent le DPO collector.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._collector = DPOCollector(config)

    def run_episode(self, intent: IntentJSON, candidates: list[tuple[str, float]]) -> bool:
        """Crée une paire DPO depuis les candidats d'un épisode.

        Args:
            intent: Intention de la tâche.
            candidates: Liste de (code, score) triée par score décroissant.

        Returns:
            True si une paire a été enregistrée.
        """
        if len(candidates) < 2:
            return False

        command = f"{intent.action} {intent.target_name}: {intent.description}"
        record = RunRecord(command=command, intent=intent)
        for code, score in candidates:
            record.add(code, score)

        return self._collector.collect(record)

    def run_batch(self, tasks: list[dict[str, object]]) -> int:
        """Lance un batch de self-play sur une liste de tâches.

        Args:
            tasks: Liste de dicts {intent, candidates}.

        Returns:
            Nombre de paires DPO créées.
        """
        # TODO: charger l'agent, générer des candidats pour chaque tâche,
        #       appeler run_episode() pour chacun
        raise NotImplementedError
