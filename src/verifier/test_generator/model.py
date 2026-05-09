"""TestGenerator — modèle ~150M params, isolé de l'ECLM."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.shared.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TestGeneratorOutput:
    """Tests générés par le TestGenerator."""

    tests: list[str] = field(default_factory=list)
    confidence: float = 0.0


class TestGenerator:
    """Génère des tests d'implémentation à partir du code source.

    Isolé totalement des candidats ECLM pour éviter le cercle vicieux
    où un code faux serait validé par des tests générés depuis ce même code.
    Ne reçoit JAMAIS les candidats ECLM avant d'avoir généré ses tests.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._loaded = False

    def generate(self, code: str) -> TestGeneratorOutput:
        """Génère des tests pytest pour le code source donné.

        Args:
            code: Code Python source (jamais un candidat ECLM non vérifié).

        Returns:
            TestGeneratorOutput avec les tests et un score de confiance.

        Raises:
            RuntimeError: Si le modèle n'est pas chargé.
        """
        if not self._loaded:
            raise RuntimeError("Appeler load() avant generate()")
        raise NotImplementedError

    def load(self) -> None:
        """Charge le modèle TestGenerator depuis le disque.

        Raises:
            FileNotFoundError: Si le modèle n'existe pas dans config.testgen_model_dir.
        """
        model_dir = self.config.testgen_model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Modèle TestGenerator introuvable: {model_dir}")
        # TODO: charger le modèle HuggingFace depuis model_dir
        self._loaded = True
