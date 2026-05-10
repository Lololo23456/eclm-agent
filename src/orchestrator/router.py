"""ModelRouter — sélectionne le modèle optimal selon la complexité de l'opération."""
from __future__ import annotations

import logging

from src.shared.config import Config

logger = logging.getLogger(__name__)

# Ops dont le résultat est déterministe ou quasi-déterministe → fast model suffisant
_FAST_OPS = frozenset({
    "ADD_DOCSTRING", "ADD_RETURN_TYPE", "ADD_DECORATOR", "ADD_IMPORT",
    "RENAME_SYMBOL", "DELETE_NODE", "REMOVE_PARAM", "ADD_PARAM",
})

# Ops génératives complexes → strong model pour la qualité
_STRONG_OPS = frozenset({
    "CREATE_CLASS", "CREATE_FUNCTION", "CREATE_MODULE", "EXTRACT_FUNCTION",
    "MERGE", "SPLIT", "MODIFY_BODY",
})


class ModelRouter:
    """Route les requêtes vers le modèle fast (7B) ou strong (32B).

    Principe : ne sortir le 32B que quand la qualité justifie la latence.
    Sur M3 Air (fast=strong=7B), le routage est transparent.
    Sur serveur 4090, fast=7B (~2s) et strong=32B (~15s).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._same_model = config.fast_model == config.strong_model

    def for_operation(self, op_type: str, complexity: str = "medium") -> str:
        """Retourne le modèle optimal pour une opération AST.

        Args:
            op_type: Type d'opération AST (ex: "CREATE_CLASS").
            complexity: Complexité estimée par l'ArchitectAgent ("low"|"medium"|"high").

        Returns:
            Nom du modèle Ollama à utiliser.
        """
        if self._same_model:
            return self.config.fast_model

        if op_type in _FAST_OPS or complexity == "low":
            logger.debug("Router → fast (%s) pour %s", self.config.fast_model, op_type)
            return self.config.fast_model

        if op_type in _STRONG_OPS or complexity == "high":
            logger.debug("Router → strong (%s) pour %s", self.config.strong_model, op_type)
            return self.config.strong_model

        # medium complexity → fast suffit pour les ops standard
        return self.config.fast_model

    def for_planning(self) -> str:
        """Planning = toujours le modèle fort (qualité du DAG critique)."""
        return self.config.strong_model

    def for_intent(self) -> str:
        """Extraction d'intent = fast (classification simple)."""
        return self.config.fast_model
