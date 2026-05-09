"""Recherche sémantique de primitives via embedding de l'intention."""
from __future__ import annotations

import hashlib
import logging

from src.library.primitive import Primitive
from src.library.store import PrimitiveStore
from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)


def _intent_to_embedding(intent: IntentJSON) -> list[float]:
    """Embedding déterministe de l'IntentJSON (placeholder pre-modèle).

    En production, remplacer par le vecteur produit par C0 CamemBERT.
    Cette version hash le texte pour produire un vecteur reproductible.

    Args:
        intent: IntentJSON à encoder.

    Returns:
        Vecteur de dimension 384 (compatible sentence-transformers).
    """
    text = f"{intent.action} {intent.target_type} {intent.target_name} {intent.description}"
    digest = hashlib.sha256(text.encode()).digest()
    # Répète le digest pour remplir 384 dimensions
    raw = list(digest * 12)[:384]
    total = sum(raw) or 1.0
    return [v / total for v in raw]


class PrimitiveRetriever:
    """Récupère les primitives pertinentes pour une intention donnée."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = PrimitiveStore(config)

    def retrieve(self, intent: IntentJSON, top_k: int = 3) -> list[Primitive]:
        """Retourne les top_k primitives les plus pertinentes pour l'intent.

        Args:
            intent: Intention extraite par C0.
            top_k: Nombre maximum de primitives à retourner.

        Returns:
            Liste de Primitive triée par pertinence, vide si aucune trouvée.
        """
        embedding = _intent_to_embedding(intent)
        primitives = self.store.search(embedding, top_k=top_k)
        if primitives:
            for p in primitives:
                self.store.update_usage(p.id)
            logger.info("Primitives récupérées: %d pour %s", len(primitives), intent.action)
        return primitives
