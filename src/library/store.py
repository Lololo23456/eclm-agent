"""ChromaDB wrapper pour la Primitive Library."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection

from src.library.primitive import Primitive
from src.shared.config import Config

logger = logging.getLogger(__name__)


class PrimitiveStore:
    """Stocke et récupère des primitives vérifiées via ChromaDB.

    Toute primitive ajoutée doit avoir passé la vérification complète
    (score >= config.min_verification_score). Ne jamais appeler add()
    directement — passer par /add-primitive qui enforce la validation.
    """

    _COLLECTION = "primitives"

    def __init__(self, config: Config) -> None:
        self.config = config
        persist_dir = config.primitives_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col: Collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def add(self, primitive: Primitive, embedding: list[float]) -> None:
        """Ajoute une primitive vérifiée dans le store.

        Args:
            primitive: Primitive à stocker (doit être vérifiée).
            embedding: Vecteur dense (intent vector) de dimension fixe.

        Raises:
            ValueError: Si la primitive a un score insuffisant.
        """
        if primitive.score < self.config.min_verification_score:
            raise ValueError(
                f"Score {primitive.score:.2f} < seuil {self.config.min_verification_score}"
            )
        self._col.upsert(
            ids=[primitive.id],
            documents=[primitive.description],
            embeddings=[embedding],
            metadatas=[primitive.to_metadata()],
        )
        logger.info("Primitive ajoutée: %s (%s)", primitive.id[:8], primitive.domain)

    def delete(self, primitive_id: str) -> None:
        """Supprime une primitive par son id.

        Args:
            primitive_id: UUID de la primitive à supprimer.
        """
        self._col.delete(ids=[primitive_id])
        logger.info("Primitive supprimée: %s", primitive_id[:8])

    def update_usage(self, primitive_id: str) -> None:
        """Incrémente usage_count après récupération réussie.

        Args:
            primitive_id: UUID de la primitive utilisée.
        """
        results = self._col.get(ids=[primitive_id], include=["metadatas", "documents"])
        if not results["ids"]:
            return
        meta: dict[str, Any] = dict(results["metadatas"][0])
        meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
        self._col.update(ids=[primitive_id], metadatas=[meta])

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Primitive]:
        """Retourne les top_k primitives les plus proches par similarité cosine.

        Args:
            query_embedding: Vecteur de requête (même espace que les embeddings stockés).
            top_k: Nombre maximum de résultats.

        Returns:
            Liste de Primitive triée par pertinence décroissante.
        """
        n = self._col.count()
        if n == 0:
            return []
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, n),
            include=["documents", "metadatas"],
        )
        primitives: list[Primitive] = []
        for pid, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            primitives.append(Primitive.from_metadata(pid, doc, meta))
        return primitives

    def get(self, primitive_id: str) -> Primitive | None:
        """Récupère une primitive par son id exact.

        Args:
            primitive_id: UUID de la primitive.

        Returns:
            Primitive ou None si introuvable.
        """
        results = self._col.get(
            ids=[primitive_id],
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return None
        return Primitive.from_metadata(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        )

    def count(self) -> int:
        """Retourne le nombre de primitives stockées."""
        return self._col.count()

    def list_by_domain(self, domain: str) -> list[Primitive]:
        """Retourne toutes les primitives d'un domaine donné.

        Args:
            domain: Domaine cible (ex: "parsing", "http", "auth").

        Returns:
            Liste de Primitive du domaine.
        """
        results = self._col.get(
            where={"domain": domain},
            include=["documents", "metadatas"],
        )
        return [
            Primitive.from_metadata(pid, doc, meta)
            for pid, doc, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]
