"""GraphRAG — index ChromaDB du codebase, mis à jour incrémentalement."""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb import Collection

from src.orchestrator.context import ASTContext, CodeChunk, build_context, extract_chunks
from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)


class CodebaseIndex:
    """Index vectoriel du codebase courant (ChromaDB, AST-aware).

    Mise à jour incrémentale : seuls les fichiers modifiés sont ré-indexés.
    Utilise les descriptions textuelles des chunks comme documents —
    l'embedding est géré par ChromaDB (embedding function par défaut).
    """

    _COLLECTION = "codebase"

    def __init__(self, config: Config, root_dir: Path) -> None:
        self.config = config
        self.root_dir = root_dir
        index_dir = config.data_dir / "codebase_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_dir))
        self._col: Collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def index_file(self, file_path: Path) -> int:
        """Indexe ou ré-indexe un fichier Python.

        Args:
            file_path: Chemin absolu du fichier .py.

        Returns:
            Nombre de chunks indexés.
        """
        chunks = extract_chunks(file_path)
        if not chunks:
            return 0

        ids = [f"{file_path}::{c.start_line}" for c in chunks]
        # Supprime les anciens chunks de ce fichier
        existing = self._col.get(where={"file_path": str(file_path)})
        if existing["ids"]:
            self._col.delete(ids=existing["ids"])

        self._col.upsert(
            ids=ids,
            documents=[c.source for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "node_type": c.node_type,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in chunks
            ],
        )
        logger.debug("Indexé %d chunks: %s", len(chunks), file_path.name)
        return len(chunks)

    def index_project(self) -> int:
        """Indexe tous les fichiers .py du projet.

        Returns:
            Nombre total de chunks indexés.
        """
        total = 0
        for py_file in self.root_dir.rglob("*.py"):
            if any(part.startswith(".") for part in py_file.parts):
                continue
            total += self.index_file(py_file)
        logger.info("Projet indexé: %d chunks au total", total)
        return total

    def get_context(self, intent: IntentJSON, top_k: int = 8) -> ASTContext:
        """Récupère le contexte AST pertinent pour une intention.

        Combine la recherche sémantique ChromaDB avec le filtrage par nom.

        Args:
            intent: IntentJSON produit par C0.
            top_k: Nombre maximum de chunks à inclure.

        Returns:
            ASTContext avec les chunks les plus pertinents.
        """
        n = self._col.count()
        if n == 0:
            # Fallback : parse directement depuis le filesystem
            return build_context(intent, self.root_dir)

        query = f"{intent.action} {intent.target_name} {intent.description}"
        results = self._col.query(
            query_texts=[query],
            n_results=min(top_k, n),
            include=["documents", "metadatas"],
        )

        chunks: list[CodeChunk] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(
                CodeChunk(
                    source=doc,
                    file_path=str(meta["file_path"]),
                    node_type=str(meta["node_type"]),
                    name=str(meta["name"]),
                    start_line=int(meta["start_line"]),
                    end_line=int(meta["end_line"]),
                )
            )

        target_file = intent.target_file or (chunks[0].file_path if chunks else None)
        return ASTContext(chunks=chunks, target_file=target_file)
