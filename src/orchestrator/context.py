"""Gestion du contexte codebase — chunks AST-aware via tree-sitter."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_python

from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)

_PY_LANG = Language(tree_sitter_python.language())
_PARSER = Parser(_PY_LANG)

# Types de nœuds tree-sitter qui constituent des chunks sémantiques
_CHUNK_NODE_TYPES = frozenset(
    {"function_definition", "class_definition", "decorated_definition"}
)


@dataclass(frozen=True)
class CodeChunk:
    """Fragment de code avec métadonnées de localisation."""

    source: str
    file_path: str
    node_type: str
    name: str
    start_line: int
    end_line: int


@dataclass
class ASTContext:
    """Contexte AST assemblé pour une requête donnée."""

    chunks: list[CodeChunk] = field(default_factory=list)
    target_file: str | None = None
    dependency_context: str = ""

    def get_target_code(self, target_name: str) -> str | None:
        """Retourne le code source du chunk dont le nom correspond à target_name.

        Args:
            target_name: Nom du symbole cible (fonction, classe…).

        Returns:
            Code source du chunk ou None si introuvable.
        """
        for chunk in self.chunks:
            if chunk.name == target_name:
                return chunk.source
        return None

    def format_for_prompt(self, max_chars: int = 4000) -> str:
        """Sérialise le contexte pour injection dans un prompt LLM.

        Args:
            max_chars: Limite de caractères pour éviter les prompts trop longs.

        Returns:
            Chaîne formatée avec les chunks pertinents.
        """
        parts: list[str] = []
        total = 0

        if self.dependency_context:
            parts.append(self.dependency_context)
            total += len(self.dependency_context)

        for chunk in self.chunks:
            block = f"# {chunk.file_path}:{chunk.start_line}\n{chunk.source}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)


def extract_chunks(file_path: Path) -> list[CodeChunk]:
    """Découpe un fichier Python en chunks sémantiques via tree-sitter.

    Args:
        file_path: Chemin vers le fichier .py à analyser.

    Returns:
        Liste de CodeChunk (fonctions, classes, méthodes décorées).
    """
    try:
        source_bytes = file_path.read_bytes()
    except OSError:
        return []

    tree = _PARSER.parse(source_bytes)
    source_text = source_bytes.decode("utf-8", errors="replace")
    chunks: list[CodeChunk] = []

    def _walk(node: Node) -> None:
        if node.type in _CHUNK_NODE_TYPES:
            name = _extract_name(node, source_bytes)
            chunk_src = source_text[node.start_byte : node.end_byte]
            chunks.append(
                CodeChunk(
                    source=chunk_src,
                    file_path=str(file_path),
                    node_type=node.type,
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return chunks


def _extract_name(node: Node, source: bytes) -> str:
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return "<anonymous>"


def build_context(intent: IntentJSON, root_dir: Path) -> ASTContext:
    """Construit le contexte AST pertinent pour une intention donnée.

    Stratégie :
    - Si target_file est connu → parse ce fichier uniquement.
    - Sinon → cherche dans tous les .py du projet, filtre par target_name.

    Args:
        intent: IntentJSON produit par C0.
        root_dir: Racine du projet codebase.

    Returns:
        ASTContext avec les chunks les plus pertinents.
    """
    if intent.target_file:
        target = root_dir / intent.target_file
        chunks = extract_chunks(target)
        relevant = _filter_by_name(chunks, intent.target_name)
        return ASTContext(chunks=relevant or chunks, target_file=intent.target_file)

    # Recherche dans tout le projet
    all_chunks: list[CodeChunk] = []
    for py_file in root_dir.rglob("*.py"):
        if any(part.startswith(".") for part in py_file.parts):
            continue
        all_chunks.extend(extract_chunks(py_file))

    relevant = _filter_by_name(all_chunks, intent.target_name)
    chosen = relevant[:10] if relevant else all_chunks[:5]
    target_file = chosen[0].file_path if chosen else None
    return ASTContext(chunks=chosen, target_file=target_file)


def _filter_by_name(chunks: list[CodeChunk], name: str) -> list[CodeChunk]:
    """Filtre les chunks dont le nom correspond (exact ou partiel)."""
    exact = [c for c in chunks if c.name == name]
    if exact:
        return exact
    lower = name.lower()
    return [c for c in chunks if lower in c.name.lower()]
