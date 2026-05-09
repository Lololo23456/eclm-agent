"""FileWriter — applique le code validé sur le système de fichiers."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)

_GENERATIVE_ACTIONS = frozenset({"CREATE"})
_REPLACE_ACTIONS = frozenset({"MODIFY", "FIX", "REFACTOR", "OPTIMIZE", "ADD", "EXTRACT"})


class FileWriter:
    """Écrit le code validé dans le fichier cible.

    - CREATE : ajoute la fonction/classe à la fin du fichier (ou crée le fichier).
    - MODIFY/FIX/… : remplace le nœud AST cible dans le fichier existant.
    """

    def write(self, code: str, intent: IntentJSON, project_root: Path) -> Path | None:
        """Applique le code validé sur le filesystem.

        Args:
            code: Code Python validé par le verifier.
            intent: IntentJSON décrivant l'opération.
            project_root: Racine du projet cible.

        Returns:
            Chemin du fichier modifié, ou None si aucune action nécessaire.
        """
        if intent.action in _GENERATIVE_ACTIONS:
            return self._write_create(code, intent, project_root)
        if intent.action in _REPLACE_ACTIONS:
            return self._write_replace(code, intent, project_root)
        return None

    # ── CREATE ────────────────────────────────────────────────────────────────

    def _write_create(self, code: str, intent: IntentJSON, project_root: Path) -> Path:
        target = self._resolve_target_file(intent, project_root)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if intent.target_name and intent.target_name in existing:
                logger.info("Symbole '%s' déjà présent dans %s — ignoré", intent.target_name, target)
                return target
            # Ajoute à la fin avec une ligne vide de séparation
            separator = "\n\n" if existing.strip() else ""
            target.write_text(existing.rstrip() + separator + code + "\n", encoding="utf-8")
        else:
            target.write_text(code + "\n", encoding="utf-8")

        logger.info("Écrit dans %s", target)
        return target

    # ── MODIFY / REPLACE ──────────────────────────────────────────────────────

    def _write_replace(self, code: str, intent: IntentJSON, project_root: Path) -> Path | None:
        target = self._resolve_target_file(intent, project_root)
        if not target.exists():
            logger.warning("Fichier cible introuvable : %s", target)
            return None

        original = target.read_text(encoding="utf-8")
        replaced = _replace_node(original, intent.target_name, code)

        if replaced is None:
            # Symbole pas trouvé → on ajoute à la fin comme fallback
            logger.warning(
                "Symbole '%s' introuvable dans %s — ajout en fin de fichier",
                intent.target_name, target,
            )
            separator = "\n\n" if original.strip() else ""
            target.write_text(original.rstrip() + separator + code + "\n", encoding="utf-8")
        else:
            target.write_text(replaced, encoding="utf-8")

        logger.info("Mis à jour : %s", target)
        return target

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_target_file(self, intent: IntentJSON, project_root: Path) -> Path:
        if intent.target_file:
            p = Path(intent.target_file)
            return p if p.is_absolute() else project_root / p

        # Inférence depuis le target_name
        name = intent.target_name or "generated"
        if intent.target_type == "class":
            filename = f"{_to_snake(name)}.py"
        else:
            filename = f"{_to_snake(name)}.py"
        return project_root / filename


def _replace_node(source: str, target_name: str, new_code: str) -> str | None:
    """Remplace la définition de target_name dans source par new_code.

    Args:
        source: Code source original du fichier.
        target_name: Nom du symbole à remplacer.
        new_code: Nouveau code de la définition.

    Returns:
        Nouveau contenu du fichier, ou None si le symbole est introuvable.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines(keepends=True)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != target_name:
            continue

        # Lignes 0-indexées
        start = node.lineno - 1
        end = node.end_lineno  # type: ignore[attr-defined]

        # Inclure les décorateurs
        if node.decorator_list:
            start = node.decorator_list[0].lineno - 1

        # Préserver l'indentation du bloc original
        indent = _leading_indent(lines[start])
        new_lines = _reindent(new_code, indent)

        new_content = "".join(lines[:start]) + new_lines + "\n" + "".join(lines[end:])
        return new_content

    return None


def _leading_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(code: str, indent: str) -> str:
    """Applique l'indentation indent à chaque ligne de code."""
    if not indent:
        return code
    return "\n".join(indent + line if line.strip() else line for line in code.splitlines())


def _to_snake(name: str) -> str:
    """PascalCase ou camelCase → snake_case."""
    import re
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()
