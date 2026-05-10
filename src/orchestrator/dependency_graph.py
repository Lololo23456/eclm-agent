"""DependencyGraph — contexte cross-fichiers pour la génération de code."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_public_api(file_path: Path) -> str:
    """Extrait les signatures publiques d'un fichier Python via ast.

    Retourne les imports, constantes, signatures de fonctions et définitions
    de classes — suffisant pour que le LLM sache quoi importer et comment appeler.

    Args:
        file_path: Chemin vers le fichier .py à analyser.

    Returns:
        String avec l'API publique, ou "" si le fichier est invalide.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        logger.debug("extract_public_api: %s — %s", file_path.name, exc)
        return ""

    lines: list[str] = []

    for node in ast.iter_child_nodes(tree):
        # Imports top-level (utiles pour reproduire dans le fichier dépendant)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))

        # Constantes publiques (UPPER_CASE)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    lines.append(ast.unparse(node))
                    break

        # Fonctions top-level publiques → signature uniquement
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                sig = f"def {node.name}({ast.unparse(node.args)})"
                if node.returns:
                    sig += f" -> {ast.unparse(node.returns)}"
                sig += ": ..."
                doc = ast.get_docstring(node)
                if doc:
                    sig += f"  # {doc.splitlines()[0][:80]}"
                lines.append(sig)

        # Classes top-level publiques → signature + méthodes publiques
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                class_line = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
                lines.append(class_line)
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f'    """{doc.splitlines()[0][:80]}"""')
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        is_public = not child.name.startswith("_")
                        is_dunder = child.name in ("__init__", "__str__", "__repr__", "__len__")
                        if is_public or is_dunder:
                            sig = f"    def {child.name}({ast.unparse(child.args)})"
                            if child.returns:
                                sig += f" -> {ast.unparse(child.returns)}"
                            sig += ": ..."
                            lines.append(sig)

    return "\n".join(lines)


class DependencyGraph:
    """Construit le contexte cross-fichiers pour les tâches d'un projet.

    Principe : avant de générer le fichier N, on lit les APIs publiques des
    fichiers déjà générés dont N dépend (via task.depends_on). Cela garantit
    que le LLM connaît les noms exacts des classes et fonctions à importer.

    Context pruning : pour les gros projets, on limite le contexte total à
    MAX_CONTEXT_CHARS. Les fichiers dont le nom apparaît dans la description
    de la tâche sont prioritaires.
    """

    MAX_CONTEXT_CHARS = 4_000
    MAX_API_PER_FILE = 600   # chars max par fichier

    def get_context_for_task(
        self,
        task: object,
        all_tasks: list[object],
        output_dir: Path,
    ) -> str:
        """Retourne le contexte des fichiers dont cette tâche dépend.

        Args:
            task: TaskRecord en cours de génération.
            all_tasks: Toutes les tâches de la session.
            output_dir: Dossier de sortie du projet.

        Returns:
            String formaté avec les APIs publiques des dépendances.
            Vide si aucune dépendance ou aucun fichier trouvé.
        """
        from src.orchestrator.project import TaskRecord

        assert isinstance(task, TaskRecord)
        if not task.depends_on:
            return ""

        dep_indices = set(task.depends_on)
        dep_tasks = [
            t for t in all_tasks
            if isinstance(t, TaskRecord) and t.index in dep_indices
        ]

        # Dédupliquer par fichier relatif
        seen: dict[str, Path] = {}
        for dep in dep_tasks:
            for file_str in dep.files_created:
                fp = Path(file_str)
                if not fp.exists() or fp.suffix != ".py":
                    continue
                try:
                    rel = fp.relative_to(output_dir).as_posix()
                except ValueError:
                    rel = fp.name
                seen[rel] = fp

        if not seen:
            return ""

        # Priorité aux fichiers dont le nom apparaît dans la description
        task_desc = (task.description + " " + task.target_file).lower()
        def _priority(rel: str) -> int:
            stem = Path(rel).stem.lower()
            return 0 if stem in task_desc else 1

        sorted_files = sorted(seen.items(), key=lambda kv: _priority(kv[0]))

        parts: list[str] = []
        total = 0
        for rel_path, abs_path in sorted_files:
            api = extract_public_api(abs_path)
            if not api:
                continue
            # Tronquer les gros fichiers pour rester dans le budget
            if len(api) > self.MAX_API_PER_FILE:
                api = api[: self.MAX_API_PER_FILE] + "\n# ..."
            block = f"# {rel_path}\n{api}"
            if total + len(block) > self.MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block)

        if not parts:
            return ""

        return (
            "=== APIs disponibles dans les fichiers déjà générés ===\n"
            + "\n\n".join(parts)
            + "\n=== Utilise ces imports et noms exacts dans ton code ==="
        )
