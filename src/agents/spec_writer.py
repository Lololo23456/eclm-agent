"""SpecWriterAgent — enrichit une tâche avec une spec précise et complète."""
from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
Tu es un expert Python qui écrit des spécifications techniques précises.
Tu reçois une description de tâche et tu produis une spec complète que
n'importe quel développeur peut implémenter sans ambiguïté.
Réponds UNIQUEMENT avec du JSON valide, sans explication."""

_PROMPT = """\
Tâche à enrichir :
action: {action}
target_type: {target_type}
target_name: {target_name}
target_file: {target_file}
description: {description}

Contexte des fichiers existants:
{context}

Produis une spec complète au format JSON :
{{
  "signature": "def foo(a: int, b: str) -> bool:  # ou class Foo: etc.",
  "fields_or_params": ["param1: type", "param2: type"],
  "return_type": "type ou null",
  "constraints": ["règle impérative 1", "règle impérative 2"],
  "imports": ["from x import y", "import z"],
  "example_usage": "résultat = foo(1, 'test')  # retourne True",
  "public_api": "foo(a: int, b: str) -> bool"
}}"""


class SpecWriterAgent(BaseAgent):
    """Enrichit une tâche du plan avec une spec Python précise.

    Input : task dict (depuis plan.json)
    Output : task dict enrichi avec spec complète
    """

    def run(self, task: dict[str, Any], project_files: dict[str, str] | None = None) -> AgentResult:  # type: ignore[override]
        context = _format_context(project_files or {}, task.get("depends_on", []))

        # Si la spec est déjà complète, ne pas régénérer
        existing_spec = task.get("spec", {})
        if existing_spec.get("signature") and existing_spec.get("imports"):
            return AgentResult(success=True, output=task)

        prompt = _PROMPT.format(
            action=task.get("action", "CREATE"),
            target_type=task.get("target_type", "function"),
            target_name=task.get("target_name", ""),
            target_file=task.get("target_file", ""),
            description=existing_spec.get("description", task.get("spec", {}).get("description", "")),
            context=context,
        )
        raw = self._call_ollama(prompt, system=_SYSTEM, temperature=0.1)
        parsed = self._parse_json_response(raw)

        if not parsed:
            logger.warning("SpecWriter: JSON invalide pour %s", task.get("target_name"))
            return AgentResult(success=False, output=task, error="JSON invalide")

        enriched = dict(task)
        enriched["spec"] = {**existing_spec, **parsed}
        return AgentResult(success=True, output=enriched)


def _format_context(project_files: dict[str, str], depends_on_indices: list[int]) -> str:
    """Formate le contexte des fichiers existants pour le prompt."""
    if not project_files:
        return "(aucun fichier existant)"
    lines = []
    for path, code in list(project_files.items())[:3]:
        lines.append(f"# {path}\n{code[:400]}")
    return "\n\n".join(lines)
