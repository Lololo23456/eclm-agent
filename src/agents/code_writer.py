"""CodeWriterAgent — génère du code Python depuis une spec + tests."""
from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
Tu es un expert Python qui implémente des spécifications précises.
Tu DOIS respecter exactement : la signature, les imports, les contraintes.
Tu génères UNIQUEMENT le code Python demandé, sans explication, sans markdown."""

_PROMPT = """\
Implémente en Python :
  Nom       : {target_name}
  Type      : {target_type}
  Fichier   : {target_file}
  Signature : {signature}
  Description : {description}

Imports OBLIGATOIRES (tous doivent être présents) :
{imports}

Contraintes IMPÉRATIVES :
{constraints}

Tests qui DOIVENT passer :
{tests}

Contexte (code existant dans les fichiers dépendants) :
{context}

{error_block}
Génère le code complet (imports + implémentation) :"""


def _strip_markdown(code: str) -> str:
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code.strip()


class CodeWriterAgent(BaseAgent):
    """Génère k candidats de code depuis une spec et des tests.

    Input  : task dict avec spec enrichie + tests
    Output : liste de strings (code Python candidat)
    """

    def run(  # type: ignore[override]
        self,
        task: dict[str, Any],
        tests: list[str],
        project_files: dict[str, str] | None = None,
        k: int = 3,
        error: str | None = None,
    ) -> AgentResult:
        spec = task.get("spec", {})
        context = _format_context(project_files or {}, task.get("target_file", ""))
        error_block = f"ERREUR À CORRIGER (tentative précédente) :\n{error}\n" if error else ""

        prompt = _PROMPT.format(
            target_name=task.get("target_name", "target"),
            target_type=task.get("target_type", "function"),
            target_file=task.get("target_file", ""),
            signature=spec.get("signature", ""),
            description=spec.get("description", ""),
            imports="\n".join(spec.get("imports", [])) or "(déduire depuis la spec)",
            constraints="\n".join(f"- {c}" for c in spec.get("constraints", [])) or "(aucune)",
            tests="\n\n".join(tests[:3]) if tests else "(aucun test fourni)",
            context=context,
            error_block=error_block,
        )

        # Beam search : k candidats à températures variées
        temperatures = [0.1, 0.3, 0.5][:k]
        candidates: list[str] = []
        for temp in temperatures:
            raw = self._call_ollama(prompt, system=_SYSTEM, temperature=temp)
            if raw:
                candidates.append(_strip_markdown(raw))

        if not candidates:
            return AgentResult(success=False, output=[], error="Ollama non disponible")

        return AgentResult(success=True, output=candidates)


def _format_context(project_files: dict[str, str], current_file: str) -> str:
    """Formate le contexte des fichiers existants (hors fichier courant)."""
    lines = []
    for path, code in project_files.items():
        if path == current_file:
            continue
        lines.append(f"# {path}\n{code[:600]}")
        if len(lines) >= 3:
            break
    return "\n\n".join(lines) if lines else "(aucun contexte)"
