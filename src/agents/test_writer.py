"""TestWriterAgent — génère des tests pytest depuis une spec (AVANT le code)."""
from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
Tu es un expert en testing Python. Tu écris des tests pytest précis et exhaustifs.
Règles absolues :
- Les fonctions de test sont STANDALONE (def test_xxx(): sans self)
- Chaque test a un seul assert clair
- Les tests couvrent : nominal, edge case, erreur attendue
- Imports explicites dans chaque test si nécessaire
Réponds UNIQUEMENT avec le code Python des fonctions de test, sans explication."""

_PROMPT = """\
Génère des tests pytest pour :
  Nom    : {target_name}
  Type   : {target_type}
  Fichier: {target_file}
  Spec   : {description}
  Signature : {signature}
  Contraintes : {constraints}

Tests déjà fournis dans le plan (à améliorer si incomplets) :
{existing_tests}

Public API : {public_api}

Retourne 3-5 fonctions de test standalone complètes :
def test_{target_name}_nominal():
    ...

def test_{target_name}_edge_case():
    ...

def test_{target_name}_invalid():
    ...
"""


def parse_test_functions(raw: str) -> list[str]:
    """Extrait les fonctions def test_*() d'une réponse LLM."""
    import ast
    import re

    # Nettoyer le markdown
    raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).replace("```", "").strip()

    tests: list[str] = []
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        # Rejeter les méthodes (ont self comme premier arg)
        args = node.args.args
        if args and args[0].arg == "self":
            continue
        # Extraire le source
        lines = raw.splitlines()
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") else start + 10
        tests.append("\n".join(lines[start:end]))

    return tests


class TestWriterAgent(BaseAgent):
    """Génère des tests pytest depuis une spec de tâche.

    Input  : task dict avec spec enrichie
    Output : liste de strings (fonctions pytest)
    ISOLATION : jamais accès au code généré par CodeWriterAgent
    """

    def run(self, task: dict[str, Any], project_files: dict[str, str] | None = None) -> AgentResult:  # type: ignore[override]
        spec = task.get("spec", {})
        existing_tests = task.get("tests", [])

        # Si tests de qualité déjà présents dans le plan, les utiliser
        if len(existing_tests) >= 2 and all("assert" in t for t in existing_tests):
            return AgentResult(success=True, output=existing_tests)

        prompt = _PROMPT.format(
            target_name=task.get("target_name", "target"),
            target_type=task.get("target_type", "function"),
            target_file=task.get("target_file", ""),
            description=spec.get("description", ""),
            signature=spec.get("signature", ""),
            constraints="\n".join(f"- {c}" for c in spec.get("constraints", [])),
            existing_tests="\n\n".join(existing_tests) if existing_tests else "(aucun)",
            public_api=spec.get("public_api", ""),
        )

        raw = self._call_ollama(prompt, system=_SYSTEM, temperature=0.2)
        tests = parse_test_functions(raw)

        if not tests:
            # Fallback : utiliser les tests du plan bruts
            if existing_tests:
                return AgentResult(success=True, output=existing_tests)
            return AgentResult(success=False, output=[], error="Aucune fonction test extraite")

        return AgentResult(success=True, output=tests)
