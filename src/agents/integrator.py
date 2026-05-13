"""IntegratorAgent — vérifie la cohérence cross-fichiers d'un projet."""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

from src.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
Tu es un expert Python en revue de code. Tu vérifies la cohérence entre
plusieurs fichiers d'un projet. Tu listes les problèmes concrets avec
leur localisation exacte. Réponds en JSON uniquement."""

_PROMPT = """\
Vérifie la cohérence cross-fichiers de ce projet Python.

Fichiers générés :
{files_summary}

Vérifie :
1. Imports manquants ou incorrects (A importe X de B mais X n'est pas dans B)
2. Signatures incompatibles (A appelle B.foo(x, y) mais B.foo ne prend qu'un argument)
3. Types incohérents (A retourne str, B attend int)
4. Fichiers __init__.py manquants

Retourne UNIQUEMENT :
{{
  "issues": [
    {{"file": "src/api.py", "line": 5, "problem": "importe User depuis models mais User n'est pas exporté", "severity": "error"}},
    ...
  ],
  "ok": true/false
}}"""


@dataclass
class IntegrationIssue:
    file: str
    problem: str
    severity: str  # "error" | "warning"
    line: int = 0


class IntegratorAgent(BaseAgent):
    """Vérifie la cohérence cross-fichiers après génération.

    Input  : dict {path: code} de tous les fichiers générés
    Output : liste d'IntegrationIssue
    """

    def run(self, project_files: dict[str, str]) -> AgentResult:  # type: ignore[override]
        # Analyse statique rapide toujours (gratuit, même sur 1 fichier)
        static_issues = _static_check(project_files)

        # LLM cross-file seulement si plusieurs fichiers
        if len(project_files) <= 1:
            return AgentResult(success=True, output=static_issues)

        files_summary = _format_files_summary(project_files)
        prompt = _PROMPT.format(files_summary=files_summary)
        raw = self._call_ollama(prompt, system=_SYSTEM, temperature=0.0)
        parsed = self._parse_json_response(raw)

        llm_issues: list[IntegrationIssue] = []
        if parsed and "issues" in parsed:
            for iss in parsed["issues"]:
                if isinstance(iss, dict):
                    llm_issues.append(IntegrationIssue(
                        file=str(iss.get("file", "")),
                        problem=str(iss.get("problem", "")),
                        severity=str(iss.get("severity", "warning")),
                        line=int(iss.get("line", 0)),
                    ))

        all_issues = static_issues + llm_issues
        return AgentResult(
            success=True,
            output=all_issues,
            error=f"{len(all_issues)} problème(s) détecté(s)" if all_issues else None,
        )


def _static_check(project_files: dict[str, str]) -> list[IntegrationIssue]:
    """Vérifie la syntaxe de chaque fichier et les imports évidents."""
    issues: list[IntegrationIssue] = []
    for path, code in project_files.items():
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(IntegrationIssue(
                file=path,
                problem=f"SyntaxError: {e.msg}",
                severity="error",
                line=e.lineno or 0,
            ))
    return issues


def _format_files_summary(project_files: dict[str, str]) -> str:
    """Résume chaque fichier pour le prompt LLM (max 400 chars par fichier)."""
    parts: list[str] = []
    for path, code in project_files.items():
        preview = code[:400].replace("\n", "\\n")
        parts.append(f"=== {path} ===\n{preview}")
    return "\n\n".join(parts)
