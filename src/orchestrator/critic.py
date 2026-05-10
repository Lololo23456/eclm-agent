"""CriticAgent — révision post-génération des incohérences cross-fichiers."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.orchestrator.router import ModelRouter
from src.shared.config import Config

logger = logging.getLogger(__name__)

_SEVERITIES = {"error", "warning", "info"}

_CRITIC_PROMPT = """\
Tu es un expert Python senior chargé de réviser un projet généré automatiquement.
Analyse les fichiers suivants et identifie UNIQUEMENT les problèmes concrets et vérifiables :

{files_block}

Cherche exclusivement :
1. Imports manquants ou incorrects (ex: `from src.models import User` mais User s'appelle UserModel)
2. Noms de fonctions/classes appelés mais non définis dans les fichiers fournis
3. Signatures incompatibles (mauvais nombre d'arguments, types incompatibles)
4. Variables utilisées avant définition
5. Fichiers tests qui importent des symboles inexistants

IGNORE : style, docstrings manquantes, optimisations potentielles, warnings mypy génériques.

Retourne UNIQUEMENT ce JSON (tableau vide [] si aucun problème) :
[
  {{
    "file": "chemin/relatif.py",
    "issue_type": "import_error|name_mismatch|signature_error|undefined_var|test_import_error",
    "description": "Description précise du problème (max 120 chars)",
    "severity": "error|warning",
    "line_hint": "extrait de code problématique (optionnel)"
  }}
]
Retourne UNIQUEMENT le JSON, sans markdown, sans commentaire."""


@dataclass
class CriticIssue:
    """Un problème détecté par le CriticAgent."""

    file: str
    issue_type: str
    description: str
    severity: str = "error"
    line_hint: str = ""


class CriticAgent:
    """Analyse cross-fichiers post-génération pour détecter les incohérences.

    Stratégie : envoie tous les fichiers générés au modèle fort (32B) avec
    un prompt focalisé sur les problèmes concrets (imports, noms, signatures).
    Appelé une seule fois à la fin de ProjectAgent.execute().
    """

    # Nombre max de chars de contexte pour éviter de dépasser la fenêtre du modèle
    _MAX_FILE_CHARS = 800
    _MAX_TOTAL_CHARS = 12_000

    def __init__(self, config: Config) -> None:
        self.config = config
        self._router = ModelRouter(config)

    def review(
        self,
        output_dir: Path,
        files: list[str],
    ) -> list[CriticIssue]:
        """Révise tous les fichiers générés et retourne les issues détectées.

        Args:
            output_dir: Dossier racine du projet généré.
            files: Liste de chemins absolus des fichiers à réviser.

        Returns:
            Liste de CriticIssue triée par sévérité (errors first).
        """
        py_files = [Path(f) for f in files if Path(f).suffix == ".py" and Path(f).exists()]
        if not py_files:
            return []

        files_block = self._build_files_block(output_dir, py_files)
        if not files_block.strip():
            return []

        raw = self._call_model(files_block)
        issues = self._parse_issues(raw)
        issues.sort(key=lambda i: (0 if i.severity == "error" else 1, i.file))
        logger.info("CriticAgent: %d issue(s) détectée(s)", len(issues))
        return issues

    def _build_files_block(self, output_dir: Path, py_files: list[Path]) -> str:
        parts: list[str] = []
        total = 0
        for fp in py_files:
            try:
                rel = fp.relative_to(output_dir).as_posix()
            except ValueError:
                rel = fp.name
            try:
                content = fp.read_text(encoding="utf-8")
            except OSError:
                continue
            # Tronquer les gros fichiers pour rester dans la fenêtre du modèle
            if len(content) > self._MAX_FILE_CHARS:
                content = content[: self._MAX_FILE_CHARS] + "\n# ... (tronqué)"
            block = f"### {rel}\n```python\n{content}\n```\n"
            if total + len(block) > self._MAX_TOTAL_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)

    def _call_model(self, files_block: str) -> str:
        model = self._router.for_planning()
        prompt = _CRITIC_PROMPT.format(files_block=files_block)
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 0},
        }).encode()

        req = urllib.request.Request(
            f"{self.config.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return str(data.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("CriticAgent Ollama error: %s", exc)
            return "[]"

    def _parse_issues(self, raw: str) -> list[CriticIssue]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        issues: list[CriticIssue] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                severity = str(item.get("severity", "error"))
                if severity not in _SEVERITIES:
                    severity = "error"
                issues.append(CriticIssue(
                    file=str(item.get("file", "?")),
                    issue_type=str(item.get("issue_type", "unknown")),
                    description=str(item.get("description", ""))[:200],
                    severity=severity,
                    line_hint=str(item.get("line_hint", "")),
                ))
            except (KeyError, TypeError):
                continue
        return issues
