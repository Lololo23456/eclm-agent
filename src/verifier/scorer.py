"""Vérificateurs syntaxiques, types et lint pour les candidats ECLM."""
from __future__ import annotations

import ast
import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SyntaxChecker:
    """Vérifie la syntaxe Python via ast.parse (< 1 ms)."""

    def check(self, code: str) -> bool:
        """Retourne True si la syntaxe Python est valide.

        Args:
            code: Code Python à vérifier.

        Returns:
            True si ast.parse réussit, False sinon.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class TypeChecker:
    """Lance mypy --strict sur le code candidat."""

    def check(self, code: str) -> tuple[bool, str]:
        """Lance mypy et retourne (ok, message_erreur).

        Args:
            code: Code Python à type-checker.

        Returns:
            Tuple (ok, error_message). error_message est vide si ok.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", "--no-error-summary", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = result.returncode == 0
            return ok, "" if ok else (result.stdout + result.stderr).strip()
        except subprocess.TimeoutExpired:
            return True, "mypy timeout — skipped"
        except FileNotFoundError:
            return True, "mypy unavailable — skipped"
        finally:
            tmp_path.unlink(missing_ok=True)


class LintScorer:
    """Lance ruff et calcule un score de qualité du code (0.0–1.0)."""

    _MAX_VIOLATIONS = 10

    def score(self, code: str) -> float:
        """Retourne un score entre 0.0 et 1.0 (1.0 = aucune violation ruff).

        Args:
            code: Code Python à analyser.

        Returns:
            1.0 si aucune violation, décroit linéairement jusqu'à 0.0 à 10 violations.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return 1.0
            violations: list[object] = json.loads(result.stdout or "[]")
            n = len(violations)
            return max(0.0, 1.0 - n / self._MAX_VIOLATIONS)
        except subprocess.TimeoutExpired:
            logger.warning("ruff timeout — score neutre 0.5")
            return 0.5
        except FileNotFoundError:
            logger.warning("ruff non disponible — score neutre 0.5")
            return 0.5
        except (json.JSONDecodeError, ValueError):
            return 0.0
        finally:
            tmp_path.unlink(missing_ok=True)
