"""TestGenerator — génère des tests d'implémentation isolés des candidats ECLM.

RÈGLE FONDAMENTALE : ce module ne reçoit JAMAIS les candidats ECLM.
Il génère ses tests depuis la description de la tâche ou le code original,
avant que l'ECLMCore ait produit quoi que ce soit.
"""
from __future__ import annotations

import ast
import logging
import re

import requests

from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)

_TEST_TIMEOUT = 10  # secondes

_PROMPT_FROM_INTENT = """\
You are a Python testing expert. Generate pytest unit tests based on this task description.

Task: {action} a {target_type} named `{target_name}`
Description: {description}

Rules:
- Each test function is standalone: `def test_xxx():` with NO `self` parameter
- Import the target using: `from solution import {target_name}`
- Test normal cases, edge cases (empty, None, zero, negative) and expected exceptions
- Use `pytest.raises` for error cases
- Output ONLY the test functions, no explanation

Write 2-4 pytest test functions:
```python
"""

_PROMPT_FROM_CODE = """\
You are a Python testing expert. Generate pytest unit tests for the following code.

```python
{code}
```

Rules:
- Each test function is standalone: `def test_xxx():` with NO `self` parameter
- Import using: `from solution import *`
- Test normal behaviour, edge cases (empty, None, zero, negative) and expected exceptions
- Use `pytest.raises` for error cases
- Output ONLY the test functions, no explanation

Write 2-4 pytest test functions:
```python
"""


class TestGeneratorOutput:
    """Résultat du TestGenerator."""

    def __init__(self, tests: list[str], confidence: float) -> None:
        self.tests = tests
        self.confidence = confidence

    def __bool__(self) -> bool:
        return bool(self.tests)


class TestGenerator:
    """Génère des tests d'implémentation isolés des candidats ECLM.

    Deux modes d'utilisation :
    - generate_from_intent() : avant toute génération (mode CREATE)
    - generate_from_code()   : depuis code existant (mode MODIFY/FIX)

    Utilise le modèle fast (7B) via Ollama. Intentionnellement séparé
    de ECLMCore pour éviter le biais "code faux validé par ses propres tests".
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate_from_intent(self, intent: IntentJSON) -> TestGeneratorOutput:
        """Génère des tests depuis la description de la tâche — AVANT la génération ECLM.

        Appelé en mode CREATE : aucun code candidat n'existe encore.

        Args:
            intent: IntentJSON décrivant la tâche à tester.

        Returns:
            TestGeneratorOutput avec les fonctions pytest générées.
        """
        prompt = _PROMPT_FROM_INTENT.format(
            action=intent.action.lower(),
            target_type=intent.target_type,
            target_name=intent.target_name,
            description=intent.description,
        )
        return self._call_and_parse(prompt)

    def generate_from_code(self, code: str) -> TestGeneratorOutput:
        """Génère des tests depuis du code DÉJÀ EXISTANT (jamais un candidat ECLM).

        Appelé en mode MODIFY/FIX sur le code original, avant que l'ECLM génère.

        Args:
            code: Code source Python existant et vérifié.

        Returns:
            TestGeneratorOutput avec les fonctions pytest générées.
        """
        if not code.strip():
            return TestGeneratorOutput(tests=[], confidence=0.0)
        prompt = _PROMPT_FROM_CODE.format(code=code[:3000])
        return self._call_and_parse(prompt)

    # ── Ancienne interface (rétrocompatibilité) ────────────────────────────────

    def load(self) -> None:
        """No-op : pas de modèle à charger, Ollama est utilisé à la demande."""

    def generate(self, code: str) -> TestGeneratorOutput:
        """Alias rétrocompatible → generate_from_code()."""
        return self.generate_from_code(code)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _call_and_parse(self, prompt: str) -> TestGeneratorOutput:
        raw = self._call_ollama(prompt)
        tests = parse_test_functions(raw)
        confidence = min(1.0, len(tests) * 0.3) if tests else 0.0
        logger.debug("TestGenerator: %d test(s) générés (confiance=%.1f)", len(tests), confidence)
        return TestGeneratorOutput(tests=tests, confidence=confidence)

    def _call_ollama(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.fast_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                },
                timeout=_TEST_TIMEOUT,
            )
            resp.raise_for_status()
            return str(resp.json().get("response", ""))
        except Exception as exc:
            logger.warning("TestGenerator Ollama error: %s", exc)
            return ""


def parse_test_functions(raw: str) -> list[str]:
    """Extrait les fonctions pytest valides depuis une réponse LLM.

    Args:
        raw: Texte brut retourné par le modèle.

    Returns:
        Liste de corps de fonctions pytest (texte complet de chaque def test_…).
    """
    # Extraire le bloc de code si présent
    if "```python" in raw:
        raw = raw.split("```python")[-1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    # Parser et filtrer les fonctions test_*
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        # Tentative de récupération partielle : chercher les blocs def test_
        return _extract_via_regex(raw)

    source_lines = raw.splitlines()
    tests: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        # Rejeter les fonctions avec `self` (méthodes de classe)
        args = node.args.args
        if args and args[0].arg == "self":
            continue
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        fn_src = "\n".join(source_lines[start:end])
        tests.append(fn_src)

    return tests


def _extract_via_regex(raw: str) -> list[str]:
    """Fallback : extrait les blocs def test_* par regex si ast.parse échoue."""
    pattern = re.compile(r"(def test_\w+\([^)]*\):.*?)(?=\ndef test_|\Z)", re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(raw)]
