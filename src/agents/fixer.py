"""FixerAgent — corrige du code à partir d'une erreur précise."""
from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
Tu es un expert Python en débogage. Tu reçois du code qui échoue et
l'erreur exacte. Tu produis UNIQUEMENT le code corrigé, sans explication.
La correction doit être minimale : ne changer QUE ce qui cause l'erreur."""

_PROMPT = """\
Code qui échoue :
```python
{code}
```

Erreur exacte :
{error}

Spec attendue :
  Signature   : {signature}
  Contraintes : {constraints}

Tests qui doivent passer :
{tests}

Génère le code Python corrigé (complet, sans markdown) :"""


def _strip_markdown(code: str) -> str:
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code.strip()


class FixerAgent(BaseAgent):
    """Corrige du code en s'appuyant sur l'erreur du Verifier.

    Input  : code échoué + message d'erreur + spec
    Output : code corrigé
    """

    def run(  # type: ignore[override]
        self,
        code: str,
        error: str,
        task: dict[str, Any],
        tests: list[str],
    ) -> AgentResult:
        spec = task.get("spec", {})

        prompt = _PROMPT.format(
            code=code,
            error=error[:800],
            signature=spec.get("signature", ""),
            constraints="\n".join(f"- {c}" for c in spec.get("constraints", [])) or "(aucune)",
            tests="\n\n".join(tests[:3]) if tests else "(aucun test)",
        )

        raw = self._call_ollama(prompt, system=_SYSTEM, temperature=0.1)
        if not raw:
            return AgentResult(success=False, output=code, error="Ollama non disponible")

        fixed = _strip_markdown(raw)
        if not fixed:
            return AgentResult(success=False, output=code, error="Réponse vide")

        return AgentResult(success=True, output=fixed)
