"""C2 — ECLMCore : manipulations AST déterministes + Ollama pour le génératif."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from src.eclm.ast_ops import ASTOperationExecutor, LLMRequiredError
from src.orchestrator.context import ASTContext
from src.shared.config import Config
from src.shared.types import ASTCandidate, ASTOperation

logger = logging.getLogger(__name__)


def _strip_markdown(code: str) -> str:
    """Supprime les balises markdown ```python … ``` si présentes."""
    import re
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code.strip()


_LLM_OPS_PROMPT = """\
Tu es un générateur de code Python expert qui travaille au niveau AST.
Opération: {op_type} sur {target}
Paramètres: {params}

Code actuel de la cible:
{current_code}

Contexte du codebase:
{context}
{error_block}
Génère UNIQUEMENT le code Python résultant (fonction/classe complète).
Sans explication, sans balises markdown:"""


class ECLMCore:
    """Cœur du pipeline de génération de code.

    Stratégie :
    1. Essaie d'appliquer l'opération de façon déterministe (ast_ops).
    2. Si LLMRequiredError → génère k candidats via Ollama (beam_width seeds).

    Remplacé à terme par un modèle ~500M entraîné sur exécution (reward-based).
    """

    # Ops simples → moins de candidats pour économiser ressources M3
    _LIGHT_OPS = frozenset({"ADD_DOCSTRING", "ADD_RETURN_TYPE", "ADD_DECORATOR", "ADD_IMPORT"})
    _HEAVY_OPS = frozenset({"CREATE_CLASS", "EXTRACT_FUNCTION", "MERGE", "SPLIT"})

    def __init__(self, config: Config) -> None:
        self.config = config
        self._executor = ASTOperationExecutor()

    def _adaptive_k(self, base_k: int, operation: ASTOperation) -> int:
        """Adapte le beam width selon la complexité de l'opération.

        Réduit le nombre de candidats sur les ops légères pour économiser
        la batterie et la mémoire sur MacBook Air M3.
        """
        if not self.config.adaptive_beam_width:
            return base_k
        if self._executor.is_deterministic(operation.op_type):
            return 1
        if operation.op_type in self._LIGHT_OPS:
            return min(2, base_k)
        if operation.op_type in self._HEAVY_OPS:
            return base_k  # Complexe → beam complet
        return min(3, base_k)  # Défaut : k=3 max sur M3

    def generate(
        self,
        operation: ASTOperation,
        context: ASTContext,
        error: str | None = None,
        k: int = 5,
    ) -> list[ASTCandidate]:
        """Génère jusqu'à k candidats pour une opération AST.

        Args:
            operation: Opération AST à réaliser.
            context: Contexte du codebase (chunks pertinents).
            error: Erreur du verifier pour self-reflection.
            k: Nombre de candidats (beam_width).

        Returns:
            Liste de ASTCandidate (au moins 1, même si vide).
        """
        effective_k = self._adaptive_k(k, operation)

        # Déterministe : retourne exactement 1 candidat parfait (si code source disponible)
        if self._executor.is_deterministic(operation.op_type):
            current = context.get_target_code(operation.target) or ""
            if current.strip():
                try:
                    result = self._executor.apply(current, operation)
                    return [ASTCandidate(code=result, operation=operation, generation_rank=0)]
                except (SyntaxError, KeyError, ValueError) as exc:
                    logger.warning("AST op déterministe échouée: %s", exc)

        # Génératif : beam search via Ollama (k adaptatif)
        return self._generate_via_ollama(operation, context, error, effective_k)

    def _generate_via_ollama(
        self,
        operation: ASTOperation,
        context: ASTContext,
        error: str | None,
        k: int,
    ) -> list[ASTCandidate]:
        current = context.get_target_code(operation.target) or ""
        error_block = f"\nERREUR À CORRIGER:\n{error}\n" if error else ""
        prompt = _LLM_OPS_PROMPT.format(
            op_type=operation.op_type,
            target=operation.target,
            params=json.dumps(operation.params, ensure_ascii=False),
            current_code=current or "(nouveau code)",
            context=context.format_for_prompt(max_chars=1500),
            error_block=error_block,
        )

        candidates: list[ASTCandidate] = []
        for rank in range(k):
            code = self._call_ollama(prompt, seed=rank)
            if code:
                candidates.append(ASTCandidate(code=code, operation=operation, generation_rank=rank))

        if not candidates:
            logger.warning("Ollama indisponible ou silencieux pour %s:%s", operation.op_type, operation.target)
            candidates.append(ASTCandidate(code=current, operation=operation, generation_rank=0))

        return candidates

    def _call_ollama(self, prompt: str, seed: int) -> str:
        payload = json.dumps({
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"seed": seed, "temperature": 0.2 + seed * 0.1},
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
                return _strip_markdown(str(data.get("response", "")).strip())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.debug("Ollama error (seed=%d): %s", seed, exc)
            return ""
