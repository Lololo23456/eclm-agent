"""C1 — ASTPlanner : décompose une IntentJSON en ASTOperationPlan."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from src.orchestrator.context import ASTContext
from src.shared.config import Config
from src.shared.types import ASTOperation, ASTOperationPlan, IntentJSON

logger = logging.getLogger(__name__)

# Actions qui mappent directement vers des ops déterministes
_DIRECT_OPS: dict[str, str] = {
    "RENAME": "RENAME_SYMBOL",
    "DELETE": "DELETE_NODE",
}

_LLM_ACTIONS = frozenset({
    "MODIFY", "FIX", "REFACTOR", "OPTIMIZE", "CREATE",
    "ADD", "EXTRACT", "MERGE", "SPLIT", "CONVERT", "TEST", "EXPLAIN",
})

_PLANNER_PROMPT = """\
Tu es un planificateur d'opérations AST. Tu reçois une intention et tu retournes \
un plan JSON d'opérations atomiques à exécuter dans l'ordre.

Types d'opérations valides:
ADD_PARAM, MODIFY_BODY, REMOVE_PARAM, ADD_RETURN_TYPE, RENAME_SYMBOL,
ADD_IMPORT, CREATE_FUNCTION, CREATE_CLASS, ADD_METHOD, DELETE_NODE,
UPDATE_CALL_SITES, ADD_DECORATOR, EXTRACT_FUNCTION, ADD_DOCSTRING, MODIFY_DECORATOR

Intention:
{intent_json}

Contexte (extrait du codebase):
{context}

Retourne UNIQUEMENT un JSON valide:
{{
  "operations": [
    {{"op_type": "...", "target": "...", "params": {{}}}}
  ],
  "estimated_complexity": 1
}}"""


class ASTPlanner:
    """Décompose une IntentJSON en plan d'opérations AST.

    Stratégie hybride :
    - Règles directes pour les intentions simples (RENAME, DELETE).
    - Ollama pour les intentions complexes qui nécessitent du raisonnement.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def plan(self, intent: IntentJSON, context: ASTContext) -> ASTOperationPlan:
        """Retourne le plan d'opérations pour une intention.

        Args:
            intent: IntentJSON validé par C0.
            context: Contexte AST du codebase.

        Returns:
            ASTOperationPlan avec les opérations ordonnées.
        """
        # Cas directs — pas besoin d'Ollama
        if intent.action in _DIRECT_OPS:
            return self._direct_plan(intent)

        # Cas génératifs simples — une seule op MODIFY_BODY ou CREATE_*
        if intent.action in {"MODIFY", "FIX", "OPTIMIZE", "REFACTOR"}:
            return self._single_op_plan(intent, "MODIFY_BODY")

        if intent.action == "CREATE":
            if intent.target_type == "class":
                op_type = "CREATE_CLASS"
            elif intent.target_type == "module":
                op_type = "CREATE_MODULE"
            else:
                op_type = "CREATE_FUNCTION"
            return self._single_op_plan(intent, op_type)

        if intent.action == "TEST":
            op_type = "CREATE_MODULE" if intent.target_type == "module" else "CREATE_FUNCTION"
            return self._single_op_plan(intent, op_type)

        # Cas complexes — Ollama planifie
        plan = self._plan_via_ollama(intent, context)
        if plan is not None:
            return plan

        # Fallback ultime
        return self._single_op_plan(intent, "MODIFY_BODY")

    def _direct_plan(self, intent: IntentJSON) -> ASTOperationPlan:
        op_type = _DIRECT_OPS[intent.action]
        params: dict[str, object] = {"description": intent.description}
        if op_type == "RENAME_SYMBOL":
            new_name = next(
                (c.split("→")[-1].strip() for c in intent.constraints if "→" in c),
                intent.target_name + "_renamed",
            )
            params = {"new_name": new_name}
        op = ASTOperation(op_type=op_type, target=intent.target_name, params=params)
        return ASTOperationPlan(
            operations=(op,), intent=intent, estimated_complexity=1
        )

    def _single_op_plan(self, intent: IntentJSON, op_type: str) -> ASTOperationPlan:
        op = ASTOperation(
            op_type=op_type,
            target=intent.target_name,
            params={
                "description": intent.description,
                "constraints": list(intent.constraints),
                "target_type": intent.target_type,
            },
        )
        return ASTOperationPlan(
            operations=(op,), intent=intent, estimated_complexity=1
        )

    def _plan_via_ollama(self, intent: IntentJSON, context: ASTContext) -> ASTOperationPlan | None:
        intent_dict = {
            "action": intent.action,
            "target_type": intent.target_type,
            "target_name": intent.target_name,
            "description": intent.description,
            "constraints": list(intent.constraints),
        }
        prompt = _PLANNER_PROMPT.format(
            intent_json=json.dumps(intent_dict, ensure_ascii=False, indent=2),
            context=context.format_for_prompt(max_chars=1000),
        )
        payload = json.dumps({
            "model": self.config.ollama_model,
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                raw = str(data.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Ollama planner indisponible: %s", exc)
            return None

        return self._parse_plan(raw, intent)

    def _parse_plan(self, raw: str, intent: IntentJSON) -> ASTOperationPlan | None:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            ops = tuple(
                ASTOperation(
                    op_type=str(o["op_type"]),
                    target=str(o.get("target", intent.target_name)),
                    params=dict(o.get("params", {})),
                )
                for o in data.get("operations", [])
            )
            if not ops:
                return None
            return ASTOperationPlan(
                operations=ops,
                intent=intent,
                estimated_complexity=int(data.get("estimated_complexity", len(ops))),
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None
