"""C0 — Extracteur d'intention français → IntentJSON via Ollama (puis CamemBERT)."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.shared.config import Config
from src.shared.types import VALID_ACTIONS, VALID_TARGET_TYPES, IntentJSON

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Tu es un parser d'intentions de codage. Tu reçois une commande de développeur en français \
et tu retournes UNIQUEMENT un objet JSON valide, sans aucun texte autour.

Actions valides: MODIFY CREATE DELETE REFACTOR FIX ADD RENAME EXPLAIN CONVERT TEST OPTIMIZE EXTRACT MERGE SPLIT
Target types valides: function class file module endpoint test

Règles:
- description: reformule la demande en anglais, précise et technique
- target_name: nom exact du symbole ciblé (snake_case), ou "" si inconnu
- target_file: chemin relatif si mentionné, sinon null
- constraints: liste de contraintes techniques extraites (peut être vide)
- confidence: 0.0-1.0 selon la clarté de la demande

Format de sortie (JSON strict):
{
  "action": "...",
  "target_type": "...",
  "target_name": "...",
  "target_file": null,
  "description": "...",
  "constraints": [],
  "confidence": 0.9
}"""


@dataclass
class ExtractionResult:
    """Résultat brut de l'extraction avant validation."""

    intent: IntentJSON | None
    raw_json: str
    error: str | None


class IntentExtractor:
    """Extrait un IntentJSON structuré depuis une commande en français.

    Utilise Ollama comme backend LLM. Compatible avec le fine-tuning CamemBERT
    ultérieur — chaque extraction réussie est logguée pour constituer le dataset.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def extract(self, user_command: str) -> IntentJSON:
        """Transforme une commande française en IntentJSON structuré.

        Args:
            user_command: Commande en français saisie par l'utilisateur.

        Returns:
            IntentJSON validé. Si la confiance est < 0.75, needs_clarification=True.

        Raises:
            RuntimeError: Si Ollama est inaccessible et qu'aucun fallback n'est possible.
        """
        result = self._extract_via_ollama(user_command)

        if result.intent is not None:
            return result.intent

        logger.warning("Extraction échouée (%s) — fallback confiance 0.0", result.error)
        return IntentJSON(
            action="MODIFY",
            target_type="function",
            target_name="",
            description=user_command,
            confidence=0.0,
            constraints=(),
        )

    def _extract_via_ollama(self, command: str) -> ExtractionResult:
        prompt = f"{_SYSTEM_PROMPT}\n\nCommande: {command}\n\nJSON:"
        payload = json.dumps({
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 42},
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
                raw = str(data.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            return ExtractionResult(intent=None, raw_json="", error=str(exc))

        return self._parse_and_validate(raw)

    def _parse_and_validate(self, raw: str) -> ExtractionResult:
        # Extrait le premier bloc JSON de la réponse
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return ExtractionResult(intent=None, raw_json=raw, error="Pas de JSON trouvé")

        json_str = json_match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return ExtractionResult(intent=None, raw_json=json_str, error=str(exc))

        # Normalise et valide
        action = str(data.get("action", "MODIFY")).upper()
        target_type = str(data.get("target_type", "function")).lower()

        if action not in VALID_ACTIONS:
            action = "MODIFY"
        if target_type not in VALID_TARGET_TYPES:
            target_type = "function"

        raw_constraints = data.get("constraints", [])
        constraints: tuple[str, ...] = tuple(
            str(c) for c in raw_constraints if isinstance(raw_constraints, list)
        )

        try:
            intent = IntentJSON(
                action=action,
                target_type=target_type,
                target_name=str(data.get("target_name", "")),
                target_file=data.get("target_file") or None,
                description=str(data.get("description", "")),
                constraints=constraints,
                confidence=float(data.get("confidence", 0.5)),
            )
        except (ValueError, TypeError) as exc:
            return ExtractionResult(intent=None, raw_json=json_str, error=str(exc))

        return ExtractionResult(intent=intent, raw_json=json_str, error=None)
