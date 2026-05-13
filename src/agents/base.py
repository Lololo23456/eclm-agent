"""Classe de base pour tous les agents spécialisés."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.shared.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Résultat standardisé d'un agent."""
    success: bool
    output: Any
    error: str | None = None
    tokens_used: int = 0


class BaseAgent(ABC):
    """Agent spécialisé — fait UNE chose, la fait bien."""

    def __init__(self, config: Config, model: str | None = None) -> None:
        self.config = config
        self.model = model or config.fast_model

    def _call_ollama(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        """Appel Ollama avec timeout strict."""
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 1024},
        }
        try:
            req = urllib.request.Request(
                f"{self.config.ollama_base_url}/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return str(json.loads(resp.read())["response"]).strip()
        except Exception as exc:
            logger.warning("%s Ollama error: %s", self.__class__.__name__, exc)
            return ""

    def _parse_json_response(self, raw: str) -> dict[str, Any] | None:
        """Extrait le JSON d'une réponse LLM (avec ou sans markdown)."""
        import re
        # Chercher un bloc ```json ... ``` ou {} direct
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
        else:
            # Trouver le premier { ... } équilibré
            start = raw.find("{")
            if start == -1:
                return None
            raw = raw[start:]
        try:
            return dict(json.loads(raw))
        except json.JSONDecodeError:
            return None

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        """Point d'entrée de l'agent."""
