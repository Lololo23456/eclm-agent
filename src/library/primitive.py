"""Primitive dataclass et helpers de sérialisation."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Primitive:
    """Fonction atomique vérifiée et stockée dans la Primitive Library."""

    code: str
    tests: list[str]
    domain: str
    description: str
    language: str = "python"
    score: float = 1.0
    usage_count: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score invalide: {self.score}")
        if not self.code.strip():
            raise ValueError("Le code ne peut pas être vide")
        if not self.description.strip():
            raise ValueError("La description ne peut pas être vide")

    def to_metadata(self) -> dict[str, Any]:
        """Sérialise les champs non-vectoriels pour ChromaDB."""
        return {
            "code": self.code,
            "tests": "\n---\n".join(self.tests),
            "domain": self.domain,
            "language": self.language,
            "score": self.score,
            "usage_count": self.usage_count,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_metadata(cls, id: str, description: str, meta: dict[str, Any]) -> Primitive:
        """Reconstruit une Primitive depuis les métadonnées ChromaDB."""
        raw_tests: str = meta.get("tests", "")
        tests = [t for t in raw_tests.split("\n---\n") if t.strip()]
        return cls(
            id=id,
            code=str(meta["code"]),
            tests=tests,
            domain=str(meta["domain"]),
            description=description,
            language=str(meta.get("language", "python")),
            score=float(meta.get("score", 1.0)),
            usage_count=int(meta.get("usage_count", 0)),
            verified_at=str(meta.get("verified_at", "")),
        )
