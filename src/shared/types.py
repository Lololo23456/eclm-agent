"""Types partagés entre tous les composants ECLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_ACTIONS = frozenset({
    "MODIFY", "CREATE", "DELETE", "REFACTOR", "FIX",
    "ADD", "RENAME", "EXPLAIN", "CONVERT", "TEST",
    "OPTIMIZE", "EXTRACT", "MERGE", "SPLIT",
})

VALID_TARGET_TYPES = frozenset({
    "function", "class", "file", "module", "endpoint", "test",
})


@dataclass(frozen=True)
class IntentJSON:
    """Sortie de C0 IntentExtractor."""
    action: str
    target_type: str
    target_name: str
    description: str
    confidence: float
    target_file: str | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"Action invalide: {self.action}")
        if self.target_type not in VALID_TARGET_TYPES:
            raise ValueError(f"Target type invalide: {self.target_type}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence invalide: {self.confidence}")

    @property
    def needs_clarification(self) -> bool:
        return self.confidence < 0.75


VALID_OP_TYPES = frozenset({
    "ADD_PARAM", "MODIFY_BODY", "REMOVE_PARAM", "ADD_RETURN_TYPE",
    "RENAME_SYMBOL", "ADD_IMPORT", "CREATE_FUNCTION", "CREATE_CLASS",
    "ADD_METHOD", "DELETE_NODE", "UPDATE_CALL_SITES", "ADD_DECORATOR",
    "EXTRACT_FUNCTION", "ADD_DOCSTRING", "MODIFY_DECORATOR",
})


@dataclass(frozen=True)
class ASTOperation:
    """Opération atomique sur l'AST."""
    op_type: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op_type not in VALID_OP_TYPES:
            raise ValueError(f"Op type invalide: {self.op_type}")

    def __hash__(self) -> int:
        return hash((self.op_type, self.target, str(sorted(self.params.items()))))


@dataclass(frozen=True)
class ASTOperationPlan:
    """Sortie de C1 ASTPlanner."""
    operations: tuple[ASTOperation, ...]
    intent: IntentJSON
    estimated_complexity: int

    def __len__(self) -> int:
        return len(self.operations)


@dataclass
class ASTCandidate:
    """Candidat généré par l'ECLM."""
    code: str
    operation: ASTOperation
    score: float = 0.0
    generation_rank: int = 0


@dataclass
class VerificationResult:
    """Résultat de la pipeline de vérification."""
    candidate: ASTCandidate
    syntax_ok: bool
    mypy_ok: bool
    lint_score: float
    behavior_tests_score: float
    impl_tests_score: float
    property_tests_score: float
    error_message: str | None = None

    @property
    def composite_score(self) -> float:
        if not self.syntax_ok or not self.mypy_ok:
            return 0.0
        return (
            0.4 * self.behavior_tests_score
            + 0.3 * self.impl_tests_score
            + 0.2 * self.property_tests_score
            + 0.1 * self.lint_score
        )

    @property
    def passes(self) -> bool:
        return self.composite_score >= 0.8


@dataclass
class Primitive:
    """Primitive vérifiée dans la Library."""
    id: str
    code: str
    tests: list[str]
    domain: str
    description: str
    language: str = "python"
    score: float = 1.0
    usage_count: int = 0
    verified_at: str = ""


@dataclass(frozen=True)
class DPOPair:
    """Paire de préférence pour DPO fine-tuning."""
    prompt: str
    chosen: str
    rejected: str
    source: str
    chosen_score: float
    rejected_score: float
    timestamp: str

    def __post_init__(self) -> None:
        if self.chosen_score <= self.rejected_score:
            raise ValueError("chosen_score doit être > rejected_score")
        if self.chosen_score < 0.8:
            raise ValueError("chosen_score doit être >= 0.8")
