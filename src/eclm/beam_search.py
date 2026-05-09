"""Beam search sur les candidats AST — filtrage rapide et re-ranking."""
from __future__ import annotations

import logging

from src.shared.types import ASTCandidate
from src.verifier.scorer import LintScorer, SyntaxChecker

logger = logging.getLogger(__name__)

_syntax = SyntaxChecker()
_lint = LintScorer()


def filter_and_rank(candidates: list[ASTCandidate], top_k: int = 5) -> list[ASTCandidate]:
    """Filtre les candidats syntaxiquement invalides et les re-classe par qualité.

    Couches appliquées dans l'ordre :
    1. Syntax check (élimine immédiatement, ~0ms)
    2. Lint score ruff (re-rank les survivants)

    Args:
        candidates: Candidats bruts produits par ECLMCore.
        top_k: Nombre maximum de candidats à retourner.

    Returns:
        Candidats valides triés par lint_score décroissant (max top_k).
    """
    valid: list[tuple[float, ASTCandidate]] = []
    for c in candidates:
        if not _syntax.check(c.code):
            logger.debug("Candidat rank=%d éliminé: syntaxe invalide", c.generation_rank)
            continue
        score = _lint.score(c.code)
        valid.append((score, c))

    valid.sort(key=lambda x: x[0], reverse=True)
    ranked = [c for _, c in valid[:top_k]]

    if not ranked and candidates:
        # Dernier recours : retourner le premier candidat même invalide
        ranked = [candidates[0]]

    return ranked
