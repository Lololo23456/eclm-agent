"""ECLMAgent — point d'entrée principal du pipeline ECLM complet."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.eclm.beam_search import filter_and_rank
from src.eclm.model import ECLMCore
from src.improvement.dpo_collector import DPOCollector, RunRecord
from src.library.retrieval import PrimitiveRetriever
from src.orchestrator.context import ASTContext
from src.orchestrator.rag import CodebaseIndex
from src.orchestrator.writer import FileWriter
from src.planner.model import ASTPlanner
from src.shared.config import Config
from src.shared.types import ASTCandidate, IntentJSON, VerificationResult
from src.verifier.pipeline import VerificationPipeline

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Réponse finale de l'agent à l'utilisateur."""

    success: bool
    code: str
    score: float
    message: str
    intent: IntentJSON
    retries_used: int = 0
    written_to: Path | None = None


class ECLMAgent:
    """Orchestre le pipeline C0→C1→C2→C3 avec collecte DPO automatique.

    Pipeline complet :
    C0 (intent)  →  déjà appliqué avant run()
    C1 (planner) →  décompose en ASTOperationPlan
    C2 (eclm)    →  génère k candidats par opération
    beam_search  →  filtre syntaxe + re-rank lint
    C3 (verifier)→  score composite + sandbox Docker
    DPO          →  collecte paires failure→success automatiquement
    """

    def __init__(self, config: Config, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._eclm = ECLMCore(config)
        self._planner = ASTPlanner(config)
        self._verifier = VerificationPipeline(config)
        self._retriever = PrimitiveRetriever(config)
        self._rag = CodebaseIndex(config, project_root)
        self._dpo = DPOCollector(config)
        self._writer = FileWriter()

    def run(
        self,
        intent: IntentJSON,
        behavior_tests: list[str] | None = None,
    ) -> AgentResponse:
        """Exécute le pipeline complet pour une intention validée.

        Args:
            intent: IntentJSON produit et validé par C0.
            behavior_tests: Tests comportement fournis par l'utilisateur.

        Returns:
            AgentResponse avec le code validé ou le meilleur candidat.
        """
        if intent.needs_clarification:
            return AgentResponse(
                success=False,
                code="",
                score=0.0,
                message=self._clarification_question(intent),
                intent=intent,
            )

        context = self._rag.get_context(intent)
        primitives = self._retriever.retrieve(intent)
        if primitives:
            logger.info("%d primitive(s) pertinente(s) récupérée(s)", len(primitives))

        # C1 — Plan
        plan = self._planner.plan(intent, context)
        logger.info("Plan: %d opération(s) (complexité=%d)", len(plan), plan.estimated_complexity)

        tests = behavior_tests or []
        dpo_record = RunRecord(
            command=f"{intent.action} {intent.target_name}: {intent.description}",
            intent=intent,
        )

        best: VerificationResult | None = None
        error: str | None = None

        for attempt in range(self.config.max_retries):
            # C2 — Génération (toutes les opérations du plan)
            all_candidates: list[ASTCandidate] = []
            for operation in plan.operations:
                raw = self._eclm.generate(
                    operation, context, error=error, k=self.config.beam_width
                )
                all_candidates.extend(raw)

            # Beam search — filtre syntaxe + re-rank
            candidates = filter_and_rank(all_candidates, top_k=self.config.beam_width)

            # C3 — Vérification
            result = self._verifier.verify(candidates, behavior_tests=tests)
            dpo_record.add(result.candidate.code, result.composite_score)

            if best is None or result.composite_score > best.composite_score:
                best = result

            if result.passes:
                self._dpo.collect(dpo_record)
                written = self._writer.write(result.candidate.code, intent, self.project_root)
                if written:
                    self._rag.index_file(written)
                logger.info(
                    "Validé en %d essai(s), score=%.3f", attempt + 1, result.composite_score
                )
                return AgentResponse(
                    success=True,
                    code=result.candidate.code,
                    score=result.composite_score,
                    message=f"Code validé (score={result.composite_score:.2f})",
                    intent=intent,
                    retries_used=attempt,
                    written_to=written,
                )

            error = result.error_message or f"Score insuffisant: {result.composite_score:.2f}"
            logger.warning("Tentative %d — score=%.3f — %s", attempt + 1, result.composite_score, error)

        self._dpo.collect(dpo_record)
        assert best is not None
        return AgentResponse(
            success=False,
            code=best.candidate.code,
            score=best.composite_score,
            message=f"Score insuffisant après {self.config.max_retries} essais ({best.composite_score:.2f})",
            intent=intent,
            retries_used=self.config.max_retries,
        )

    def index_project(self) -> int:
        """Indexe le codebase courant dans ChromaDB.

        Returns:
            Nombre de chunks indexés.
        """
        return self._rag.index_project()

    def _clarification_question(self, intent: IntentJSON) -> str:
        target = intent.target_name or "la cible"
        return (
            f"Intention peu claire (confiance: {intent.confidence:.0%}). "
            f"Que voulez-vous faire exactement avec `{target}` ?"
        )
