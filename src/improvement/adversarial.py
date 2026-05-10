"""Self-play adversarial — Generator vs Critic pour DPO automatique.

Génère des paires DPO sans intervention humaine :
1. Pour chaque tâche (IntentJSON), génère N candidats à températures variées.
2. Score chacun via VerificationPipeline.
3. Paire (meilleur, pire) → DPOCollector.

run_from_sessions() alimente le self-play depuis les sessions projet existantes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.improvement.dpo_collector import DPOCollector, RunRecord
from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)

_TEMPERATURES = [0.2, 0.5, 0.8, 1.0]  # températures pour diversité des candidats


class AdversarialLoop:
    """Generator vs Critic : génère des paires DPO sans intervention humaine.

    Stratégie :
    1. Generator produit plusieurs candidats par température pour une tâche.
    2. Critic (verifier) les score.
    3. Les paires (meilleur, pire) alimentent le DPO collector.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._collector = DPOCollector(config)

    def run_episode(self, intent: IntentJSON, candidates: list[tuple[str, float]]) -> bool:
        """Crée une paire DPO depuis les candidats d'un épisode.

        Args:
            intent: Intention de la tâche.
            candidates: Liste de (code, score) — n'importe quel ordre.

        Returns:
            True si une paire a été enregistrée.
        """
        if len(candidates) < 2:
            return False

        command = f"{intent.action} {intent.target_name}: {intent.description}"
        record = RunRecord(command=command, intent=intent)
        for code, score in candidates:
            record.add(code, score)

        return self._collector.collect(record)

    def run_batch(self, tasks: list[dict[str, object]]) -> int:
        """Lance un batch de self-play : génère des candidats, les score, crée des paires DPO.

        Chaque task doit contenir :
          - "intent": IntentJSON
          - "behavior_tests" (optionnel): list[str]
          - "current_code" (optionnel): str — contexte pour les ops MODIFY

        Args:
            tasks: Liste de tâches à traiter.

        Returns:
            Nombre de paires DPO créées.
        """
        # Import tardif pour éviter les imports circulaires
        from src.eclm.model import ECLMCore
        from src.eclm.beam_search import filter_and_rank
        from src.orchestrator.context import ASTContext
        from src.planner.model import ASTPlanner
        from src.verifier.pipeline import VerificationPipeline

        eclm = ECLMCore(self.config)
        planner = ASTPlanner(self.config)
        verifier = VerificationPipeline(self.config)

        pairs_created = 0

        for task_dict in tasks:
            intent = task_dict["intent"]
            if not isinstance(intent, IntentJSON):
                try:
                    intent = IntentJSON(**dict(intent))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    logger.warning("Intent invalide, tâche ignorée: %s", exc)
                    continue

            behavior_tests = list(task_dict.get("behavior_tests", []))  # type: ignore[arg-type]
            current_code = str(task_dict.get("current_code", ""))
            context = ASTContext(dependency_context=current_code) if current_code else ASTContext()

            try:
                plan = planner.plan(intent, context)
            except Exception as exc:
                logger.warning("Plan échoué pour %s: %s", intent.target_name, exc)
                continue

            # Générer des candidats à différentes températures pour plus de diversité
            all_scored: list[tuple[str, float]] = []
            for temperature in _TEMPERATURES:
                try:
                    for operation in plan.operations:
                        raw = eclm.generate(
                            operation, context, k=2, complexity="medium",
                        )
                        ranked = filter_and_rank(raw, top_k=2)
                        if not ranked:
                            continue
                        result = verifier.verify(ranked, behavior_tests=behavior_tests)
                        all_scored.append((result.candidate.code, result.composite_score))
                except Exception as exc:
                    logger.debug("Génération température=%.1f échouée: %s", temperature, exc)
                    continue

            if len(all_scored) >= 2:
                if self.run_episode(intent, all_scored):
                    pairs_created += 1
                    logger.info(
                        "Paire DPO créée: %s (best=%.2f worst=%.2f)",
                        intent.target_name,
                        max(s for _, s in all_scored),
                        min(s for _, s in all_scored),
                    )

        logger.info("Self-play terminé: %d/%d paires créées", pairs_created, len(tasks))
        return pairs_created

    def run_from_sessions(self, sessions_dir: Path, max_sessions: int = 10) -> int:
        """Alimente le self-play depuis les sessions projet existantes.

        Extrait les tâches réussies des sessions et rejoue le self-play
        pour créer de nouvelles paires DPO avec diversité de températures.

        Args:
            sessions_dir: Répertoire data/sessions/ contenant les JSON de sessions.
            max_sessions: Nombre maximum de sessions à traiter.

        Returns:
            Nombre total de paires DPO créées.
        """
        session_files = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        session_files = session_files[:max_sessions]

        if not session_files:
            logger.warning("Aucune session trouvée dans %s", sessions_dir)
            return 0

        tasks: list[dict[str, object]] = []
        for session_file in session_files:
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Session %s illisible: %s", session_file.name, exc)
                continue

            for task in data.get("tasks", []):
                if task.get("status") != "done":
                    continue
                try:
                    intent = IntentJSON(
                        action=str(task.get("action", "CREATE")),
                        target_type=str(task.get("target_type", "function")),
                        target_name=str(task.get("target_name", task.get("label", "unknown"))),
                        target_file=task.get("target_file"),
                        description=str(task.get("label", "")),
                        confidence=0.9,
                    )
                    tasks.append({"intent": intent, "behavior_tests": [], "current_code": ""})
                except (TypeError, ValueError) as exc:
                    logger.debug("Tâche ignorée: %s", exc)

        logger.info(
            "%d tâches extraites depuis %d sessions → self-play",
            len(tasks), len(session_files),
        )
        return self.run_batch(tasks)

    def run_from_dpo_prompts(self, dpo_dir: Path, max_pairs: int = 50) -> int:
        """Rejoue le self-play depuis les prompts DPO existants pour augmenter la diversité.

        Args:
            dpo_dir: Répertoire data/dpo_pairs/ contenant les JSONL.
            max_pairs: Nombre maximum de paires à rejouer.

        Returns:
            Nombre de nouvelles paires DPO créées.
        """
        tasks: list[dict[str, object]] = []
        count = 0

        for dpo_file in sorted(dpo_dir.glob("dpo_*.jsonl")):
            if count >= max_pairs:
                break
            try:
                with open(dpo_file, encoding="utf-8") as f:
                    for line in f:
                        if count >= max_pairs:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        pair = json.loads(line)
                        prompt_str = str(pair.get("prompt", ""))
                        if not prompt_str:
                            continue
                        # Reconstruire un IntentJSON minimal depuis le prompt DPO
                        parts = prompt_str.split(":", 1)
                        action_target = parts[0].strip().split()
                        action = action_target[0].upper() if action_target else "CREATE"
                        target_name = action_target[-1] if len(action_target) > 1 else "target"
                        try:
                            intent = IntentJSON(
                                action=action if action in {
                                    "MODIFY", "CREATE", "DELETE", "REFACTOR", "FIX",
                                    "ADD", "RENAME", "EXPLAIN", "CONVERT", "TEST",
                                    "OPTIMIZE", "EXTRACT", "MERGE", "SPLIT",
                                } else "CREATE",
                                target_type="function",
                                target_name=target_name,
                                description=parts[1].strip() if len(parts) > 1 else prompt_str,
                                confidence=0.9,
                            )
                            tasks.append({"intent": intent, "current_code": str(pair.get("chosen", ""))})
                            count += 1
                        except ValueError:
                            continue
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Fichier DPO illisible: %s", exc)

        logger.info("%d prompts DPO extraits → self-play", len(tasks))
        return self.run_batch(tasks)
