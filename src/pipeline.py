"""AgentPipeline — coordinateur du pipeline multi-agents ECLM.

Workflow par tâche :
  [1] SpecWriter    → enrichit la spec (signature, imports, contraintes)
  [2] TestWriter    → génère les tests AVANT le code (isolation)
  [3] CodeWriter    → génère k candidats
  [4] Verifier      → score réel par exécution
  [5] Fixer         → patch si score < seuil (max 2 passes)
  [6] Integrator    → cohérence cross-fichiers (fin de projet)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.code_writer import CodeWriterAgent
from src.agents.fixer import FixerAgent
from src.agents.integrator import IntegratorAgent, IntegrationIssue
from src.agents.spec_writer import SpecWriterAgent
from src.agents.test_writer import TestWriterAgent
from src.shared.config import Config
from src.shared.types import ASTCandidate, ASTOperation, VerificationResult
from src.verifier.pipeline import VerificationPipeline

logger = logging.getLogger(__name__)

_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_DIM = "\033[2m"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _ANSI_RESET


@dataclass
class TaskResult:
    """Résultat d'une tâche individuelle."""
    task: dict[str, Any]
    success: bool
    code: str
    score: float
    tests: list[str] = field(default_factory=list)
    error: str | None = None
    fix_passes: int = 0
    written_to: Path | None = None


@dataclass
class PipelineResult:
    """Résultat complet d'un projet."""
    plan_name: str
    task_results: list[TaskResult]
    integration_issues: list[IntegrationIssue] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.success) / len(self.task_results)

    @property
    def avg_score(self) -> float:
        scores = [r.score for r in self.task_results]
        return sum(scores) / len(scores) if scores else 0.0


class AgentPipeline:
    """Exécute un plan.json via le pipeline d'agents spécialisés."""

    def __init__(self, config: Config, output_dir: Path | None = None) -> None:
        self.config = config
        self.output_dir = output_dir
        self._spec_writer = SpecWriterAgent(config)
        self._test_writer = TestWriterAgent(config)
        self._code_writer = CodeWriterAgent(config)
        self._fixer = FixerAgent(config)
        self._integrator = IntegratorAgent(config)
        self._verifier = VerificationPipeline(config)

    def run_from_plan(self, plan_path: Path) -> PipelineResult:
        """Exécute un plan.json complet, tâche par tâche."""
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        name = plan.get("name", plan_path.stem)

        tasks = plan.get("tasks", [])
        n = len(tasks)
        print(f"\n{_c('Pipeline', _ANSI_BOLD)} — {n} tâche(s) · plan: {name}\n")

        project_files: dict[str, str] = {}
        results: list[TaskResult] = []

        # Scaffolding initial
        self._scaffold_project(plan, self.output_dir)

        for task in sorted(tasks, key=lambda t: t.get("index", 0)):
            result = self._run_task(task, project_files, total=n)
            results.append(result)

            if result.success and result.written_to:
                # Accumuler le code généré comme contexte pour les tâches suivantes
                rel = str(result.written_to.relative_to(self.output_dir)) if self.output_dir else task["target_file"]
                project_files[rel] = result.code

        # Intégration cross-fichiers
        print(f"\n{_c('[Integrator]', _ANSI_DIM)} Vérification cohérence cross-fichiers...")
        int_result = self._integrator.run(project_files)
        issues: list[IntegrationIssue] = int_result.output or []
        if issues:
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                print(f"  {_c('⚠', _ANSI_YELLOW)} {len(errors)} problème(s) cross-fichiers:")
                for iss in errors[:3]:
                    print(f"    {iss.file}: {iss.problem}")
        else:
            print(f"  {_c('✓', _ANSI_GREEN)} Cohérence OK")

        pipeline_result = PipelineResult(
            plan_name=name,
            task_results=results,
            integration_issues=issues,
        )
        _print_summary(pipeline_result)
        return pipeline_result

    def _run_task(
        self,
        task: dict[str, Any],
        project_files: dict[str, str],
        total: int,
    ) -> TaskResult:
        idx = task.get("index", 0)
        name = task.get("target_name", "?")
        target_file = task.get("target_file", "unknown.py")
        prefix = f"[{idx + 1}/{total}]"

        print(f"  {_c(prefix, _ANSI_DIM)} {_c(task.get('action', 'CREATE'), _ANSI_BOLD)} "
              f"{target_file}:{name}", end="  ", flush=True)

        # [1] SpecWriter — enrichir la spec
        spec_result = self._spec_writer.run(task, project_files)
        enriched_task = spec_result.output if spec_result.success else task

        # [2] TestWriter — générer les tests (AVANT le code)
        test_result = self._test_writer.run(enriched_task, project_files)
        tests: list[str] = test_result.output if test_result.success else task.get("tests", [])

        # [3] CodeWriter + [4] Verifier + [5] Fixer (loop)
        best_code = ""
        best_score = 0.0
        last_error: str | None = None
        fix_passes = 0
        max_fixes = self.config.max_retries

        for attempt in range(max_fixes + 1):
            # Générer candidats
            write_result = self._code_writer.run(
                enriched_task, tests, project_files,
                k=3 if attempt == 0 else 1,
                error=last_error,
            )
            candidates = write_result.output if write_result.success else []

            if not candidates:
                break

            # Vérifier le meilleur (avec contexte des fichiers déjà générés)
            best_candidate, score, error = self._verify_candidates(
                candidates, tests, target_file, project_files=project_files
            )
            if score > best_score:
                best_score = score
                best_code = best_candidate

            if score >= self.config.min_verification_score:
                break

            if attempt < max_fixes:
                fix_passes += 1
                last_error = error
                # [5] Fixer — patch ciblé
                fix_result = self._fixer.run(best_code, error or "score trop bas", enriched_task, tests)
                if fix_result.success:
                    fixed_code = fix_result.output
                    _, fixed_score, _ = self._verify_candidates(
                        [fixed_code], tests, target_file, project_files=project_files
                    )
                    if fixed_score > best_score:
                        best_score = fixed_score
                        best_code = fixed_code

        success = best_score >= self.config.min_verification_score
        icon = _c("✓", _ANSI_GREEN) if success else _c("✗", _ANSI_RED)
        print(f"{icon} score={best_score:.2f}")

        # Écrire le fichier si output_dir défini
        written_to: Path | None = None
        if best_code and self.output_dir:
            written_to = self.output_dir / target_file
            written_to.parent.mkdir(parents=True, exist_ok=True)
            written_to.write_text(best_code, encoding="utf-8")

        return TaskResult(
            task=task,
            success=success,
            code=best_code,
            score=best_score,
            tests=tests,
            error=last_error if not success else None,
            fix_passes=fix_passes,
            written_to=written_to,
        )

    def _verify_candidates(
        self,
        candidates: list[str],
        tests: list[str],
        target_file: str,
        project_files: dict[str, str] | None = None,
    ) -> tuple[str, float, str | None]:
        """Vérifie les candidats, retourne (meilleur_code, meilleur_score, erreur)."""
        ast_candidates = [
            ASTCandidate(
                code=c,
                operation=ASTOperation(op_type="CREATE_FUNCTION", target=target_file),
                generation_rank=i,
            )
            for i, c in enumerate(candidates)
        ]
        # target_filename = juste le nom du fichier (ex: "models.py")
        target_filename = Path(target_file).name
        result: VerificationResult = self._verifier.verify(
            ast_candidates,
            behavior_tests=tests,
            impl_tests=[],
            project_files=project_files if project_files else None,
            target_filename=target_filename,
        )
        error = result.error_message
        return result.candidate.code, result.composite_score, error

    def _scaffold_project(self, plan: dict[str, Any], output_dir: Path | None) -> None:
        """Crée les fichiers de scaffolding (conftest.py, __init__.py, pyproject.toml)."""
        if not output_dir:
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        # conftest.py
        conftest = output_dir / "conftest.py"
        if not conftest.exists():
            conftest.write_text("import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n")

        # pyproject.toml minimal
        pyproject = output_dir / "pyproject.toml"
        if not pyproject.exists():
            deps = plan.get("stack", {}).get("dependencies", [])
            deps_str = "\n".join(f'  "{d}",' for d in deps)
            pyproject.write_text(
                f'[project]\nname = "{plan.get("name", "project")}"\n'
                f'version = "0.1.0"\ndependencies = [\n{deps_str}\n]\n'
            )

        # __init__.py pour src/
        src = output_dir / "src"
        src.mkdir(exist_ok=True)
        init = src / "__init__.py"
        if not init.exists():
            init.write_text("")


def _print_summary(result: PipelineResult) -> None:
    ok = sum(1 for r in result.task_results if r.success)
    total = len(result.task_results)
    avg = result.avg_score
    errors = [i for i in result.integration_issues if i.severity == "error"]

    print(f"\n{'─' * 52}")
    print(f"  {_c(result.plan_name, _ANSI_BOLD)} : {ok}/{total} tâches  avg_score={avg:.2f}")
    if errors:
        print(f"  {_c('⚠', _ANSI_YELLOW)} {len(errors)} problème(s) d'intégration")
    print(f"{'─' * 52}\n")
