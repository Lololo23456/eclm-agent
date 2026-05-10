"""Pipeline de vérification multi-couches pour les candidats ECLM."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from src.shared.config import Config
from src.shared.types import ASTCandidate, VerificationResult
from src.verifier.sandbox import DockerSandbox, LocalSandbox, SandboxResult, auto_sandbox
from src.verifier.scorer import LintScorer, SyntaxChecker, TypeChecker

logger = logging.getLogger(__name__)

# Pattern pytest summary: "3 passed, 1 failed" ou "5 passed"
_PYTEST_SUMMARY_RE = re.compile(
    r"(?:(\d+) passed)?.*?(?:(\d+) failed)?.*?(?:(\d+) error)?",
    re.DOTALL,
)


def _parse_pytest_score(stdout: str) -> float:
    """Calcule un score partiel à partir de la sortie pytest.

    Args:
        stdout: Sortie texte de pytest.

    Returns:
        Score entre 0.0 et 1.0 basé sur le ratio tests passés / total.
    """
    passed = 0
    failed = 0
    for line in stdout.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m:
            failed += int(m.group(1))
        m = re.search(r"(\d+) error", line)
        if m:
            failed += int(m.group(1))
    total = passed + failed
    return passed / total if total > 0 else 0.0


class VerificationPipeline:
    """Orchestre les couches de vérification et sélectionne le meilleur candidat.

    Layers dans l'ordre :
    1. Syntax check (ast.parse)
    2. Type check (mypy --strict)
    3. Lint score (ruff)
    4. Tests comportement (fournis par l'utilisateur)
    5. Tests implémentation (générés par TestGenerator)
    6. Property tests (Hypothesis)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._syntax = SyntaxChecker()
        self._types = TypeChecker()
        self._lint = LintScorer()
        self._sandbox: DockerSandbox | LocalSandbox = auto_sandbox(config)

    def run_project_tests(self, output_dir: Path) -> dict[str, object]:
        """Lance pytest sur l'intégralité du dossier projet généré.

        Args:
            output_dir: Dossier racine du projet généré.

        Returns:
            Dict avec passed, failed, errors, score, stdout.
        """
        result = self._sandbox.run_project_tests(output_dir)
        passed = 0
        failed = 0
        for line in result.stdout.splitlines():
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+) (?:failed|error)", line)
            if m:
                failed += int(m.group(1))
        total = passed + failed
        score = passed / total if total > 0 else (0.0 if result.exit_code != 0 else 1.0)
        return {
            "passed": passed,
            "failed": failed,
            "total": total,
            "score": score,
            "stdout": result.stdout,
            "timed_out": result.timed_out,
        }

    def verify(
        self,
        candidates: list[ASTCandidate],
        behavior_tests: list[str],
        impl_tests: list[str] | None = None,
        property_tests: list[str] | None = None,
        project_files: dict[str, str] | None = None,
        target_filename: str = "solution.py",
    ) -> VerificationResult:
        """Vérifie tous les candidats et retourne le meilleur résultat.

        Args:
            candidates: Candidats ECLM à évaluer (max beam_width).
            behavior_tests: Tests comportement fournis par l'utilisateur.
            impl_tests: Tests d'implémentation générés par TestGenerator.
            property_tests: Tests de propriété Hypothesis.

        Returns:
            VerificationResult du candidat avec le score composite le plus élevé.

        Raises:
            ValueError: Si candidates est vide.
        """
        if not candidates:
            raise ValueError("La liste de candidats est vide")

        results = [
            self._verify_one(
                candidate=c,
                behavior_tests=behavior_tests,
                impl_tests=impl_tests or [],
                property_tests=property_tests or [],
                project_files=project_files,
                target_filename=target_filename,
            )
            for c in candidates
        ]
        best = max(results, key=lambda r: r.composite_score)
        logger.info(
            "Meilleur candidat: score=%.3f syntax=%s mypy=%s",
            best.composite_score,
            best.syntax_ok,
            best.mypy_ok,
        )
        return best

    def _verify_one(
        self,
        candidate: ASTCandidate,
        behavior_tests: list[str],
        impl_tests: list[str],
        property_tests: list[str],
        project_files: dict[str, str] | None = None,
        target_filename: str = "solution.py",
    ) -> VerificationResult:
        syntax_ok = self._syntax.check(candidate.code)
        if not syntax_ok:
            return VerificationResult(
                candidate=candidate,
                syntax_ok=False,
                mypy_ok=False,
                lint_score=0.0,
                behavior_tests_score=0.0,
                impl_tests_score=0.0,
                property_tests_score=0.0,
                error_message="SyntaxError",
            )

        mypy_ok, mypy_error = self._types.check(candidate.code)
        lint_score = self._lint.score(candidate.code)

        if project_files and behavior_tests:
            behavior_score = self._run_tests_in_project(
                candidate.code, behavior_tests, project_files, target_filename
            )
        else:
            behavior_score = self._run_tests(candidate.code, behavior_tests)

        impl_score = self._run_tests(candidate.code, impl_tests) if impl_tests else 1.0
        property_score = (
            self._run_tests(candidate.code, property_tests) if property_tests else 1.0
        )

        return VerificationResult(
            candidate=candidate,
            syntax_ok=syntax_ok,
            mypy_ok=mypy_ok,
            lint_score=lint_score,
            behavior_tests_score=behavior_score,
            impl_tests_score=impl_score,
            property_tests_score=property_score,
            error_message=mypy_error or None,
        )

    def _run_tests(self, code: str, tests: list[str]) -> float:
        if not tests:
            return 1.0
        result: SandboxResult = self._sandbox.run(code, tests)
        if result.timed_out:
            return 0.0
        return _parse_pytest_score(result.stdout)

    def _run_tests_in_project(
        self,
        code: str,
        tests: list[str],
        project_files: dict[str, str],
        target_filename: str,
    ) -> float:
        result: SandboxResult = self._sandbox.run_with_project_files(
            code, target_filename, tests, project_files
        )
        if result.timed_out:
            return 0.0
        return _parse_pytest_score(result.stdout)
