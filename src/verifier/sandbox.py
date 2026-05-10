"""Sandbox d'exécution — Docker (isolé) ou Local (fallback léger)."""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.shared.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Résultat d'une exécution en sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    backend: str = "unknown"

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _write_test_files(workdir: Path, code: str, tests: list[str]) -> None:
    (workdir / "solution.py").write_text(code, encoding="utf-8")
    test_body = "\n\n".join(tests)
    test_code = f"import pytest\nfrom solution import *\n\n{test_body}\n"
    (workdir / "test_solution.py").write_text(test_code, encoding="utf-8")


def _write_project_files(
    workdir: Path,
    code: str,
    target_filename: str,
    project_files: dict[str, str],
    behavior_tests: list[str],
) -> None:
    """Reconstruit l'arborescence projet dans workdir pour un test contextualisé."""
    # Écrire les fichiers de contexte (dépendances)
    for rel_path, content in project_files.items():
        dest = workdir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    # Écrire le code candidat à sa position dans le projet
    target = workdir / target_filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        target.write_text(existing.rstrip() + "\n\n" + code + "\n", encoding="utf-8")
    else:
        target.write_text(code + "\n", encoding="utf-8")

    # Créer __init__.py manquants pour que les imports fonctionnent
    for p in workdir.rglob("*.py"):
        pkg = p.parent
        if pkg != workdir and not (pkg / "__init__.py").exists():
            (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Tests comportement — fichier séparé
    if behavior_tests:
        test_body = "\n\n".join(behavior_tests)
        (workdir / "test_behavior.py").write_text(
            f"import pytest\nimport sys, os\n"
            f"sys.path.insert(0, os.path.dirname(__file__))\n\n{test_body}\n",
            encoding="utf-8",
        )


def _run_pytest(workdir: Path, timeout: int, backend: str) -> SandboxResult:
    """Lance pytest dans workdir et retourne le résultat."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", ".", "-q", "--tb=short", "--no-header"],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(workdir),
        )
        return SandboxResult(
            exit_code=proc.returncode, stdout=proc.stdout,
            stderr=proc.stderr, backend=backend,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(exit_code=-1, stdout="", stderr="Timeout",
                             timed_out=True, backend=backend)


class DockerSandbox:
    """Exécute du code Python dans un container Docker isolé sans réseau.

    Mode sécurisé : --network=none, mémoire limitée, utilisateur non-root.
    Utiliser quand la sécurité est critique (code non fiable).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, code: str, tests: list[str]) -> SandboxResult:
        """Exécute code + tests dans un container Docker isolé.

        Args:
            code: Code Python source à évaluer.
            tests: Corps de fonctions pytest à exécuter.

        Returns:
            SandboxResult avec backend="docker".

        Raises:
            RuntimeError: Si Docker n'est pas disponible.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            _write_test_files(workdir, code, tests)
            return self._run_container(workdir)

    def run_with_project_files(
        self,
        code: str,
        target_filename: str,
        behavior_tests: list[str],
        project_files: dict[str, str],
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            _write_project_files(workdir, code, target_filename, project_files, behavior_tests)
            return self._run_container(workdir)

    def run_project_tests(self, output_dir: Path) -> SandboxResult:
        return self._run_container(output_dir)

    def _run_container(self, workdir: Path) -> SandboxResult:
        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory={self.config.docker_memory_limit}",
            "--cpus=1",
            f"--volume={workdir}:/workspace:ro",
            "--workdir=/workspace",
            self.config.docker_image,
            "python", "-m", "pytest", "test_solution.py", "-q", "--tb=short",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.config.docker_timeout_seconds,
            )
            return SandboxResult(
                exit_code=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, backend="docker",
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(exit_code=-1, stdout="", stderr="Timeout",
                                 timed_out=True, backend="docker")
        except FileNotFoundError as exc:
            raise RuntimeError("Docker non disponible") from exc


class LocalSandbox:
    """Exécute du code Python localement avec timeout — fallback sans Docker.

    Mode dégradé : pas d'isolation réseau ni mémoire, mais léger et rapide.
    Adapté au développement sur M3 Air où Docker coûte cher en batterie.
    Utiliser uniquement pour du code de confiance (code généré par l'agent).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, code: str, tests: list[str]) -> SandboxResult:
        """Exécute code + tests via subprocess Python local."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            _write_test_files(workdir, code, tests)
            return _run_pytest(workdir, self.config.docker_timeout_seconds, "local")

    def run_with_project_files(
        self,
        code: str,
        target_filename: str,
        behavior_tests: list[str],
        project_files: dict[str, str],
    ) -> SandboxResult:
        """Exécute code dans le contexte de l'arborescence projet.

        Args:
            code: Code candidat (une fonction/classe/module).
            target_filename: Chemin relatif du fichier cible dans le projet.
            behavior_tests: Tests comportement à exécuter.
            project_files: Dict {chemin_relatif: contenu} des fichiers du projet.

        Returns:
            SandboxResult avec backend="local_project".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            _write_project_files(workdir, code, target_filename, project_files, behavior_tests)
            return _run_pytest(workdir, self.config.docker_timeout_seconds, "local_project")

    def run_project_tests(self, output_dir: Path) -> SandboxResult:
        """Lance pytest directement sur le dossier output d'un projet généré.

        Args:
            output_dir: Dossier contenant les fichiers générés (avec tests/).

        Returns:
            SandboxResult avec le rapport pytest complet.
        """
        return _run_pytest(output_dir, self.config.docker_timeout_seconds * 2, "local_project_full")


def auto_sandbox(config: Config) -> DockerSandbox | LocalSandbox:
    """Sélectionne automatiquement le meilleur sandbox disponible.

    Préfère Docker pour l'isolation. Fallback sur LocalSandbox si Docker
    est absent ou si config.prefer_local_sandbox est True (M3 Air mode).

    Args:
        config: Configuration du projet.

    Returns:
        DockerSandbox ou LocalSandbox selon l'environnement.
    """
    if config.prefer_local_sandbox:
        logger.info("LocalSandbox sélectionné (prefer_local_sandbox=True)")
        return LocalSandbox(config)
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, timeout=3, check=False
        )
        logger.info("Docker disponible — DockerSandbox sélectionné")
        return DockerSandbox(config)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Docker absent — LocalSandbox activé (mode dégradé)")
        return LocalSandbox(config)
