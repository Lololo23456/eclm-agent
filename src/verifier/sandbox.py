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
        """Exécute code + tests via subprocess Python local.

        Args:
            code: Code Python source à évaluer.
            tests: Corps de fonctions pytest à exécuter.

        Returns:
            SandboxResult avec backend="local".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            _write_test_files(workdir, code, tests)
            return self._run_local(workdir)

    def _run_local(self, workdir: Path) -> SandboxResult:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=short"],
                capture_output=True, text=True,
                timeout=self.config.docker_timeout_seconds,
                cwd=str(workdir),
            )
            return SandboxResult(
                exit_code=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, backend="local",
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(exit_code=-1, stdout="", stderr="Timeout",
                                 timed_out=True, backend="local")


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
