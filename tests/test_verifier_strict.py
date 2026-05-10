"""Tests pour le sandbox contextuel et run_project_tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.shared.config import Config
from src.verifier.pipeline import VerificationPipeline
from src.verifier.sandbox import LocalSandbox, _write_project_files


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def sandbox(config: Config) -> LocalSandbox:
    return LocalSandbox(config)


# ── _write_project_files ──────────────────────────────────────────────────────

class TestWriteProjectFiles:
    def test_writes_context_files(self, tmp_path: Path) -> None:
        _write_project_files(
            tmp_path,
            code="def greet(name: str) -> str:\n    return f'Hello {name}'",
            target_filename="src/greet.py",
            project_files={"src/config.py": "DB = 'sqlite:///test.db'\n"},
            behavior_tests=[],
        )
        assert (tmp_path / "src" / "config.py").exists()
        assert (tmp_path / "src" / "greet.py").exists()

    def test_creates_init_files(self, tmp_path: Path) -> None:
        _write_project_files(
            tmp_path,
            code="x = 1",
            target_filename="pkg/module.py",
            project_files={},
            behavior_tests=[],
        )
        assert (tmp_path / "pkg" / "__init__.py").exists()

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        existing = tmp_path / "src"
        existing.mkdir()
        (existing / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
        _write_project_files(
            tmp_path,
            code="def bar(): pass",
            target_filename="src/mod.py",
            project_files={},
            behavior_tests=[],
        )
        content = (existing / "mod.py").read_text(encoding="utf-8")
        assert "foo" in content
        assert "bar" in content


# ── LocalSandbox.run_with_project_files ───────────────────────────────────────

class TestSandboxProjectFiles:
    def test_passing_test_with_context(self, sandbox: LocalSandbox) -> None:
        result = sandbox.run_with_project_files(
            code="def add(a, b): return a + b",
            target_filename="math_ops.py",
            behavior_tests=["def test_add():\n    from math_ops import add\n    assert add(1, 2) == 3"],
            project_files={},
        )
        assert result.exit_code == 0
        assert "1 passed" in result.stdout

    def test_failing_test_detected(self, sandbox: LocalSandbox) -> None:
        result = sandbox.run_with_project_files(
            code="def add(a, b): return a - b",  # bug intentionnel
            target_filename="math_ops.py",
            behavior_tests=["def test_add():\n    from math_ops import add\n    assert add(1, 2) == 3"],
            project_files={},
        )
        assert result.exit_code != 0

    def test_cross_file_import(self, sandbox: LocalSandbox) -> None:
        result = sandbox.run_with_project_files(
            code="from models import User\ndef get_name(u: User) -> str:\n    return u.name",
            target_filename="service.py",
            behavior_tests=[
                "def test_get_name():\n"
                "    from models import User\n"
                "    from service import get_name\n"
                "    u = User('Alice')\n"
                "    assert get_name(u) == 'Alice'"
            ],
            project_files={"models.py": "class User:\n    def __init__(self, name):\n        self.name = name\n"},
        )
        assert result.exit_code == 0


# ── run_project_tests ─────────────────────────────────────────────────────────

class TestRunProjectTests:
    def test_passing_project(self, sandbox: LocalSandbox, tmp_path: Path) -> None:
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text(
            "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))\n"
            "from calc import add\ndef test_add(): assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        result = sandbox.run_project_tests(tmp_path)
        assert result.exit_code == 0
        assert "1 passed" in result.stdout

    def test_failing_project(self, sandbox: LocalSandbox, tmp_path: Path) -> None:
        (tmp_path / "calc.py").write_text("def add(a, b): return a * b\n", encoding="utf-8")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text(
            "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))\n"
            "from calc import add\ndef test_add(): assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        result = sandbox.run_project_tests(tmp_path)
        assert result.exit_code != 0

    def test_empty_project(self, sandbox: LocalSandbox, tmp_path: Path) -> None:
        result = sandbox.run_project_tests(tmp_path)
        # Pas de test = pytest exit 5 (no tests collected)
        assert result.exit_code in (0, 4, 5)


# ── VerificationPipeline.run_project_tests ────────────────────────────────────

class TestPipelineProjectTests:
    def test_score_calculation(self, config: Config, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text("def mul(a, b): return a * b\n", encoding="utf-8")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_mod.py").write_text(
            "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))\n"
            "from mod import mul\n"
            "def test_ok(): assert mul(2, 3) == 6\n"
            "def test_fail(): assert mul(2, 3) == 99\n",
            encoding="utf-8",
        )
        pipeline = VerificationPipeline(config)
        res = pipeline.run_project_tests(tmp_path)
        assert res["passed"] == 1
        assert res["failed"] == 1
        assert res["score"] == pytest.approx(0.5)
