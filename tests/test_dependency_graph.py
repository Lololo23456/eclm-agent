"""Tests pour src/orchestrator/dependency_graph.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestrator.dependency_graph import DependencyGraph, extract_public_api


@pytest.fixture
def tmp_py(tmp_path: Path):
    """Crée un fichier Python temporaire et retourne son chemin."""
    def _make(content: str) -> Path:
        p = tmp_path / "module.py"
        p.write_text(content, encoding="utf-8")
        return p
    return _make


# ── extract_public_api ────────────────────────────────────────────────────────

class TestExtractPublicApi:
    def test_function_signature(self, tmp_py):
        p = tmp_py("def add(x: int, y: int) -> int: return x + y\n")
        api = extract_public_api(p)
        assert "def add(x: int, y: int) -> int: ..." in api

    def test_private_function_excluded(self, tmp_py):
        p = tmp_py("def _helper(): pass\ndef public(): pass\n")
        api = extract_public_api(p)
        assert "_helper" not in api
        assert "def public" in api

    def test_class_with_init(self, tmp_py):
        p = tmp_py(
            "class Foo:\n"
            "    def __init__(self, x: int) -> None: ...\n"
            "    def bar(self) -> str: ...\n"
            "    def _private(self): ...\n"
        )
        api = extract_public_api(p)
        assert "class Foo:" in api
        assert "__init__" in api
        assert "def bar" in api
        assert "_private" not in api

    def test_upper_case_constant(self, tmp_py):
        p = tmp_py("MAX_SIZE = 100\nlower = 1\n")
        api = extract_public_api(p)
        assert "MAX_SIZE" in api
        assert "lower" not in api

    def test_imports_included(self, tmp_py):
        p = tmp_py("import os\nfrom pathlib import Path\n")
        api = extract_public_api(p)
        assert "import os" in api
        assert "from pathlib import Path" in api

    def test_function_docstring_hint(self, tmp_py):
        p = tmp_py(
            'def compute(n: int) -> int:\n'
            '    """Computes the result."""\n'
            '    return n * 2\n'
        )
        api = extract_public_api(p)
        assert "Computes the result" in api

    def test_invalid_file_returns_empty(self, tmp_path):
        p = tmp_path / "bad.py"
        p.write_text("def (broken syntax", encoding="utf-8")
        assert extract_public_api(p) == ""

    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "nonexistent.py"
        assert extract_public_api(p) == ""

    def test_class_bases(self, tmp_py):
        p = tmp_py("class Child(Base, Mixin): pass\n")
        api = extract_public_api(p)
        assert "class Child(Base, Mixin):" in api


# ── DependencyGraph ───────────────────────────────────────────────────────────

class TestDependencyGraph:
    def _make_task(self, index, depends_on=None, files_created=None):
        """Crée un faux TaskRecord sans importer project.py (évite effets de bord)."""
        from src.orchestrator.project import TaskRecord
        return TaskRecord(
            index=index,
            action="CREATE",
            target_type="class",
            target_name=f"Task{index}",
            target_file=f"src/t{index}.py",
            description="...",
            depends_on=depends_on or [],
            files_created=files_created or [],
        )

    def test_no_dependencies_returns_empty(self, tmp_path):
        dg = DependencyGraph()
        task = self._make_task(1, depends_on=[])
        assert dg.get_context_for_task(task, [task], tmp_path) == ""

    def test_dep_file_included_in_context(self, tmp_path):
        dep_file = tmp_path / "models.py"
        dep_file.write_text("class User:\n    def __init__(self): ...\n", encoding="utf-8")

        dg = DependencyGraph()
        dep_task = self._make_task(0, files_created=[str(dep_file)])
        cur_task = self._make_task(1, depends_on=[0])

        ctx = dg.get_context_for_task(cur_task, [dep_task, cur_task], tmp_path)
        assert "class User" in ctx
        assert "APIs disponibles" in ctx

    def test_non_py_file_excluded(self, tmp_path):
        dep_file = tmp_path / "data.json"
        dep_file.write_text('{"key": "value"}', encoding="utf-8")

        dg = DependencyGraph()
        dep_task = self._make_task(0, files_created=[str(dep_file)])
        cur_task = self._make_task(1, depends_on=[0])

        ctx = dg.get_context_for_task(cur_task, [dep_task, cur_task], tmp_path)
        assert ctx == ""

    def test_missing_file_skipped_gracefully(self, tmp_path):
        dg = DependencyGraph()
        dep_task = self._make_task(0, files_created=[str(tmp_path / "ghost.py")])
        cur_task = self._make_task(1, depends_on=[0])

        ctx = dg.get_context_for_task(cur_task, [dep_task, cur_task], tmp_path)
        assert ctx == ""

    def test_multiple_dep_files(self, tmp_path):
        f1 = tmp_path / "models.py"
        f1.write_text("class User: ...\n", encoding="utf-8")
        f2 = tmp_path / "config.py"
        f2.write_text("DB_URL = 'sqlite:///db'\n", encoding="utf-8")

        dg = DependencyGraph()
        t0 = self._make_task(0, files_created=[str(f1)])
        t1 = self._make_task(1, files_created=[str(f2)])
        t2 = self._make_task(2, depends_on=[0, 1])

        ctx = dg.get_context_for_task(t2, [t0, t1, t2], tmp_path)
        assert "models.py" in ctx
        assert "config.py" in ctx
