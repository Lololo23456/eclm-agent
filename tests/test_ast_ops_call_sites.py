"""Tests pour UPDATE_CALL_SITES dans ASTOperationExecutor."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.eclm.ast_ops import ASTOperationExecutor
from src.shared.types import ASTOperation


@pytest.fixture
def executor() -> ASTOperationExecutor:
    return ASTOperationExecutor()


def _op(old: str, new: str) -> ASTOperation:
    return ASTOperation(op_type="UPDATE_CALL_SITES", target=old, params={"new_name": new})


class TestUpdateCallSitesDeterministic:
    def test_is_deterministic(self, executor: ASTOperationExecutor) -> None:
        assert executor.is_deterministic("UPDATE_CALL_SITES")

    def test_renames_function_call(self, executor: ASTOperationExecutor) -> None:
        code = "result = old_func(x, y)"
        out = executor.apply(code, _op("old_func", "new_func"))
        assert "new_func" in out
        assert "old_func" not in out

    def test_renames_name_reference(self, executor: ASTOperationExecutor) -> None:
        code = "x = old_name\nprint(old_name)"
        out = executor.apply(code, _op("old_name", "new_name"))
        assert out.count("new_name") == 2
        assert "old_name" not in out

    def test_renames_attribute_access(self, executor: ASTOperationExecutor) -> None:
        code = "obj.old_method()\nresult = obj.old_method"
        out = executor.apply(code, _op("old_method", "new_method"))
        assert "new_method" in out
        assert "old_method" not in out

    def test_renames_import_from(self, executor: ASTOperationExecutor) -> None:
        code = "from models import old_name\nx = old_name()"
        out = executor.apply(code, _op("old_name", "new_name"))
        assert "from models import new_name" in out

    def test_does_not_rename_unrelated_names(self, executor: ASTOperationExecutor) -> None:
        code = "other_func()\nold_func_extended()"
        out = executor.apply(code, _op("old_func", "new_func"))
        assert "other_func" in out
        # old_func_extended ne doit PAS être renommé (match exact)
        assert "old_func_extended" in out

    def test_empty_code_unchanged(self, executor: ASTOperationExecutor) -> None:
        out = executor.apply("", _op("foo", "bar"))
        assert out == ""


class TestUpdateCallSitesInProject:
    def test_modifies_files_with_old_name(self, executor: ASTOperationExecutor, tmp_path: Path) -> None:
        f1 = tmp_path / "caller.py"
        f1.write_text("from utils import old_func\nold_func(1, 2)\n", encoding="utf-8")
        f2 = tmp_path / "other.py"
        f2.write_text("def unrelated(): pass\n", encoding="utf-8")

        modified = executor.update_call_sites_in_project(tmp_path, "old_func", "new_func")

        assert f1 in modified
        assert f2 not in modified
        assert "new_func" in f1.read_text(encoding="utf-8")

    def test_skips_invalid_syntax_files(self, executor: ASTOperationExecutor, tmp_path: Path) -> None:
        bad = tmp_path / "bad.py"
        bad.write_text("def broken(:\n", encoding="utf-8")
        # Ne doit pas lever d'exception
        modified = executor.update_call_sites_in_project(tmp_path, "x", "y")
        assert bad not in modified

    def test_returns_empty_for_no_matches(self, executor: ASTOperationExecutor, tmp_path: Path) -> None:
        (tmp_path / "clean.py").write_text("def foo(): return 1\n", encoding="utf-8")
        modified = executor.update_call_sites_in_project(tmp_path, "nonexistent", "other")
        assert modified == []
