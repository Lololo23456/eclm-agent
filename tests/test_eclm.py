"""Tests pour src/eclm/."""
from __future__ import annotations

import ast

import pytest

from src.eclm.ast_ops import ASTOperationExecutor, LLMRequiredError
from src.eclm.beam_search import filter_and_rank
from src.shared.types import ASTCandidate, ASTOperation

_exec = ASTOperationExecutor()

_BASE = "def foo(x: int) -> int:\n    return x\n"
_CLASS = "class Bar:\n    def method(self) -> None:\n        pass\n"


def _op(op_type: str, target: str = "foo", **params: object) -> ASTOperation:
    return ASTOperation(op_type=op_type, target=target, params=dict(params))


def _candidate(code: str, rank: int = 0) -> ASTCandidate:
    return ASTCandidate(code=code, operation=_op("MODIFY_BODY"), generation_rank=rank)


# ── is_deterministic ─────────────────────────────────────────────────────────

class TestIsDeterministic:
    def test_add_param_is_deterministic(self) -> None:
        assert _exec.is_deterministic("ADD_PARAM") is True

    def test_modify_body_is_not(self) -> None:
        assert _exec.is_deterministic("MODIFY_BODY") is False

    def test_create_function_is_not(self) -> None:
        assert _exec.is_deterministic("CREATE_FUNCTION") is False


# ── ADD_PARAM ─────────────────────────────────────────────────────────────────

class TestAddParam:
    def test_appends_param(self) -> None:
        result = _exec.apply(_BASE, _op("ADD_PARAM", param_name="y", param_type="str"))
        tree = ast.parse(result)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        names = [a.arg for a in func.args.args]
        assert "y" in names

    def test_param_with_default(self) -> None:
        result = _exec.apply(_BASE, _op("ADD_PARAM", param_name="z", default_value=42))
        assert "z" in result

    def test_insert_at_position(self) -> None:
        result = _exec.apply(_BASE, _op("ADD_PARAM", param_name="first", position=0))
        tree = ast.parse(result)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        assert func.args.args[0].arg == "first"


# ── REMOVE_PARAM ──────────────────────────────────────────────────────────────

class TestRemoveParam:
    def test_removes_existing_param(self) -> None:
        result = _exec.apply(_BASE, _op("REMOVE_PARAM", param_name="x"))
        tree = ast.parse(result)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        assert not any(a.arg == "x" for a in func.args.args)

    def test_no_error_on_missing_param(self) -> None:
        result = _exec.apply(_BASE, _op("REMOVE_PARAM", param_name="nonexistent"))
        assert "foo" in result


# ── ADD_RETURN_TYPE ───────────────────────────────────────────────────────────

class TestAddReturnType:
    def test_sets_return_annotation(self) -> None:
        code = "def bar(x: int):\n    return x\n"
        result = _exec.apply(code, _op("ADD_RETURN_TYPE", target="bar", type="str"))
        tree = ast.parse(result)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        assert func.returns is not None


# ── RENAME_SYMBOL ─────────────────────────────────────────────────────────────

class TestRenameSymbol:
    def test_renames_function(self) -> None:
        result = _exec.apply(_BASE, _op("RENAME_SYMBOL", new_name="bar"))
        assert "def bar" in result
        assert "def foo" not in result

    def test_renames_class(self) -> None:
        result = _exec.apply(_CLASS, _op("RENAME_SYMBOL", target="Bar", new_name="Baz"))
        assert "class Baz" in result


# ── ADD_IMPORT ────────────────────────────────────────────────────────────────

class TestAddImport:
    def test_adds_import(self) -> None:
        result = _exec.apply(_BASE, _op("ADD_IMPORT", target="foo", module="pathlib", symbol="Path"))
        assert "from pathlib import Path" in result

    def test_no_duplicate(self) -> None:
        code = "from pathlib import Path\ndef foo(): pass\n"
        result = _exec.apply(code, _op("ADD_IMPORT", target="foo", module="pathlib", symbol="Path"))
        assert result.count("from pathlib import Path") == 1


# ── DELETE_NODE ───────────────────────────────────────────────────────────────

class TestDeleteNode:
    def test_deletes_function(self) -> None:
        code = "def foo(): pass\ndef bar(): pass\n"
        result = _exec.apply(code, _op("DELETE_NODE"))
        assert "def foo" not in result
        assert "def bar" in result


# ── ADD_DECORATOR ─────────────────────────────────────────────────────────────

class TestAddDecorator:
    def test_adds_decorator(self) -> None:
        result = _exec.apply(_BASE, _op("ADD_DECORATOR", decorator="staticmethod"))
        assert "@staticmethod" in result


# ── ADD_DOCSTRING ─────────────────────────────────────────────────────────────

class TestAddDocstring:
    def test_inserts_docstring(self) -> None:
        result = _exec.apply(_BASE, _op("ADD_DOCSTRING", docstring="Does something."))
        assert "Does something." in result

    def test_replaces_existing_docstring(self) -> None:
        code = 'def foo():\n    """Old."""\n    pass\n'
        result = _exec.apply(code, _op("ADD_DOCSTRING", docstring="New."))
        assert "New." in result
        assert "Old." not in result


# ── LLMRequiredError ──────────────────────────────────────────────────────────

class TestLLMRequired:
    def test_modify_body_raises(self) -> None:
        with pytest.raises(LLMRequiredError):
            _exec.apply(_BASE, _op("MODIFY_BODY"))

    def test_create_function_raises(self) -> None:
        with pytest.raises(LLMRequiredError):
            _exec.apply(_BASE, _op("CREATE_FUNCTION"))


# ── beam_search ───────────────────────────────────────────────────────────────

class TestBeamSearch:
    def test_filters_invalid_syntax(self) -> None:
        bad = _candidate("def broken(:", rank=0)
        good = _candidate("def ok(): pass", rank=1)
        result = filter_and_rank([bad, good])
        assert all(c.code != "def broken(:" for c in result)

    def test_returns_at_least_one(self) -> None:
        c = _candidate("def f(): pass")
        assert len(filter_and_rank([c])) >= 1

    def test_top_k_respected(self) -> None:
        candidates = [_candidate(f"def f{i}(): pass", rank=i) for i in range(10)]
        result = filter_and_rank(candidates, top_k=3)
        assert len(result) <= 3

    def test_empty_input_returns_empty(self) -> None:
        assert filter_and_rank([]) == []
