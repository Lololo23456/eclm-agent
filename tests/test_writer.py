"""Tests pour src/orchestrator/writer.py."""
from __future__ import annotations

import pytest
from pathlib import Path

from src.orchestrator.writer import FileWriter, _replace_node, _to_snake
from src.shared.types import IntentJSON


def _intent(
    action: str = "CREATE",
    target_name: str = "my_func",
    target_type: str = "function",
    target_file: str | None = None,
) -> IntentJSON:
    return IntentJSON(
        action=action,
        target_type=target_type,
        target_name=target_name,
        description="test",
        confidence=0.9,
        target_file=target_file,
    )


_SIMPLE_FUNC = "def my_func(x: int) -> int:\n    return x\n"
_OTHER_FUNC = "def other(y: int) -> int:\n    return y\n"


class TestReplaceNode:
    def test_replaces_existing_function(self) -> None:
        source = "def foo(x: int) -> int:\n    return x\n\ndef bar(): pass\n"
        new_code = "def foo(x: int) -> int:\n    return x * 2\n"
        result = _replace_node(source, "foo", new_code)
        assert result is not None
        assert "return x * 2" in result
        assert "def bar" in result

    def test_returns_none_when_not_found(self) -> None:
        assert _replace_node("def foo(): pass\n", "nonexistent", "def nonexistent(): pass") is None

    def test_handles_decorated_function(self) -> None:
        source = "@staticmethod\ndef foo(): pass\n"
        result = _replace_node(source, "foo", "def foo(): return 1")
        assert result is not None
        assert "@staticmethod" not in result  # decorator replaced with new code


class TestToSnake:
    def test_pascal_case(self) -> None:
        assert _to_snake("MyFunction") == "my_function"

    def test_already_snake(self) -> None:
        assert _to_snake("my_function") == "my_function"

    def test_camel_case(self) -> None:
        assert _to_snake("myFunction") == "my_function"


class TestFileWriter:
    def test_create_new_file(self, tmp_path: Path) -> None:
        writer = FileWriter()
        intent = _intent(action="CREATE", target_file="output.py")
        result = writer.write(_SIMPLE_FUNC, intent, tmp_path)
        assert result is not None
        assert result.exists()
        assert "def my_func" in result.read_text()

    def test_create_appends_to_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "output.py"
        target.write_text(_OTHER_FUNC)
        writer = FileWriter()
        intent = _intent(action="CREATE", target_file="output.py")
        writer.write(_SIMPLE_FUNC, intent, tmp_path)
        content = target.read_text()
        assert "def other" in content
        assert "def my_func" in content

    def test_create_skips_duplicate_symbol(self, tmp_path: Path) -> None:
        target = tmp_path / "output.py"
        target.write_text(_SIMPLE_FUNC)
        writer = FileWriter()
        intent = _intent(action="CREATE", target_file="output.py")
        writer.write(_SIMPLE_FUNC, intent, tmp_path)
        # Pas de doublon
        assert target.read_text().count("def my_func") == 1

    def test_modify_replaces_function(self, tmp_path: Path) -> None:
        target = tmp_path / "funcs.py"
        target.write_text("def my_func(x: int) -> int:\n    return x\n")
        writer = FileWriter()
        intent = _intent(action="MODIFY", target_file="funcs.py")
        new_code = "def my_func(x: int) -> int:\n    return x * 2\n"
        writer.write(new_code, intent, tmp_path)
        assert "return x * 2" in target.read_text()

    def test_infers_filename_from_target_name(self, tmp_path: Path) -> None:
        writer = FileWriter()
        intent = _intent(action="CREATE", target_name="calculate_discount", target_file=None)
        result = writer.write(_SIMPLE_FUNC, intent, tmp_path)
        assert result is not None
        assert result.name == "calculate_discount.py"
