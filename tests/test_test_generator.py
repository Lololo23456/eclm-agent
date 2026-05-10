"""Tests pour src/verifier/test_generator/model.py."""
from __future__ import annotations

import pytest

from src.shared.config import Config
from src.shared.types import IntentJSON
from src.verifier.test_generator.model import (
    TestGenerator,
    TestGeneratorOutput,
    parse_test_functions,
)


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def gen(config: Config) -> TestGenerator:
    return TestGenerator(config)


# ── parse_test_functions ──────────────────────────────────────────────────────

class TestParseTestFunctions:
    def test_extracts_standalone_tests(self) -> None:
        raw = """
def test_add():
    from solution import add
    assert add(1, 2) == 3

def test_add_negative():
    from solution import add
    assert add(-1, 1) == 0
"""
        tests = parse_test_functions(raw)
        assert len(tests) == 2
        assert all("def test_" in t for t in tests)

    def test_rejects_self_methods(self) -> None:
        raw = """
def test_ok():
    assert True

def test_with_self(self):
    assert True
"""
        tests = parse_test_functions(raw)
        assert len(tests) == 1
        assert "test_ok" in tests[0]

    def test_extracts_from_markdown_block(self) -> None:
        raw = """```python
def test_foo():
    assert 1 + 1 == 2
```"""
        tests = parse_test_functions(raw)
        assert len(tests) == 1

    def test_handles_syntax_error_with_regex_fallback(self) -> None:
        raw = "def test_broken():\n    assert True ==\n\ndef test_ok():\n    assert True"
        # Should not raise — fallback to regex
        tests = parse_test_functions(raw)
        # At least the valid one (regex may recover partial)
        assert isinstance(tests, list)

    def test_returns_empty_for_no_tests(self) -> None:
        raw = "def helper():\n    return 42"
        tests = parse_test_functions(raw)
        assert tests == []

    def test_handles_empty_string(self) -> None:
        assert parse_test_functions("") == []


# ── TestGeneratorOutput ───────────────────────────────────────────────────────

class TestTestGeneratorOutput:
    def test_bool_true_with_tests(self) -> None:
        out = TestGeneratorOutput(tests=["def test_x(): pass"], confidence=0.5)
        assert bool(out)

    def test_bool_false_without_tests(self) -> None:
        out = TestGeneratorOutput(tests=[], confidence=0.0)
        assert not bool(out)


# ── TestGenerator interface ───────────────────────────────────────────────────

class TestTestGeneratorInterface:
    def test_load_is_noop(self, gen: TestGenerator) -> None:
        gen.load()  # Should not raise

    def test_generate_from_code_empty_returns_empty(self, gen: TestGenerator) -> None:
        out = gen.generate_from_code("")
        assert out.tests == []
        assert out.confidence == 0.0

    def test_generate_is_alias_for_from_code(self, gen: TestGenerator) -> None:
        # Both should return TestGeneratorOutput (even if empty due to no Ollama)
        out = gen.generate("")
        assert isinstance(out, TestGeneratorOutput)

    def test_generate_from_intent_returns_output(self, gen: TestGenerator) -> None:
        intent = IntentJSON(
            action="CREATE",
            target_type="function",
            target_name="multiply",
            description="Multiply two numbers and return the result",
            confidence=0.95,
        )
        # Ollama may not be available in test env — just check type
        out = gen.generate_from_intent(intent)
        assert isinstance(out, TestGeneratorOutput)
        assert isinstance(out.tests, list)
        assert 0.0 <= out.confidence <= 1.0

    def test_generate_from_code_returns_output(self, gen: TestGenerator) -> None:
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        out = gen.generate_from_code(code)
        assert isinstance(out, TestGeneratorOutput)
        assert isinstance(out.tests, list)

    def test_isolation_from_eclm_candidates(self, gen: TestGenerator) -> None:
        """Vérifie que generate_from_intent() ne prend pas de candidats ECLM en paramètre."""
        import inspect
        sig = inspect.signature(gen.generate_from_intent)
        # Only 'intent' parameter — no way to accidentally pass candidates
        params = list(sig.parameters.keys())
        assert params == ["intent"]
