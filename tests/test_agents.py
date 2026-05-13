"""Tests pour les agents spécialisés src/agents/."""
from __future__ import annotations

import pytest

from src.agents.base import AgentResult, BaseAgent
from src.agents.code_writer import CodeWriterAgent, _strip_markdown
from src.agents.fixer import FixerAgent
from src.agents.integrator import IntegratorAgent, _static_check
from src.agents.spec_writer import SpecWriterAgent
from src.agents.test_writer import TestWriterAgent, parse_test_functions
from src.shared.config import Config


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


# ── parse_test_functions ──────────────────────────────────────────────────────

def test_parse_test_functions_extracts_standalone() -> None:
    raw = """
def test_add_nominal():
    assert add(1, 2) == 3

def test_add_zero():
    assert add(0, 0) == 0
"""
    tests = parse_test_functions(raw)
    assert len(tests) == 2
    assert all("test_add" in t for t in tests)


def test_parse_test_functions_rejects_self_methods() -> None:
    raw = """
class TestFoo:
    def test_method(self):
        assert True

def test_standalone():
    assert True
"""
    tests = parse_test_functions(raw)
    assert len(tests) == 1
    assert "test_standalone" in tests[0]


def test_parse_test_functions_strips_markdown() -> None:
    raw = """```python
def test_foo():
    assert 1 + 1 == 2
```"""
    tests = parse_test_functions(raw)
    assert len(tests) == 1


def test_parse_test_functions_invalid_syntax_returns_empty() -> None:
    tests = parse_test_functions("def test_broken(\n    assert True")
    assert tests == []


def test_parse_test_functions_empty_string() -> None:
    assert parse_test_functions("") == []


# ── _strip_markdown ───────────────────────────────────────────────────────────

def test_strip_markdown_removes_fence() -> None:
    code = "```python\ndef foo(): pass\n```"
    assert _strip_markdown(code) == "def foo(): pass"


def test_strip_markdown_noop_on_plain() -> None:
    code = "def foo(): pass"
    assert _strip_markdown(code) == "def foo(): pass"


# ── _static_check ─────────────────────────────────────────────────────────────

def test_static_check_valid_files() -> None:
    files = {"src/a.py": "x = 1", "src/b.py": "y = 2"}
    issues = _static_check(files)
    assert issues == []


def test_static_check_detects_syntax_error() -> None:
    files = {"src/bad.py": "def foo(\n    pass"}
    issues = _static_check(files)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].file == "src/bad.py"


# ── SpecWriterAgent ───────────────────────────────────────────────────────────

def test_spec_writer_returns_task_unchanged_if_complete(config: Config) -> None:
    agent = SpecWriterAgent(config)
    task = {
        "index": 0,
        "action": "CREATE",
        "target_type": "function",
        "target_name": "add",
        "target_file": "src/math.py",
        "spec": {
            "description": "Additionne a et b",
            "signature": "def add(a: int, b: int) -> int:",
            "imports": [],
            "constraints": [],
        }
    }
    result = agent.run(task)
    assert result.success
    assert result.output["spec"]["signature"] == "def add(a: int, b: int) -> int:"


def test_spec_writer_result_type(config: Config) -> None:
    agent = SpecWriterAgent(config)
    task = {"index": 0, "action": "CREATE", "target_name": "foo",
            "target_type": "function", "target_file": "src/x.py", "spec": {}}
    result = agent.run(task)
    assert isinstance(result, AgentResult)


# ── TestWriterAgent ───────────────────────────────────────────────────────────

def test_test_writer_uses_existing_tests_if_complete(config: Config) -> None:
    agent = TestWriterAgent(config)
    existing = [
        "def test_add_nominal():\n    assert add(1, 2) == 3",
        "def test_add_zero():\n    assert add(0, 0) == 0",
    ]
    task = {
        "spec": {"description": "Add", "signature": "def add(a, b): ...", "constraints": []},
        "tests": existing,
    }
    result = agent.run(task)
    assert result.success
    assert result.output == existing


def test_test_writer_result_is_list(config: Config) -> None:
    agent = TestWriterAgent(config)
    task = {"spec": {}, "tests": []}
    result = agent.run(task)
    assert isinstance(result.output, list)


# ── CodeWriterAgent ───────────────────────────────────────────────────────────

def test_code_writer_result_structure(config: Config) -> None:
    agent = CodeWriterAgent(config)
    task = {
        "target_name": "add",
        "target_type": "function",
        "target_file": "src/math.py",
        "spec": {
            "description": "Additionne a et b",
            "signature": "def add(a: int, b: int) -> int:",
            "imports": [],
            "constraints": [],
        }
    }
    result = agent.run(task, tests=[])
    assert isinstance(result, AgentResult)
    assert isinstance(result.output, list)


# ── FixerAgent ────────────────────────────────────────────────────────────────

def test_fixer_result_structure(config: Config) -> None:
    agent = FixerAgent(config)
    task = {"spec": {"signature": "def add(a, b):", "constraints": []}}
    result = agent.run(
        code="def add(a, b):\n    return a - b",
        error="AssertionError: assert add(1, 2) == 3",
        task=task,
        tests=["def test_add():\n    assert add(1, 2) == 3"],
    )
    assert isinstance(result, AgentResult)
    assert isinstance(result.output, str)


# ── IntegratorAgent ───────────────────────────────────────────────────────────

def test_integrator_no_issues_on_valid_project(config: Config) -> None:
    agent = IntegratorAgent(config)
    files = {
        "src/models.py": "x: int = 1\n",
        "src/api.py": "from src.models import x\n",
    }
    result = agent.run(files)
    assert isinstance(result, AgentResult)
    assert isinstance(result.output, list)


def test_integrator_detects_syntax_error(config: Config) -> None:
    agent = IntegratorAgent(config)
    files = {"src/bad.py": "def foo(\n    pass"}
    result = agent.run(files)
    errors = [i for i in result.output if i.severity == "error"]
    assert len(errors) >= 1


def test_integrator_single_file_returns_static_only(config: Config) -> None:
    agent = IntegratorAgent(config)
    result = agent.run({"src/only.py": "x = 1"})
    assert result.success
    assert result.output == []  # fichier valide → aucun problème
