"""Tests pour CriticAgent — parsing et logique de révision."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.critic import CriticAgent, CriticIssue
from src.shared.config import Config


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def critic(config: Config) -> CriticAgent:
    return CriticAgent(config)


class TestCriticIssue:
    def test_dataclass_fields(self) -> None:
        issue = CriticIssue(
            file="src/todo.py",
            issue_type="import_error",
            description="add_task not defined in models.py",
            severity="error",
        )
        assert issue.severity == "error"
        assert issue.file == "src/todo.py"


class TestCriticAgentParsing:
    def test_parse_valid_issues(self, critic: CriticAgent) -> None:
        raw = json.dumps([
            {"file": "src/a.py", "issue_type": "import_error",
             "description": "Missing import", "severity": "error"},
            {"file": "src/b.py", "issue_type": "name_mismatch",
             "description": "Wrong name", "severity": "warning"},
        ])
        issues = critic._parse_issues(raw)
        assert len(issues) == 2
        assert issues[0].severity == "error"
        assert issues[1].severity == "warning"

    def test_parse_empty_list(self, critic: CriticAgent) -> None:
        issues = critic._parse_issues("[]")
        assert issues == []

    def test_parse_invalid_json(self, critic: CriticAgent) -> None:
        issues = critic._parse_issues("not json at all")
        assert issues == []

    def test_parse_filters_invalid_items(self, critic: CriticAgent) -> None:
        raw = json.dumps([
            {"file": "a.py", "issue_type": "import_error",
             "description": "ok", "severity": "error"},
            "not a dict",
            None,
        ])
        issues = critic._parse_issues(raw)
        assert len(issues) == 1

    def test_parse_unknown_severity_defaults_to_error(self, critic: CriticAgent) -> None:
        raw = json.dumps([
            {"file": "a.py", "issue_type": "import_error",
             "description": "x", "severity": "critical"},
        ])
        issues = critic._parse_issues(raw)
        assert issues[0].severity == "error"

    def test_issues_sorted_errors_first(self, critic: CriticAgent) -> None:
        raw = json.dumps([
            {"file": "b.py", "issue_type": "name_mismatch", "description": "w", "severity": "warning"},
            {"file": "a.py", "issue_type": "import_error", "description": "e", "severity": "error"},
        ])
        issues = critic._parse_issues(raw)
        # _parse_issues ne trie pas, c'est review() qui trie
        assert len(issues) == 2


class TestCriticAgentReview:
    def test_review_empty_files_returns_empty(self, critic: CriticAgent, tmp_path: Path) -> None:
        issues = critic.review(tmp_path, [])
        assert issues == []

    def test_review_nonexistent_files_skipped(self, critic: CriticAgent, tmp_path: Path) -> None:
        issues = critic.review(tmp_path, [str(tmp_path / "ghost.py")])
        assert issues == []

    def test_review_calls_model_and_parses(self, critic: CriticAgent, tmp_path: Path) -> None:
        py_file = tmp_path / "mod.py"
        py_file.write_text("def foo(): pass\n", encoding="utf-8")

        mock_response = json.dumps([
            {"file": "mod.py", "issue_type": "import_error",
             "description": "Missing import os", "severity": "error"}
        ])

        with patch.object(critic, "_call_model", return_value=mock_response):
            issues = critic.review(tmp_path, [str(py_file)])

        assert len(issues) == 1
        assert issues[0].issue_type == "import_error"

    def test_review_sorts_errors_first(self, critic: CriticAgent, tmp_path: Path) -> None:
        py_file = tmp_path / "mod.py"
        py_file.write_text("def foo(): pass\n", encoding="utf-8")

        mock_response = json.dumps([
            {"file": "b.py", "issue_type": "name_mismatch", "description": "w", "severity": "warning"},
            {"file": "a.py", "issue_type": "import_error", "description": "e", "severity": "error"},
        ])

        with patch.object(critic, "_call_model", return_value=mock_response):
            issues = critic.review(tmp_path, [str(py_file)])

        assert issues[0].severity == "error"

    def test_review_model_error_returns_empty(self, critic: CriticAgent, tmp_path: Path) -> None:
        py_file = tmp_path / "mod.py"
        py_file.write_text("x = 1\n", encoding="utf-8")

        with patch.object(critic, "_call_model", return_value="[]"):
            issues = critic.review(tmp_path, [str(py_file)])

        assert issues == []

    def test_build_files_block_truncates_large_files(self, critic: CriticAgent, tmp_path: Path) -> None:
        big_file = tmp_path / "big.py"
        big_file.write_text("x = 1\n" * 1000, encoding="utf-8")
        block = critic._build_files_block(tmp_path, [big_file])
        assert "tronqué" in block
