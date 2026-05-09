"""Tests pour src/orchestrator/project.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.agent import AgentResponse
from src.orchestrator.project import ProjectAgent, ProjectSession, TaskRecord
from src.shared.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data", models_dir=tmp_path / "models")


@pytest.fixture
def project_agent(config: Config, tmp_path: Path) -> ProjectAgent:
    return ProjectAgent(config, tmp_path)


# ── TaskRecord ────────────────────────────────────────────────────────────────

class TestTaskRecord:
    def test_done_property_true_when_status_done(self) -> None:
        task = TaskRecord(
            index=0, action="CREATE", target_type="function",
            target_name="foo", target_file="foo.py", description="test",
        )
        task.status = "done"
        assert task.done is True

    def test_done_property_false_when_pending(self) -> None:
        task = TaskRecord(
            index=0, action="CREATE", target_type="function",
            target_name="foo", target_file="foo.py", description="test",
        )
        assert task.done is False

    def test_label_format(self) -> None:
        task = TaskRecord(
            index=0, action="CREATE", target_type="function",
            target_name="foo", target_file="foo.py", description="test",
        )
        assert "CREATE" in task.label
        assert "foo.py" in task.label


# ── ProjectSession ────────────────────────────────────────────────────────────

class TestProjectSession:
    def _make_session(self, n: int = 3) -> ProjectSession:
        tasks = [
            TaskRecord(
                index=i, action="CREATE", target_type="function",
                target_name=f"fn_{i}", target_file="main.py",
                description=f"task {i}",
                depends_on=[i - 1] if i > 0 else [],
            )
            for i in range(n)
        ]
        return ProjectSession(
            id="test-id", brief="test brief",
            created_at="2026-01-01T00:00:00Z", tasks=tasks,
        )

    def test_done_count(self) -> None:
        session = self._make_session(3)
        session.tasks[0].status = "done"
        session.tasks[1].status = "done"
        assert session.done_count == 2

    def test_total(self) -> None:
        session = self._make_session(3)
        assert session.total == 3

    def test_next_pending_respects_dependencies(self) -> None:
        session = self._make_session(3)
        # task 0 is pending, no deps → should be next
        next_task = session.next_pending
        assert next_task is not None
        assert next_task.index == 0

    def test_next_pending_skips_blocked_tasks(self) -> None:
        session = self._make_session(3)
        # task 1 depends on task 0 (not done yet) → only task 0 is eligible
        next_task = session.next_pending
        assert next_task is not None and next_task.index == 0

    def test_next_pending_none_when_all_done(self) -> None:
        session = self._make_session(2)
        for t in session.tasks:
            t.status = "done"
        assert session.next_pending is None

    def test_all_files_created_deduped(self) -> None:
        session = self._make_session(3)
        session.tasks[0].files_created = ["a.py", "b.py"]
        session.tasks[1].files_created = ["b.py", "c.py"]
        files = session.all_files_created
        assert files.count("b.py") == 1
        assert set(files) == {"a.py", "b.py", "c.py"}


# ── ProjectAgent ──────────────────────────────────────────────────────────────

class TestProjectAgent:
    def test_fallback_plan_creates_single_task(
        self, project_agent: ProjectAgent
    ) -> None:
        tasks = project_agent._fallback_plan("créer une API REST")
        assert len(tasks) == 1
        assert tasks[0].action == "CREATE"
        assert tasks[0].index == 0

    def test_parse_plan_valid_json(self, project_agent: ProjectAgent) -> None:
        raw = json.dumps({
            "tasks": [
                {
                    "index": 0, "action": "CREATE", "target_type": "class",
                    "target_name": "User", "target_file": "models.py",
                    "description": "User model", "depends_on": [],
                }
            ],
            "estimated_files": ["models.py"],
        })
        tasks = project_agent._parse_plan(raw)
        assert tasks is not None
        assert len(tasks) == 1
        assert tasks[0].target_name == "User"

    def test_parse_plan_invalid_json_returns_none(
        self, project_agent: ProjectAgent
    ) -> None:
        result = project_agent._parse_plan("not json at all")
        assert result is None

    def test_save_and_load_roundtrip(
        self, project_agent: ProjectAgent
    ) -> None:
        tasks = [
            TaskRecord(
                index=0, action="CREATE", target_type="function",
                target_name="main", target_file="main.py", description="main fn",
            )
        ]
        session = ProjectSession(
            id="roundtrip-test", brief="brief",
            created_at="2026-01-01T00:00:00Z", tasks=tasks,
        )
        project_agent._save(session)

        loaded = project_agent.load("roundtrip-test")
        assert loaded.id == session.id
        assert loaded.brief == session.brief
        assert len(loaded.tasks) == 1
        assert loaded.tasks[0].target_name == "main"

    def test_list_sessions_returns_metadata(
        self, project_agent: ProjectAgent
    ) -> None:
        tasks = [
            TaskRecord(
                index=0, action="CREATE", target_type="function",
                target_name="foo", target_file="foo.py", description="foo",
                status="done",
            )
        ]
        session = ProjectSession(
            id="list-test", brief="a brief",
            created_at="2026-01-01T00:00:00Z", tasks=tasks,
        )
        project_agent._save(session)

        sessions = project_agent.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == "list-test"
        assert sessions[0]["done"] == 1

    def test_execute_marks_tasks_done_on_success(
        self, project_agent: ProjectAgent
    ) -> None:
        from src.shared.types import IntentJSON
        dummy_intent = IntentJSON(
            action="CREATE", target_type="function", target_name="foo",
            target_file="foo.py", description="foo", confidence=0.9,
        )
        mock_response = AgentResponse(
            success=True, message="ok", code="def foo(): pass",
            score=1.0, retries_used=0, intent=dummy_intent,
        )
        with patch.object(project_agent, "_run_task", return_value=mock_response):
            with patch.object(project_agent._agent._rag, "index_file"):
                tasks = [
                    TaskRecord(
                        index=0, action="CREATE", target_type="function",
                        target_name="foo", target_file="foo.py", description="foo",
                    )
                ]
                session = ProjectSession(
                    id="exec-test", brief="brief",
                    created_at="2026-01-01T00:00:00Z", tasks=tasks,
                )
                result = project_agent.execute(session)

        assert result.tasks[0].status == "done"
        assert result.tasks[0].verification_score == 1.0
