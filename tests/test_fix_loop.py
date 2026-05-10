"""Tests pour la boucle de correction automatique dans ProjectAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.project import ProjectAgent, ProjectSession, TaskRecord
from src.shared.config import Config


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


def _make_session(tmp_path: Path, tasks: list[TaskRecord]) -> ProjectSession:
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    return ProjectSession(
        id="test-session",
        brief="test",
        created_at="2026-01-01T00:00:00Z",
        tasks=tasks,
        output_dir=str(out_dir),
    )


def _make_task(index: int, files: list[str] | None = None) -> TaskRecord:
    return TaskRecord(
        index=index,
        action="CREATE",
        target_type="function",
        target_name=f"task_{index}",
        target_file=f"src/mod{index}.py",
        description="test task",
        files_created=files or [],
        status="done",
    )


class TestCollectFailingFiles:
    def test_parses_pytest_failures(self, config: Config, tmp_path: Path) -> None:
        agent = ProjectAgent.__new__(ProjectAgent)
        agent.config = config

        session = _make_session(tmp_path, [_make_task(0)])
        test_results = {
            "stdout": "FAILED tests/test_foo.py::test_bar - AssertionError\n",
            "passed": 0, "failed": 1, "total": 1, "score": 0.0,
        }
        failing = agent._collect_failing_files(
            session, test_results, [], Path(tmp_path / "project")
        )
        assert "tests/test_foo.py" in failing

    def test_adds_critic_errors(self, config: Config, tmp_path: Path) -> None:
        from src.orchestrator.critic import CriticIssue
        agent = ProjectAgent.__new__(ProjectAgent)
        agent.config = config

        session = _make_session(tmp_path, [_make_task(0)])
        issue = CriticIssue(
            file="src/mod.py",
            issue_type="import_error",
            description="Missing import",
            severity="error",
        )
        failing = agent._collect_failing_files(
            session, {"stdout": "", "score": 1.0, "total": 0}, [issue],
            Path(tmp_path / "project"),
        )
        assert "src/mod.py" in failing
        assert "Missing import" in failing["src/mod.py"]

    def test_ignores_critic_warnings(self, config: Config, tmp_path: Path) -> None:
        from src.orchestrator.critic import CriticIssue
        agent = ProjectAgent.__new__(ProjectAgent)
        agent.config = config

        session = _make_session(tmp_path, [_make_task(0)])
        issue = CriticIssue(
            file="src/mod.py",
            issue_type="style",
            description="No docstring",
            severity="warning",
        )
        failing = agent._collect_failing_files(
            session, {"stdout": "", "score": 1.0, "total": 0}, [issue],
            Path(tmp_path / "project"),
        )
        assert failing == {}


class TestContextPruning:
    def test_max_context_chars_respected(self, tmp_path: Path) -> None:
        from src.orchestrator.dependency_graph import DependencyGraph

        dg = DependencyGraph()

        # Créer des fichiers avec beaucoup de contenu
        for i in range(10):
            f = tmp_path / f"mod{i}.py"
            f.write_text(f"def func_{i}(x: int) -> int:\n    return x + {i}\n" * 20, encoding="utf-8")

        tasks = []
        for i in range(10):
            t = TaskRecord(
                index=i, action="CREATE", target_type="function",
                target_name=f"f{i}", target_file=f"mod{i}.py", description="x",
                files_created=[str(tmp_path / f"mod{i}.py")],
                status="done",
            )
            tasks.append(t)

        final_task = TaskRecord(
            index=10, action="CREATE", target_type="function",
            target_name="final", target_file="final.py", description="uses all mods",
            depends_on=list(range(10)),
        )
        tasks.append(final_task)

        ctx = dg.get_context_for_task(final_task, tasks, tmp_path)
        assert len(ctx) <= DependencyGraph.MAX_CONTEXT_CHARS + 200  # léger dépassement toléré sur le dernier fichier

    def test_prioritizes_files_mentioned_in_description(self, tmp_path: Path) -> None:
        from src.orchestrator.dependency_graph import DependencyGraph

        dg = DependencyGraph()

        priority_file = tmp_path / "models.py"
        priority_file.write_text("class User:\n    def __init__(self): ...\n", encoding="utf-8")
        other_file = tmp_path / "utils.py"
        other_file.write_text("def helper(): pass\n", encoding="utf-8")

        t0 = TaskRecord(index=0, action="CREATE", target_type="class",
                        target_name="User", target_file="models.py",
                        description="x", files_created=[str(priority_file)], status="done")
        t1 = TaskRecord(index=1, action="CREATE", target_type="function",
                        target_name="helper", target_file="utils.py",
                        description="x", files_created=[str(other_file)], status="done")
        cur = TaskRecord(index=2, action="CREATE", target_type="function",
                         target_name="create_user", target_file="service.py",
                         description="creates a user using models.py",
                         depends_on=[0, 1])

        ctx = dg.get_context_for_task(cur, [t0, t1, cur], tmp_path)
        # models.py doit apparaître avant utils.py dans le contexte
        pos_models = ctx.find("models.py")
        pos_utils = ctx.find("utils.py")
        assert pos_models < pos_utils
