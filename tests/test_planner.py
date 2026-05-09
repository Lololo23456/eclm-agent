"""Tests pour src/planner/."""
from __future__ import annotations

import pytest

from src.orchestrator.context import ASTContext
from src.planner.model import ASTPlanner
from src.shared.config import Config
from src.shared.types import ASTOperationPlan, IntentJSON


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def planner(config: Config) -> ASTPlanner:
    return ASTPlanner(config)


@pytest.fixture
def empty_context() -> ASTContext:
    return ASTContext(chunks=[], target_file=None)


def _intent(
    action: str,
    target_type: str = "function",
    target_name: str = "my_func",
    description: str = "do something",
    confidence: float = 0.9,
    constraints: tuple[str, ...] = (),
) -> IntentJSON:
    return IntentJSON(
        action=action,
        target_type=target_type,
        target_name=target_name,
        description=description,
        confidence=confidence,
        constraints=constraints,
    )


class TestASTPlanner:
    def test_rename_produces_rename_symbol(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("RENAME", constraints=("my_func → my_function",))
        plan = planner.plan(intent, empty_context)
        assert isinstance(plan, ASTOperationPlan)
        assert plan.operations[0].op_type == "RENAME_SYMBOL"

    def test_delete_produces_delete_node(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("DELETE")
        plan = planner.plan(intent, empty_context)
        assert plan.operations[0].op_type == "DELETE_NODE"

    def test_modify_produces_modify_body(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("MODIFY")
        plan = planner.plan(intent, empty_context)
        assert plan.operations[0].op_type == "MODIFY_BODY"

    def test_fix_produces_modify_body(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("FIX")
        plan = planner.plan(intent, empty_context)
        assert plan.operations[0].op_type == "MODIFY_BODY"

    def test_create_function_produces_create_function(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("CREATE", target_type="function")
        plan = planner.plan(intent, empty_context)
        assert plan.operations[0].op_type == "CREATE_FUNCTION"

    def test_create_class_produces_create_class(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("CREATE", target_type="class")
        plan = planner.plan(intent, empty_context)
        assert plan.operations[0].op_type == "CREATE_CLASS"

    def test_plan_always_returns_at_least_one_op(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        for action in ["MODIFY", "FIX", "REFACTOR", "OPTIMIZE", "CREATE", "TEST", "RENAME", "DELETE"]:
            intent = _intent(action)
            plan = planner.plan(intent, empty_context)
            assert len(plan.operations) >= 1, f"Plan vide pour action={action}"

    def test_plan_intent_is_preserved(self, planner: ASTPlanner, empty_context: ASTContext) -> None:
        intent = _intent("MODIFY")
        plan = planner.plan(intent, empty_context)
        assert plan.intent is intent
