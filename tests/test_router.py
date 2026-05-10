"""Tests pour src/orchestrator/router.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestrator.router import ModelRouter, _FAST_OPS, _STRONG_OPS
from src.shared.config import Config


@pytest.fixture
def config_dual(tmp_path: Path) -> Config:
    """Config avec fast != strong (simule le setup 4090)."""
    cfg = Config(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    cfg.fast_model = "qwen2.5-coder:7b"
    cfg.strong_model = "qwen2.5-coder:32b"
    return cfg


@pytest.fixture
def config_single(tmp_path: Path) -> Config:
    """Config avec fast == strong (simule M3 Air sans 32B)."""
    cfg = Config(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    cfg.fast_model = "qwen2.5-coder:7b"
    cfg.strong_model = "qwen2.5-coder:7b"
    return cfg


class TestModelRouter:
    def test_fast_ops_use_fast_model(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        for op in _FAST_OPS:
            assert router.for_operation(op) == config_dual.fast_model

    def test_strong_ops_use_strong_model(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        for op in _STRONG_OPS:
            assert router.for_operation(op) == config_dual.strong_model

    def test_low_complexity_uses_fast(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        assert router.for_operation("CREATE_FUNCTION", complexity="low") == config_dual.fast_model

    def test_high_complexity_uses_strong(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        assert router.for_operation("MODIFY_BODY", complexity="high") == config_dual.strong_model

    def test_planning_always_strong(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        assert router.for_planning() == config_dual.strong_model

    def test_intent_always_fast(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        assert router.for_intent() == config_dual.fast_model

    def test_single_model_always_returns_fast(self, config_single: Config) -> None:
        router = ModelRouter(config_single)
        # Même les ops lourdes retournent fast quand fast==strong
        assert router.for_operation("CREATE_CLASS", complexity="high") == config_single.fast_model
        assert router.for_planning() == config_single.strong_model

    def test_medium_complexity_uses_fast(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        assert router.for_operation("UNKNOWN_OP", complexity="medium") == config_dual.fast_model

    def test_unknown_op_medium_uses_fast(self, config_dual: Config) -> None:
        router = ModelRouter(config_dual)
        result = router.for_operation("SOME_FUTURE_OP")
        assert result == config_dual.fast_model
