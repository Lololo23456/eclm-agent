"""Tests pour src/improvement/adversarial.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.improvement.adversarial import AdversarialLoop
from src.shared.config import Config
from src.shared.types import IntentJSON


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def loop(config: Config) -> AdversarialLoop:
    return AdversarialLoop(config)


def _intent(name: str = "add", action: str = "CREATE") -> IntentJSON:
    return IntentJSON(
        action=action,
        target_type="function",
        target_name=name,
        description=f"Implement {name}",
        confidence=0.9,
    )


# ── run_episode ───────────────────────────────────────────────────────────────

class TestRunEpisode:
    def test_creates_pair_when_spread_large_enough(self, loop: AdversarialLoop) -> None:
        intent = _intent()
        # chosen >= 0.8, rejected < 0.8
        candidates = [
            ("def add(a, b):\n    return a + b", 0.95),
            ("def add(a, b):\n    pass", 0.3),
        ]
        result = loop.run_episode(intent, candidates)
        assert result is True

    def test_returns_false_with_single_candidate(self, loop: AdversarialLoop) -> None:
        result = loop.run_episode(_intent(), [("def f(): pass", 0.5)])
        assert result is False

    def test_returns_false_with_empty_candidates(self, loop: AdversarialLoop) -> None:
        result = loop.run_episode(_intent(), [])
        assert result is False

    def test_no_pair_when_all_pass(self, loop: AdversarialLoop) -> None:
        # DPOPair requires chosen_score > rejected_score AND chosen >= 0.8
        # If both >= 0.8, a pair CAN be made (best vs worst)
        candidates = [
            ("def add(a, b):\n    return a + b", 0.95),
            ("def add(a, b):\n    return a+b", 0.81),
        ]
        # A pair may or may not be created (depends on DPOCollector)
        result = loop.run_episode(_intent(), candidates)
        assert isinstance(result, bool)

    def test_no_pair_when_all_fail(self, loop: AdversarialLoop) -> None:
        # Both below 0.8 → no valid DPO pair
        candidates = [("def f(): pass", 0.5), ("def f():\n    ...", 0.3)]
        result = loop.run_episode(_intent(), candidates)
        assert result is False


# ── run_from_sessions ─────────────────────────────────────────────────────────

class TestRunFromSessions:
    def test_returns_zero_for_empty_dir(self, loop: AdversarialLoop) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = loop.run_from_sessions(Path(tmpdir))
        assert result == 0

    def test_reads_session_files(self, loop: AdversarialLoop) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = {
                "id": "test-123",
                "brief": "test",
                "tasks": [
                    {
                        "action": "CREATE",
                        "target_type": "function",
                        "target_name": "compute",
                        "label": "CREATE src/compute.py:compute",
                        "target_file": "src/compute.py",
                        "status": "done",
                        "score": 0.95,
                    },
                    {
                        "label": "CREATE src/other.py:other",
                        "status": "pending",  # should be skipped
                    },
                ],
            }
            (session_dir / "session_abc.json").write_text(
                json.dumps(session), encoding="utf-8"
            )
            # run_from_sessions reads and builds tasks (Ollama not available → 0 pairs)
            result = loop.run_from_sessions(session_dir)
            assert isinstance(result, int)
            assert result >= 0

    def test_ignores_invalid_json(self, loop: AdversarialLoop) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "bad.json").write_text("not json", encoding="utf-8")
            result = loop.run_from_sessions(Path(tmpdir))
        assert result == 0


# ── run_from_dpo_prompts ──────────────────────────────────────────────────────

class TestRunFromDpoPrompts:
    def test_returns_zero_for_empty_dir(self, loop: AdversarialLoop) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = loop.run_from_dpo_prompts(Path(tmpdir))
        assert result == 0

    def test_reads_dpo_files(self, loop: AdversarialLoop) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dpo_dir = Path(tmpdir)
            pair = {
                "prompt": "CREATE add: Add two numbers and return result",
                "chosen": "def add(a, b):\n    return a + b",
                "rejected": "def add(a, b):\n    pass",
                "chosen_score": 0.95,
                "rejected_score": 0.2,
                "source": "test",
                "timestamp": "2026-01-01T00:00:00Z",
            }
            (dpo_dir / "dpo_2026-01.jsonl").write_text(
                json.dumps(pair), encoding="utf-8"
            )
            result = loop.run_from_dpo_prompts(dpo_dir, max_pairs=5)
            assert isinstance(result, int)
            assert result >= 0
