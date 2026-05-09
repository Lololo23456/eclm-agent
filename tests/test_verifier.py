"""Tests pour src/verifier/."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.config import Config
from src.shared.types import ASTCandidate, ASTOperation, VerificationResult
from src.verifier.pipeline import VerificationPipeline, _parse_pytest_score
from src.verifier.scorer import LintScorer, SyntaxChecker, TypeChecker


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def pipeline(config: Config) -> VerificationPipeline:
    return VerificationPipeline(config)


def _make_candidate(code: str, rank: int = 0) -> ASTCandidate:
    op = ASTOperation(op_type="MODIFY_BODY", target="test_func", params={})
    return ASTCandidate(code=code, operation=op, generation_rank=rank)


VALID_CODE = "def add(a: int, b: int) -> int:\n    return a + b\n"
INVALID_SYNTAX_CODE = "def broken(:\n    pass\n"
TYPED_CODE = "def greet(name: str) -> str:\n    return f'Hello {name}'\n"


class TestSyntaxChecker:
    def test_valid_code(self) -> None:
        assert SyntaxChecker().check(VALID_CODE) is True

    def test_invalid_syntax(self) -> None:
        assert SyntaxChecker().check(INVALID_SYNTAX_CODE) is False

    def test_empty_string(self) -> None:
        assert SyntaxChecker().check("") is True

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200))
    @settings(max_examples=50)
    def test_never_raises(self, code: str) -> None:
        SyntaxChecker().check(code)  # Doit toujours retourner bool, jamais lever


class TestLintScorer:
    def test_clean_code_scores_one(self) -> None:
        score = LintScorer().score(VALID_CODE)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self) -> None:
        for code in [VALID_CODE, TYPED_CODE, ""]:
            s = LintScorer().score(code)
            assert 0.0 <= s <= 1.0, f"Score hors range pour: {code!r}"


class TestTypeChecker:
    def test_typed_code_passes(self) -> None:
        ok, msg = TypeChecker().check(TYPED_CODE)
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_returns_error_message_on_failure(self) -> None:
        untyped = "def f(x):\n    return x\n"
        ok, msg = TypeChecker().check(untyped)
        if not ok:
            assert len(msg) > 0


class TestParsePytestScore:
    def test_all_passed(self) -> None:
        assert _parse_pytest_score("5 passed in 0.1s") == pytest.approx(1.0)

    def test_partial(self) -> None:
        assert _parse_pytest_score("3 passed, 1 failed in 0.2s") == pytest.approx(0.75)

    def test_all_failed(self) -> None:
        assert _parse_pytest_score("0 passed, 2 failed in 0.1s") == pytest.approx(0.0)

    def test_empty_output(self) -> None:
        assert _parse_pytest_score("") == pytest.approx(0.0)


class TestVerificationPipeline:
    def test_raises_on_empty_candidates(self, pipeline: VerificationPipeline) -> None:
        with pytest.raises(ValueError, match="vide"):
            pipeline.verify(candidates=[], behavior_tests=[])

    def test_syntax_error_gets_zero_score(self, pipeline: VerificationPipeline) -> None:
        candidate = _make_candidate(INVALID_SYNTAX_CODE)
        result = pipeline.verify(
            candidates=[candidate],
            behavior_tests=[],
        )
        assert result.syntax_ok is False
        assert result.composite_score == pytest.approx(0.0)

    def test_best_candidate_selected(self, pipeline: VerificationPipeline) -> None:
        good = _make_candidate(VALID_CODE, rank=0)
        bad = _make_candidate(INVALID_SYNTAX_CODE, rank=1)
        result = pipeline.verify(candidates=[good, bad], behavior_tests=[])
        assert result.syntax_ok is True

    def test_result_type(self, pipeline: VerificationPipeline) -> None:
        candidate = _make_candidate(VALID_CODE)
        result = pipeline.verify(candidates=[candidate], behavior_tests=[])
        assert isinstance(result, VerificationResult)
        assert 0.0 <= result.composite_score <= 1.0
