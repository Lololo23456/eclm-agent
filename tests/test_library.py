"""Tests pour src/library/."""
from __future__ import annotations

import pytest

from src.library.primitive import Primitive
from src.library.store import PrimitiveStore
from src.shared.config import Config


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def store(config: Config) -> PrimitiveStore:
    return PrimitiveStore(config)


def _embedding(dim: int = 384, seed: int = 1) -> list[float]:
    """Vecteur aléatoire déterministe normalisé."""
    import hashlib
    digest = hashlib.sha256(seed.to_bytes(4, "big")).digest() * 12
    raw = list(digest[:dim])
    total = sum(raw) or 1.0
    return [v / total for v in raw]


def _primitive(domain: str = "parsing", score: float = 1.0) -> Primitive:
    return Primitive(
        code="def parse(s: str) -> list[str]:\n    return s.split()",
        tests=["def test_parse():\n    assert parse('a b') == ['a', 'b']"],
        domain=domain,
        description="Split string into tokens",
        score=score,
    )


class TestPrimitive:
    def test_valid_primitive(self) -> None:
        p = _primitive()
        assert p.id != ""
        assert p.verified_at != ""

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError, match="Score"):
            _primitive(score=1.5)

    def test_empty_code_raises(self) -> None:
        with pytest.raises(ValueError, match="code"):
            Primitive(code="  ", tests=[], domain="io", description="test")

    def test_roundtrip_metadata(self) -> None:
        p = _primitive()
        meta = p.to_metadata()
        restored = Primitive.from_metadata(p.id, p.description, meta)
        assert restored.code == p.code
        assert restored.domain == p.domain
        assert restored.tests == p.tests


class TestPrimitiveStore:
    def test_empty_store_count_zero(self, store: PrimitiveStore) -> None:
        assert store.count() == 0

    def test_add_and_retrieve(self, store: PrimitiveStore) -> None:
        p = _primitive()
        emb = _embedding(seed=1)
        store.add(p, emb)
        assert store.count() == 1
        got = store.get(p.id)
        assert got is not None
        assert got.code == p.code

    def test_search_returns_results(self, store: PrimitiveStore) -> None:
        p = _primitive()
        store.add(p, _embedding(seed=1))
        results = store.search(_embedding(seed=1), top_k=3)
        assert len(results) == 1
        assert results[0].id == p.id

    def test_search_empty_store(self, store: PrimitiveStore) -> None:
        assert store.search(_embedding(), top_k=3) == []

    def test_add_rejects_low_score(self, config: Config) -> None:
        store = PrimitiveStore(config)
        p = Primitive(
            code="def f(): pass",
            tests=[],
            domain="io",
            description="low score",
            score=0.5,
        )
        with pytest.raises(ValueError, match="Score"):
            store.add(p, _embedding())

    def test_delete(self, store: PrimitiveStore) -> None:
        p = _primitive()
        store.add(p, _embedding(seed=2))
        store.delete(p.id)
        assert store.get(p.id) is None

    def test_update_usage(self, store: PrimitiveStore) -> None:
        p = _primitive()
        store.add(p, _embedding(seed=3))
        store.update_usage(p.id)
        got = store.get(p.id)
        assert got is not None
        assert got.usage_count == 1

    def test_list_by_domain(self, store: PrimitiveStore) -> None:
        store.add(_primitive(domain="parsing"), _embedding(seed=4))
        store.add(_primitive(domain="http"), _embedding(seed=5))
        parsing = store.list_by_domain("parsing")
        assert len(parsing) == 1
        assert parsing[0].domain == "parsing"

    def test_upsert_replaces_existing(self, store: PrimitiveStore) -> None:
        p = _primitive()
        store.add(p, _embedding(seed=6))
        store.add(p, _embedding(seed=6))  # upsert — ne doit pas dupliquer
        assert store.count() == 1
