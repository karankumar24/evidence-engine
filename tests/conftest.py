"""Shared pytest fixtures for the EvidenceEngine test suite."""

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest.fixture
def mock_nli_probs(monkeypatch):
    """Patch ``nli_probs_for_pair`` to return deterministic tuples from a queue.

    Returns a setter ``set_probs(values)`` where ``values`` is a list of
    ``(p_entail, p_neutral, p_contra)`` tuples or ``None``. Each call to
    ``nli_probs_for_pair`` pops one value from the front of the queue.
    Once the queue is empty, subsequent calls return ``None`` — the same
    signal the real function uses when the singleton failed to load.

    Plan 03 reuses this fixture in ``tests/test_explanation.py`` and the
    updated ``tests/test_classifier_backend.py``.
    """
    from evidenceengine.classification import nli_classifier

    queue: list = []

    def _fake(premise, hypothesis):
        if not queue:
            return None
        return queue.pop(0)

    monkeypatch.setattr(nli_classifier, "nli_probs_for_pair", _fake)

    def set_probs(values):
        queue.clear()
        queue.extend(values)

    return set_probs

# Use test database URL from environment or default to the dev database
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine",
)


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Return the test database URL."""
    return TEST_DATABASE_URL


@pytest_asyncio.fixture(scope="session")
async def test_engine(test_database_url: str):
    """Create a test engine (session-scoped for speed)."""
    engine = create_async_engine(test_database_url, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an AsyncSession for a single test, rolling back after each test."""
    async_session = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()
