"""Shared pytest fixtures for the EvidenceEngine test suite."""

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
