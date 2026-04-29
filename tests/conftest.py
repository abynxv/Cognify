"""
Shared test fixtures.

Tests use an in-memory SQLite DB so no external services are needed.
Qdrant and Redis are mocked at the service level.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from unittest.mock import AsyncMock, MagicMock

from app.db.database import Base, get_db
from app.db.vector_store import VectorStore, get_vector_store
from app.main import create_app
from app.services.cache_service import CacheService, get_cache_service
from app.services.llm_service import LLMService
from app.api.dependencies import get_llm_service

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def engine():
    eng = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncSession:
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_vector_store() -> VectorStore:
    store = MagicMock(spec=VectorStore)
    store.search = AsyncMock(return_value=[])
    store.upsert_chunks = AsyncMock()
    store.delete_by_document = AsyncMock()
    store.ensure_collection = AsyncMock()
    store.count = AsyncMock(return_value=0)
    return store


@pytest.fixture
def mock_llm_service() -> LLMService:
    svc = MagicMock(spec=LLMService)
    svc.embed = AsyncMock(return_value=[[0.1] * 1536])
    svc.complete = AsyncMock(return_value=("Test answer.", 100, 50))
    svc.model_name = "gpt-4o-test"

    async def fake_stream(*args, **kwargs):
        for token in ["Hello", " world", "."]:
            yield token

    svc.stream = fake_stream
    return svc


@pytest.fixture
def mock_cache() -> CacheService:
    cache = MagicMock(spec=CacheService)
    cache.get_query = AsyncMock(return_value=None)
    cache.set_query = AsyncMock()
    cache.ping = AsyncMock(return_value=True)
    cache.close = AsyncMock()
    return cache


@pytest_asyncio.fixture
async def client(db_session, mock_vector_store, mock_llm_service, mock_cache) -> AsyncClient:
    app: FastAPI = create_app()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_service
    app.dependency_overrides[get_cache_service] = lambda: mock_cache

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
