"""Tests for the query endpoint (streaming + non-streaming)."""
from __future__ import annotations

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio


async def test_query_no_stream(client: AsyncClient, mock_vector_store) -> None:
    from qdrant_client.http.models import ScoredPoint

    mock_vector_store.search = AsyncMock(
        return_value=[
            ScoredPoint(
                id="chunk-1",
                version=1,
                score=0.92,
                payload={
                    "document_id": "doc-1",
                    "document_name": "test.txt",
                    "chunk_index": 0,
                    "page_number": 1,
                    "content": "Cognify is a RAG backend.",
                },
            )
        ]
    )

    response = await client.post(
        "/api/v1/query",
        json={"query": "What is Cognify?", "stream": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_name"] == "test.txt"


async def test_query_insufficient_context(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/query",
        json={"query": "What is the meaning of life?", "stream": False},
    )
    # mock_vector_store.search returns [] by default → InsufficientContextError
    assert response.status_code == 404


async def test_query_history(client: AsyncClient) -> None:
    response = await client.get("/api/v1/query/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
