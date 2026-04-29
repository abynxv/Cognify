"""Tests for document upload, list, get, delete endpoints."""
from __future__ import annotations

import io

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_upload_txt_document(client: AsyncClient) -> None:
    content = b"This is a test document for Cognify RAG."
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert data["filename"] == "test.txt"
    assert "document_id" in data


async def test_upload_unsupported_extension(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("bad.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert response.status_code == 415


async def test_list_documents(client: AsyncClient) -> None:
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


async def test_get_nonexistent_document(client: AsyncClient) -> None:
    import uuid
    response = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert response.status_code == 404
