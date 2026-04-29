from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Async Qdrant wrapper — one instance shared via DI."""

    def __init__(self) -> None:
        if settings.QDRANT_URL:
            self._client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            self._client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
        self._collection = settings.QDRANT_COLLECTION

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist. Called at app startup."""
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qdrant_models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=20_000,  # build HNSW index after 20k vectors
                ),
            )
            # Index on document_id for fast per-document filtering
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="document_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("qdrant_collection_created", collection=self._collection)

    async def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        points = [
            qdrant_models.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload,
            )
            for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads)
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[qdrant_models.ScoredPoint]:
        query_filter = None
        if document_ids:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchAny(any=document_ids),
                    )
                ]
            )

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return results

    async def delete_by_document(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("qdrant_vectors_deleted", document_id=document_id)

    async def count(self, document_id: str | None = None) -> int:
        count_filter = None
        if document_id:
            count_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id),
                    )
                ]
            )
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=count_filter,
            exact=True,
        )
        return result.count

    async def close(self) -> None:
        await self._client.close()


# Singleton — reused across requests
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
