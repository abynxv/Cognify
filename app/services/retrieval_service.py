from __future__ import annotations

import uuid

from app.core.config import settings
from app.core.exceptions import InsufficientContextError
from app.core.logging import get_logger
from app.db.vector_store import VectorStore
from app.schemas.query import SourceReference
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class RetrievalService:
    def __init__(self, vector_store: VectorStore, llm_service: LLMService) -> None:
        self._vector_store = vector_store
        self._llm_service = llm_service

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> tuple[list[SourceReference], str]:
        """
        Embed query, search Qdrant, return (sources, context_string).
        Raises InsufficientContextError if nothing meets the threshold.
        """
        k = top_k or settings.TOP_K
        doc_id_strs = [str(d) for d in document_ids] if document_ids else None

        # Embed in a single call (list of 1)
        embeddings = await self._llm_service.embed([query])
        query_vector = embeddings[0]

        results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=k,
            document_ids=doc_id_strs,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )

        if not results:
            raise InsufficientContextError(
                "No relevant content found for this query. "
                "Try rephrasing or uploading related documents."
            )

        sources: list[SourceReference] = []
        context_parts: list[str] = []

        for i, hit in enumerate(results, start=1):
            payload = hit.payload or {}
            source = SourceReference(
                chunk_id=str(hit.id),
                document_id=payload.get("document_id", ""),
                document_name=payload.get("document_name", "unknown"),
                chunk_index=payload.get("chunk_index", 0),
                page_number=payload.get("page_number"),
                section=payload.get("section"),
                content_preview=payload.get("content", "")[:300],
                relevance_score=round(hit.score, 4),
            )
            sources.append(source)

            # Format context block with citation marker
            context_parts.append(
                f"[Source {i}: {source.document_name}"
                + (f", page {source.page_number}" if source.page_number else "")
                + f"]\n{payload.get('content', '')}"
            )

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "retrieval_complete",
            query_preview=query[:80],
            hits=len(results),
            top_score=results[0].score if results else 0,
        )
        return sources, context
