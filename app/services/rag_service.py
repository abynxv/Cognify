from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import QueryHistory
from app.db.vector_store import VectorStore
from app.schemas.query import QueryRequest, QueryResponse, SourceReference
from app.services.cache_service import CacheService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)

# System prompt — kept tight to prevent hallucination outside context
_SYSTEM_PROMPT = """You are Cognify, an expert document assistant.
Answer the user's question using ONLY the information from the provided context.
Rules:
- If the answer is not in the context, say: "I don't have enough information to answer this question based on the provided documents."
- Always cite the source (Source number) when referencing information.
- Be concise, accurate, and professional.
- Do not fabricate facts or add information not present in the context."""


def _build_messages(
    query: str, context: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]


class RAGService:
    def __init__(
        self,
        db: AsyncSession,
        vector_store: VectorStore,
        llm_service: LLMService,
        cache_service: CacheService,
    ) -> None:
        self._db = db
        self._retrieval = RetrievalService(vector_store, llm_service)
        self._llm = llm_service
        self._cache = cache_service

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Non-streaming RAG query — returns complete response."""
        doc_id_strs = [str(d) for d in request.document_ids] if request.document_ids else None

        # Check cache first
        cached = await self._cache.get_query(request.query, doc_id_strs, request.top_k)
        if cached:
            logger.info("query_cache_hit", query_preview=request.query[:60])
            return QueryResponse(**cached)

        start = time.perf_counter()
        sources, context = await self._retrieval.retrieve(
            request.query, request.top_k, request.document_ids
        )
        messages = _build_messages(request.query, context)
        answer, in_tok, out_tok = await self._llm.complete(messages)
        latency_ms = int((time.perf_counter() - start) * 1000)

        query_id = uuid.uuid4()
        response = QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=self._llm.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            was_cached=False,
            created_at=__import__("datetime").datetime.utcnow(),
        )

        await self._persist_history(response, doc_id_strs)
        await self._cache.set_query(
            request.query, doc_id_strs, request.top_k, response.model_dump(mode="json")
        )

        logger.info(
            "query_complete",
            query_id=str(query_id),
            latency_ms=latency_ms,
            sources=len(sources),
            in_tokens=in_tok,
            out_tokens=out_tok,
        )
        return response

    async def stream_query(
        self, request: QueryRequest
    ) -> AsyncGenerator[tuple[str, list[SourceReference] | None], None]:
        """
        Yields (token, None) for each streamed token, then
        (final_metadata_json, sources) as the last event.
        """
        doc_id_strs = [str(d) for d in request.document_ids] if request.document_ids else None

        start = time.perf_counter()
        sources, context = await self._retrieval.retrieve(
            request.query, request.top_k, request.document_ids
        )
        messages = _build_messages(request.query, context)

        query_id = uuid.uuid4()
        full_answer: list[str] = []

        async for token in self._llm.stream(messages):
            full_answer.append(token)
            yield token, None

        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = "".join(full_answer)

        await self._persist_history_raw(
            query_id=query_id,
            query=request.query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )

        logger.info(
            "stream_query_complete",
            query_id=str(query_id),
            latency_ms=latency_ms,
            sources=len(sources),
        )
        # Final sentinel: yield metadata so the SSE handler can send the done event
        yield str(query_id), sources

    async def _persist_history(
        self, response: QueryResponse, document_ids: list[str] | None
    ) -> None:
        try:
            record = QueryHistory(
                id=response.query_id,
                query=response.query,
                answer=response.answer,
                sources=[s.model_dump() for s in response.sources],
                document_filter=document_ids,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                model_used=response.model_used,
                was_cached=response.was_cached,
            )
            self._db.add(record)
            await self._db.flush()
        except Exception as exc:
            logger.warning("history_persist_failed", error=str(exc))

    async def _persist_history_raw(
        self,
        query_id: uuid.UUID,
        query: str,
        answer: str,
        sources: list[SourceReference],
        latency_ms: int,
    ) -> None:
        try:
            record = QueryHistory(
                id=query_id,
                query=query,
                answer=answer,
                sources=[s.model_dump() for s in sources],
                latency_ms=latency_ms,
                model_used=self._llm.model_name,
            )
            self._db.add(record)
            await self._db.flush()
        except Exception as exc:
            logger.warning("stream_history_persist_failed", error=str(exc))
