from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_rag_service
from app.core.config import settings
from app.core.logging import get_logger
from app.core.rate_limiter import limiter
from app.db.database import get_db
from app.db.models import QueryHistory
from app.schemas.query import (
    QueryHistoryResponse,
    QueryRequest,
    QueryResponse,
    SSEDoneEvent,
    SSEErrorEvent,
    SSETokenEvent,
)
from app.services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)


@router.post(
    "",
    summary="Query documents — streams via SSE or returns JSON",
    responses={
        200: {"description": "JSON response (stream=false)"},
        "2xx": {"description": "SSE stream (stream=true, Content-Type: text/event-stream)"},
    },
)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query(
    request: Request,
    body: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse | QueryResponse:
    if body.stream:
        return StreamingResponse(
            _sse_generator(body, rag_service),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",       # disable nginx buffering
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )
    # Non-streaming path — useful for programmatic callers and testing
    return await rag_service.query(body)


async def _sse_generator(
    request: QueryRequest, rag_service: RAGService
) -> AsyncGenerator[str, None]:
    """
    SSE wire format:
        event: token\ndata: {...}\n\n
        event: done\ndata: {...}\n\n
        event: error\ndata: {...}\n\n
    """

    def _event(event_type: str, payload: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    try:
        start = time.perf_counter()
        final_query_id: str | None = None
        final_sources = None

        async for token_or_id, maybe_sources in rag_service.stream_query(request):
            if maybe_sources is not None:
                # This is the final sentinel from stream_query
                final_query_id = token_or_id
                final_sources = maybe_sources
            else:
                yield _event("token", SSETokenEvent(content=token_or_id).model_dump())

        latency_ms = int((time.perf_counter() - start) * 1000)
        done_event = SSEDoneEvent(
            query_id=final_query_id or str(uuid.uuid4()),
            sources=[s.model_dump() for s in (final_sources or [])],
            latency_ms=latency_ms,
        )
        yield _event("done", done_event.model_dump())

    except Exception as exc:
        logger.error("sse_stream_error", error=str(exc), exc_info=True)
        yield _event("error", SSEErrorEvent(message=str(exc)).model_dump())


@router.get(
    "/history",
    response_model=list[QueryHistoryResponse],
    summary="Retrieve recent query history",
)
async def query_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[QueryHistoryResponse]:
    result = await db.execute(
        select(QueryHistory)
        .order_by(QueryHistory.created_at.desc())
        .limit(min(limit, 100))
    )
    records = result.scalars().all()
    return [QueryHistoryResponse.model_validate(r) for r in records]
