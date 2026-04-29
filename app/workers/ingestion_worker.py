"""
Background ingestion worker.

FastAPI's BackgroundTasks is used for per-request background work.
For heavy production loads, replace the `run_ingestion` call with
a Celery/ARQ task push and process on a separate worker fleet.
"""
from __future__ import annotations

import uuid

from app.core.logging import get_logger
from app.db.database import AsyncSessionLocal
from app.db.vector_store import get_vector_store
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService

logger = get_logger(__name__)


async def run_ingestion(document_id: uuid.UUID, reindex: bool = False) -> None:
    """
    Entry point for background ingestion.
    Opens its own DB session — cannot share the request session
    because it lives past the request lifetime.
    """
    logger.info("background_ingestion_start", document_id=str(document_id))

    async with AsyncSessionLocal() as session:
        try:
            vector_store = get_vector_store()
            llm_service = LLMService()
            ingestion_service = IngestionService(session, vector_store, llm_service)
            chunk_count = await ingestion_service.ingest(document_id, reindex=reindex)
            await session.commit()
            logger.info(
                "background_ingestion_done",
                document_id=str(document_id),
                chunks=chunk_count,
            )
        except Exception as exc:
            await session.rollback()
            logger.error(
                "background_ingestion_error",
                document_id=str(document_id),
                error=str(exc),
                exc_info=True,
            )
