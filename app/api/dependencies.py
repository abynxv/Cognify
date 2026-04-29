from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.vector_store import VectorStore, get_vector_store
from app.services.cache_service import CacheService, get_cache_service
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


def get_llm_service() -> LLMService:
    return LLMService()


def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    return DocumentService(db)


def get_ingestion_service(
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service),
) -> IngestionService:
    return IngestionService(db, vector_store, llm_service)


def get_rag_service(
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service),
    cache_service: CacheService = Depends(get_cache_service),
) -> RAGService:
    return RAGService(db, vector_store, llm_service, cache_service)
