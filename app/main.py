from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import documents, query
from app.core.config import settings
from app.core.exceptions import (
    CognifyError,
    cognify_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
)
from app.core.logging import get_logger, setup_logging
from app.core.rate_limiter import limiter
from app.db.database import create_tables
from app.db.vector_store import get_vector_store
from app.services.cache_service import get_cache_service

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("cognify_starting", version=settings.APP_VERSION)

    # Ensure upload directory exists
    import pathlib
    pathlib.Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize DB tables (use Alembic for production migrations)
    await create_tables()

    # Ensure Qdrant collection + index exist
    vector_store = get_vector_store()
    await vector_store.ensure_collection()

    # Verify Redis connectivity (warn but don't fail — cache is non-critical)
    cache = get_cache_service()
    if not await cache.ping():
        logger.warning("redis_unreachable", url=settings.REDIS_URL)
    else:
        logger.info("redis_connected")

    logger.info("cognify_ready")
    yield

    logger.info("cognify_shutting_down")
    await vector_store.close()
    await cache.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-ready RAG backend. "
            "Upload documents, ingest them into a vector store, "
            "and query with streaming LLM responses."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ─────────────────────────────────────────────────────
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_exception_handler(CognifyError, cognify_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # ── Routers ────────────────────────────────────────────────────────────────
    prefix = settings.API_V1_PREFIX
    app.include_router(documents.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)

    # ── Health endpoints ───────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"], include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok", "version": settings.APP_VERSION}

    @app.get("/health/ready", tags=["Health"], include_in_schema=False)
    async def ready() -> dict:
        cache_ok = await get_cache_service().ping()
        return {
            "status": "ok",
            "cache": "connected" if cache_ok else "degraded",
        }

    return app


app = create_app()
