from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class CognifyError(Exception):
    """Base exception for all Cognify errors."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class DocumentNotFoundError(CognifyError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Document not found"


class DocumentAlreadyIngestedError(CognifyError):
    status_code = status.HTTP_409_CONFLICT
    detail = "Document has already been ingested"


class DocumentNotIngestedError(CognifyError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Document has not been ingested yet"


class UnsupportedFileTypeError(CognifyError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    detail = "File type not supported. Allowed: PDF, TXT"


class FileTooLargeError(CognifyError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    detail = "File exceeds maximum allowed size"


class FileProcessingError(CognifyError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Failed to process document"


class InsufficientContextError(CognifyError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "No relevant context found for the query"


class EmbeddingError(CognifyError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Failed to generate embeddings"


class LLMError(CognifyError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "LLM service is unavailable"


class CacheError(CognifyError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Cache operation failed"


class IngestionInProgressError(CognifyError):
    status_code = status.HTTP_409_CONFLICT
    detail = "Document ingestion is already in progress"


# ── FastAPI Exception Handlers ────────────────────────────────────────────────

async def cognify_exception_handler(request: Request, exc: CognifyError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.error("unhandled_exception", path=str(request.url.path), error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path),
        },
    )
