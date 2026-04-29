import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile, status

from app.api.dependencies import get_document_service, get_ingestion_service
from app.core.config import settings
from app.core.exceptions import IngestionInProgressError
from app.core.rate_limiter import limiter
from app.db.models import DocumentStatus
from app.schemas.document import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestRequest,
    IngestResponse,
    UploadResponse,
)
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.workers.ingestion_worker import run_ingestion

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document (PDF or TXT)",
)
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    doc = await doc_service.upload(file)
    return UploadResponse(
        document_id=doc.id,
        filename=doc.original_filename,
        file_size=doc.file_size,
        status=doc.status,
        message="Document uploaded successfully. Call POST /ingest/{document_id} to process.",
    )


@router.post(
    "/{document_id}/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger async ingestion for an uploaded document",
)
async def ingest_document(
    document_id: uuid.UUID,
    request_body: IngestRequest = IngestRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    doc_service: DocumentService = Depends(get_document_service),
) -> IngestResponse:
    doc = await doc_service.get(document_id)

    if doc.status == DocumentStatus.PROCESSING:
        raise IngestionInProgressError()

    # Kick off in background — returns immediately to caller
    background_tasks.add_task(run_ingestion, document_id, request_body.reindex)

    return IngestResponse(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        message="Ingestion started. Poll GET /documents/{id} for status.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents with pagination",
)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    docs, total = await doc_service.list_documents(page=page, page_size=page_size)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document metadata and ingestion status",
)
async def get_document(
    document_id: uuid.UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    doc = await doc_service.get(document_id)
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete document, its chunks, and vectors",
)
async def delete_document(
    document_id: uuid.UUID,
    doc_service: DocumentService = Depends(get_document_service),
    ingest_service: IngestionService = Depends(get_ingestion_service),
) -> DeleteResponse:
    # Remove vectors from Qdrant first
    await ingest_service._vector_store.delete_by_document(str(document_id))
    await doc_service.delete(document_id)
    return DeleteResponse(
        document_id=document_id,
        message="Document, chunks, and vectors deleted successfully.",
    )
