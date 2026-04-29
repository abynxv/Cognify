import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    file_size: int
    status: DocumentStatus
    message: str


class IngestRequest(BaseModel):
    reindex: bool = Field(
        default=False,
        description="If true, delete existing vectors and re-ingest",
    )


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    message: str
    task_id: str | None = None


class DeleteResponse(BaseModel):
    document_id: uuid.UUID
    message: str


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    char_count: int
    chunk_metadata: dict
    content_preview: str = ""

    @classmethod
    def from_orm_with_preview(cls, chunk: object, preview_len: int = 200) -> "ChunkResponse":
        obj = cls.model_validate(chunk)
        obj.content_preview = getattr(chunk, "content", "")[:preview_len]
        return obj
