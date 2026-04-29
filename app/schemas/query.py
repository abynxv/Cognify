import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Restrict search to specific documents. Searches all if omitted.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    stream: bool = Field(default=True, description="Whether to stream the response via SSE")


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    content_preview: str
    relevance_score: float


class QueryResponse(BaseModel):
    query_id: uuid.UUID
    query: str
    answer: str
    sources: list[SourceReference]
    model_used: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int
    was_cached: bool = False
    created_at: datetime


# ── SSE Event schemas ──────────────────────────────────────────────────────────

class SSETokenEvent(BaseModel):
    """Emitted for each streamed token."""
    type: str = "token"
    content: str


class SSEDoneEvent(BaseModel):
    """Final event — includes full metadata after generation completes."""
    type: str = "done"
    query_id: str
    sources: list[SourceReference]
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int


class SSEErrorEvent(BaseModel):
    type: str = "error"
    message: str


class QueryHistoryResponse(BaseModel):
    id: uuid.UUID
    query: str
    answer: str | None
    sources: list
    latency_ms: int | None
    was_cached: bool
    created_at: datetime
