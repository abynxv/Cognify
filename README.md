# Cognify

**Production-ready RAG (Retrieval-Augmented Generation) backend built with FastAPI.**

Upload documents → Ingest into vector DB → Query with streaming LLM responses + citations.

---

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Uvicorn |
| Vector DB | Qdrant |
| LLM / Embeddings | OpenAI (gpt-4o / text-embedding-3-small) |
| Relational DB | PostgreSQL + SQLAlchemy (async) |
| Cache | Redis |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| PDF Extraction | pdfplumber |
| Retries | Tenacity |
| Rate Limiting | SlowAPI |
| Logging | Structlog (JSON in prod, colored in dev) |
| Migrations | Alembic |

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/your-username/cognify
cd cognify
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

API is live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### 3. Run locally (dev)

```bash
# Start infrastructure only
docker compose up -d postgres qdrant redis

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run migrations
alembic upgrade head

uvicorn app.main:app --reload
```

---

## API Reference

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf"
```

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "report.pdf",
  "file_size": 204800,
  "status": "pending",
  "message": "Document uploaded successfully. Call POST /ingest/{document_id} to process."
}
```

---

### Ingest (async)

```bash
curl -X POST http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000/ingest
```

```json
{
  "document_id": "550e8400-...",
  "status": "processing",
  "message": "Ingestion started. Poll GET /documents/{id} for status."
}
```

Poll status:
```bash
curl http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000
# { "status": "ingested", "chunk_count": 42, ... }
```

---

### Query with streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "stream": true}'
```

SSE stream:
```
event: token
data: {"type": "token", "content": "The key"}

event: token
data: {"type": "token", "content": " findings are"}

event: done
data: {"type": "done", "query_id": "...", "sources": [...], "latency_ms": 1240}
```

---

### Query without streaming

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document", "stream": false}'
```

```json
{
  "query_id": "...",
  "query": "Summarize the document",
  "answer": "The document discusses...",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "document_name": "report.pdf",
      "chunk_index": 3,
      "page_number": 2,
      "content_preview": "Key findings include...",
      "relevance_score": 0.91
    }
  ],
  "model_used": "gpt-4o",
  "input_tokens": 820,
  "output_tokens": 215,
  "latency_ms": 1840,
  "was_cached": false
}
```

---

### Query with document filter

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the revenue?",
    "document_ids": ["550e8400-e29b-41d4-a716-446655440000"],
    "top_k": 3,
    "stream": false
  }'
```

---

### Delete a document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000
```

---

### Re-index a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/550e8400-.../ingest \
  -H "Content-Type: application/json" \
  -d '{"reindex": true}'
```

---

## Project Structure

```
cognify/
├── app/
│   ├── api/
│   │   ├── dependencies.py        # DI wiring
│   │   └── routes/
│   │       ├── documents.py       # upload, ingest, list, delete
│   │       └── query.py           # RAG query + history
│   ├── core/
│   │   ├── config.py              # Pydantic settings (env-driven)
│   │   ├── exceptions.py          # Domain exceptions + handlers
│   │   ├── logging.py             # Structlog setup
│   │   └── rate_limiter.py        # SlowAPI
│   ├── db/
│   │   ├── database.py            # Async SQLAlchemy engine
│   │   ├── models.py              # Document, Chunk, QueryHistory ORM
│   │   └── vector_store.py        # Qdrant async wrapper
│   ├── schemas/
│   │   ├── document.py            # Upload/ingest Pydantic schemas
│   │   └── query.py               # Query request/response/SSE schemas
│   ├── services/
│   │   ├── document_service.py    # File upload, metadata CRUD
│   │   ├── ingestion_service.py   # PDF/TXT → chunks → embeddings → Qdrant
│   │   ├── retrieval_service.py   # Semantic search + context builder
│   │   ├── llm_service.py         # OpenAI wrapper (embed + stream + retry)
│   │   ├── rag_service.py         # Orchestrates retrieval + generation
│   │   └── cache_service.py       # Redis query cache
│   ├── workers/
│   │   └── ingestion_worker.py    # Background task entry point
│   └── main.py                    # App factory, lifespan, middleware
├── alembic/                       # DB migrations
├── tests/
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

---

## Configuration

All config lives in `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_LLM_MODEL` | `gpt-4o` | Swap to `gpt-4o-mini` to cut costs |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.65` | Min cosine similarity to include a chunk |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `CACHE_TTL` | `3600` | Query cache TTL in seconds |
| `RATE_LIMIT_QUERY` | `30/minute` | Per-IP query rate limit |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size |

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=app
```

---

## Production Notes

- **Scale ingestion**: Replace `BackgroundTasks` with a Celery/ARQ worker queue for large document volumes.
- **Auth**: Add JWT middleware in `app/main.py` and swap `get_remote_address` for a user-ID key function in `rate_limiter.py`.
- **Qdrant Cloud**: Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` to use managed Qdrant.
- **Migrations**: Run `alembic upgrade head` in CI before deploying — never use `create_tables()` in production.
- **Observability**: Structlog emits JSON in production — pipe to Datadog / Loki / CloudWatch.
