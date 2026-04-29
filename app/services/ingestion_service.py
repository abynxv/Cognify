from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import FileProcessingError
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus
from app.db.vector_store import VectorStore
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService

logger = get_logger(__name__)

# Separators follow sentence/paragraph structure — avoids mid-sentence cuts
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
    length_function=len,
    add_start_index=True,
)


class IngestionService:
    def __init__(
        self,
        db: AsyncSession,
        vector_store: VectorStore,
        llm_service: LLMService,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._doc_service = DocumentService(db)

    async def ingest(self, document_id: uuid.UUID, reindex: bool = False) -> int:
        doc = await self._doc_service.get(document_id)

        if reindex:
            await self._vector_store.delete_by_document(str(document_id))
            # Remove existing chunks from DB
            from sqlalchemy import delete
            await self._db.execute(delete(Chunk).where(Chunk.document_id == document_id))
            await self._db.flush()

        await self._doc_service.update_status(document_id, DocumentStatus.PROCESSING)

        try:
            pages = await self._extract_text(doc)
            chunks = self._create_chunks(doc, pages)
            chunk_count = await self._embed_and_store(doc, chunks)
            await self._doc_service.update_status(
                document_id, DocumentStatus.INGESTED, chunk_count=chunk_count
            )
            logger.info("ingestion_complete", document_id=str(document_id), chunks=chunk_count)
            return chunk_count
        except Exception as exc:
            await self._doc_service.update_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
            logger.error("ingestion_failed", document_id=str(document_id), error=str(exc))
            raise

    async def _extract_text(self, doc: Document) -> list[dict[str, Any]]:
        """
        Returns list of {"page": int, "text": str} dicts.
        TXT files are treated as page 1.
        """
        path = Path(doc.file_path)
        if not path.exists():
            raise FileProcessingError(f"File not found at {doc.file_path}")

        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                return await self._extract_pdf(path)
            elif suffix == ".txt":
                return await self._extract_txt(path)
            else:
                raise FileProcessingError(f"Unsupported file type: {suffix}")
        except FileProcessingError:
            raise
        except Exception as exc:
            raise FileProcessingError(f"Text extraction failed: {exc}") from exc

    async def _extract_pdf(self, path: Path) -> list[dict[str, Any]]:
        import asyncio
        loop = asyncio.get_event_loop()
        # pdfplumber is sync — run in executor to avoid blocking
        return await loop.run_in_executor(None, self._sync_extract_pdf, path)

    @staticmethod
    def _sync_extract_pdf(path: Path) -> list[dict[str, Any]]:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({"page": i, "text": text})
        if not pages:
            raise FileProcessingError("PDF contains no extractable text (may be image-only)")
        return pages

    @staticmethod
    async def _extract_txt(path: Path) -> list[dict[str, Any]]:
        import aiofiles
        async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
            text = await f.read()
        return [{"page": 1, "text": text.strip()}]

    def _create_chunks(
        self, doc: Document, pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        all_chunks: list[dict[str, Any]] = []
        chunk_index = 0

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]
            splits = _SPLITTER.create_documents(
                [text], metadatas=[{"page": page_num}]
            )
            for split in splits:
                all_chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "document_id": str(doc.id),
                        "chunk_index": chunk_index,
                        "content": split.page_content,
                        "char_count": len(split.page_content),
                        "metadata": {
                            "page_number": page_num,
                            "start_index": split.metadata.get("start_index"),
                            "document_name": doc.original_filename,
                        },
                    }
                )
                chunk_index += 1

        return all_chunks

    async def _embed_and_store(
        self, doc: Document, chunks: list[dict[str, Any]]
    ) -> int:
        batch_size = settings.EMBEDDING_BATCH_SIZE
        db_chunks: list[Chunk] = []

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [c["content"] for c in batch]

            embeddings = await self._llm_service.embed(texts)

            # Persist to Qdrant
            await self._vector_store.upsert_chunks(
                chunk_ids=[c["id"] for c in batch],
                embeddings=embeddings,
                payloads=[
                    {
                        "document_id": c["document_id"],
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        **c["metadata"],
                    }
                    for c in batch
                ],
            )

            # Build ORM objects for SQL
            for chunk_data in batch:
                db_chunks.append(
                    Chunk(
                        id=uuid.UUID(chunk_data["id"]),
                        document_id=doc.id,
                        chunk_index=chunk_data["chunk_index"],
                        content=chunk_data["content"],
                        char_count=chunk_data["char_count"],
                        chunk_metadata=chunk_data["metadata"],
                    )
                )

        self._db.add_all(db_chunks)
        await self._db.flush()
        return len(db_chunks)
