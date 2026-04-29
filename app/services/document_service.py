from __future__ import annotations

import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def upload(self, file: UploadFile) -> Document:
        self._validate_file(file)

        # Stream file to disk while tracking size — avoids loading into RAM
        document_id = uuid.uuid4()
        dest_dir = Path(settings.UPLOAD_DIR) / str(document_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename or "upload").name
        dest_path = dest_dir / safe_name
        total_bytes = 0

        async with aiofiles.open(dest_path, "wb") as out:
            while chunk := await file.read(1024 * 256):  # 256 KB chunks
                total_bytes += len(chunk)
                if total_bytes > settings.max_file_size_bytes:
                    dest_path.unlink(missing_ok=True)
                    raise FileTooLargeError(
                        f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit"
                    )
                await out.write(chunk)

        mime = file.content_type or "application/octet-stream"
        doc = Document(
            id=document_id,
            filename=f"{document_id}_{safe_name}",
            original_filename=safe_name,
            file_path=str(dest_path),
            file_size=total_bytes,
            mime_type=mime,
            status=DocumentStatus.PENDING,
        )
        self._db.add(doc)
        await self._db.flush()
        logger.info("document_uploaded", document_id=str(document_id), filename=safe_name, size=total_bytes)
        return doc

    async def get(self, document_id: uuid.UUID) -> Document:
        result = await self._db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        return doc

    async def list_documents(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[Document], int]:
        offset = (page - 1) * page_size
        result = await self._db.execute(
            select(Document).order_by(Document.created_at.desc()).offset(offset).limit(page_size)
        )
        docs = list(result.scalars().all())
        count_result = await self._db.execute(select(func.count(Document.id)))
        total = count_result.scalar_one()
        return docs, total

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        doc = await self.get(document_id)
        doc.status = status
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error_message is not None:
            doc.error_message = error_message
        await self._db.flush()
        return doc

    async def delete(self, document_id: uuid.UUID) -> None:
        doc = await self.get(document_id)
        file_path = Path(doc.file_path)
        parent_dir = file_path.parent

        await self._db.delete(doc)
        await self._db.flush()

        # Remove file from disk after DB delete succeeds
        try:
            file_path.unlink(missing_ok=True)
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
        except OSError as exc:
            logger.warning("file_delete_failed", path=str(file_path), error=str(exc))

        logger.info("document_deleted", document_id=str(document_id))

    def _validate_file(self, file: UploadFile) -> None:
        if not file.filename:
            raise UnsupportedFileTypeError("No filename provided")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in settings.ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Extension '{suffix}' not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
