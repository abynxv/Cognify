"""Unit tests for ingestion service text extraction and chunking."""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from app.services.ingestion_service import IngestionService, _SPLITTER


def test_splitter_respects_chunk_size() -> None:
    long_text = "This is a sentence. " * 200  # ~4000 chars
    docs = _SPLITTER.create_documents([long_text])
    for doc in docs:
        assert len(doc.page_content) <= 1200  # size + some overlap tolerance


def test_splitter_preserves_content() -> None:
    text = "Hello world.\n\nSecond paragraph here."
    docs = _SPLITTER.create_documents([text])
    combined = " ".join(d.page_content for d in docs)
    assert "Hello world" in combined
    assert "Second paragraph" in combined


@pytest.mark.asyncio
async def test_extract_txt(tmp_path: pathlib.Path) -> None:
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Line one.\nLine two.\nLine three.")

    pages = await IngestionService._extract_txt(txt_file)
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "Line one" in pages[0]["text"]
