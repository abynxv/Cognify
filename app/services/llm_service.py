from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from app.core.config import settings
from app.core.exceptions import EmbeddingError, LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)
_tenacity_logger = logging.getLogger("tenacity")


class LLMService:
    """Async OpenAI wrapper with retries and streaming support."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._embed_model = settings.OPENAI_EMBEDDING_MODEL
        self._llm_model = settings.OPENAI_LLM_MODEL

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        try:
            return await self._embed_with_retry(texts)
        except Exception as exc:
            logger.error("embedding_failed", error=str(exc))
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(
            min=settings.LLM_RETRY_MIN_WAIT,
            max=settings.LLM_RETRY_MAX_WAIT,
        ),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )
    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            input=texts,
            model=self._embed_model,
        )
        return [item.embedding for item in response.data]

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """
        Non-streaming completion. Returns (answer, input_tokens, output_tokens).
        """
        try:
            return await self._complete_with_retry(messages, max_tokens)
        except Exception as exc:
            logger.error("llm_completion_failed", error=str(exc))
            raise LLMError(f"LLM completion failed: {exc}") from exc

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(
            min=settings.LLM_RETRY_MIN_WAIT,
            max=settings.LLM_RETRY_MAX_WAIT,
        ),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )
    async def _complete_with_retry(
        self, messages: list[dict[str, str]], max_tokens: int | None
    ) -> tuple[str, int, int]:
        response = await self._client.chat.completions.create(
            model=self._llm_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        return content, usage.prompt_tokens, usage.completion_tokens

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield streamed tokens. Caller is responsible for error handling
        since we can't retry mid-stream.
        """
        try:
            async with self._client.chat.completions.stream(
                model=self._llm_model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE,
            ) as stream:
                async for event in stream:
                    delta = event.choices[0].delta if event.choices else None
                    if delta and delta.content:
                        yield delta.content
        except (APITimeoutError, RateLimitError, APIError) as exc:
            logger.error("llm_stream_failed", error=str(exc))
            raise LLMError(f"LLM streaming failed: {exc}") from exc

    @property
    def model_name(self) -> str:
        return self._llm_model
