from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    def __init__(self) -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )

    def _query_key(self, query: str, document_ids: list[str] | None, top_k: int) -> str:
        raw = json.dumps(
            {"q": query.lower().strip(), "docs": sorted(document_ids or []), "k": top_k},
            sort_keys=True,
        )
        return f"cognify:query:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def get_query(
        self, query: str, document_ids: list[str] | None, top_k: int
    ) -> dict[str, Any] | None:
        key = self._query_key(query, document_ids, top_k)
        try:
            cached = await self._client.get(key)
            if cached:
                logger.debug("cache_hit", key=key)
                return json.loads(cached)
        except Exception as exc:
            # Cache failure must never break the request path
            logger.warning("cache_get_failed", error=str(exc))
        return None

    async def set_query(
        self,
        query: str,
        document_ids: list[str] | None,
        top_k: int,
        value: dict[str, Any],
    ) -> None:
        key = self._query_key(query, document_ids, top_k)
        try:
            await self._client.set(key, json.dumps(value), ex=settings.CACHE_TTL)
        except Exception as exc:
            logger.warning("cache_set_failed", error=str(exc))

    async def invalidate_document(self, document_id: str) -> None:
        """Delete all cached query results that involved this document."""
        # Pattern invalidation is expensive; in production prefer versioned keys
        # or a dedicated cache index. This is a best-effort clean-up.
        try:
            pattern = "cognify:query:*"
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    raw = await self._client.get(key)
                    if raw and document_id in raw:
                        await self._client.delete(key)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("cache_invalidate_failed", error=str(exc))

    async def ping(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
