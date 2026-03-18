from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SearXNGError(RuntimeError):
    pass


class SearchRequestError(SearXNGError):
    pass


class _SearchResponse(BaseModel):
    results: list[dict[str, Any]] = []


@dataclass
class SearXNGResult:
    title: str
    url: str
    snippet: str
    source: str


class SearXNGClient:
    """Small async wrapper around a self-hosted SearXNG endpoint."""

    def __init__(self, base_url: str, timeout_seconds: int = 8, client_ip: str = "127.0.0.1"):
        base_url = (base_url or "").rstrip("/")
        if not base_url:
            raise SearchRequestError("SearXNG base URL is empty")
        self.base_url = base_url
        self.timeout_seconds = max(2, int(timeout_seconds))
        self.client_ip = (client_ip or "127.0.0.1").strip() or "127.0.0.1"

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "X-Forwarded-For": self.client_ip,
            "X-Real-IP": self.client_ip,
            "X-Forwarded-Proto": "http",
        }

    async def search(
        self,
        query: str,
        max_results: int = 5,
        engines: str = "bing",
    ) -> list[SearXNGResult]:
        if not query:
            return []
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en-US",
            "engines": engines,
        }
        if max_results and max_results > 0:
            params["pageno"] = 1
            params["count"] = int(max_results)
        url = f"{self.base_url}/search"
        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params, headers=self._request_headers())
        except httpx.TimeoutException as exc:
            raise SearchRequestError(f"SearXNG search timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise SearchRequestError(f"SearXNG request failed: {exc}") from exc

        if response.status_code >= 400:
            raise SearchRequestError(
                f"SearXNG request failed with HTTP {response.status_code}: "
                f"{response.text[:250].strip()}"
            )

        try:
            payload = _SearchResponse.model_validate(response.json())
        except Exception as exc:
            raise SearchRequestError(f"SearXNG response was not valid JSON: {exc}") from exc

        results: list[SearXNGResult] = []
        for row in payload.results[:max_results]:
            result = self._coerce_result(row)
            if result is None:
                continue
            results.append(result)
        return results

    @staticmethod
    def _coerce_result(row: dict[str, Any]) -> SearXNGResult | None:
        title = str(row.get("title", "") or "").strip()
        url = str(row.get("url", "") or "").strip()
        snippet = str(row.get("content", "") or row.get("snippet", "") or "").strip()
        source = str(row.get("engine", "") or row.get("source", "") or "").strip()
        if not title or not url:
            return None
        return SearXNGResult(title=title, url=url, snippet=snippet, source=source)
