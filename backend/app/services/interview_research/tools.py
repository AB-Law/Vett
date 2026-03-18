from __future__ import annotations

import logging
import httpx
import re
from html import unescape
from typing import Any

from sqlalchemy.orm import Session

from ..interview_docs import fetch_context_from_interview_documents
from ...config import get_settings
from ..llm import generate_embedding
from .searxng_client import SearXNGClient

logger = logging.getLogger(__name__)


DEFAULT_SEARXNG_RESULTS = 6
DEFAULT_PAGE_SNIPPET_CHARS = 900


def build_interview_questions_query(role: str, company: str | None) -> str:
    role_value = (role or "").strip()
    company_value = (company or "").strip()
    if company_value:
        return (
            f'{role_value} "{company_value}" interview questions '
            f'"site:glassdoor.com" OR site:reddit.com OR site:leetcode.com'
        )
    if not role_value:
        return 'interview questions site:glassdoor.com OR site:reddit.com OR site:leetcode.com'
    return f'{role_value} interview questions site:glassdoor.com OR site:reddit.com OR site:leetcode.com'


def build_role_skills_query(role: str, company: str | None) -> str:
    role_value = (role or "").strip()
    company_fragment = f'"{company}"' if company else ""
    base_role = role_value or "software engineer"
    return f'{base_role} {company_fragment} commonly tested skills interview {"site:linkedin.com/jobs" if company_fragment else ""}'.strip()


def build_company_culture_query(company: str, role: str | None = None) -> str:
    company_value = (company or "").strip()
    role_fragment = (role or "").strip()
    prefix = f'"{company_value}" {role_fragment} engineering culture and architecture'.strip()
    return f'{prefix} site:eng.indeed.com OR site:medium.com OR site:reflectoring.io OR site:github.com'


def build_distributed_systems_followup_query(role: str | None, company: str | None, focus: str | None = None) -> str:
    role_value = (role or "engineering").strip()
    company_value = (company or "").strip()
    followup = (focus or "distributed systems").strip()
    company_part = f'"{company_value}" ' if company_value else ""
    return f'{company_part}{role_value} {followup} interview questions site:github.com OR site:infoq.com'


def _coerce_questions_from_snippet(snippet: str, title: str, limit: int = 4) -> list[str]:
    text = unescape((snippet or "").strip())
    if not text:
        return []
    if title and title.lower() not in text.lower():
        text = f"{title}. {text}"
    # Pull short question-like lines/sentences.
    parts = re.split(r"(?<=[.!?])\\s+", text)
    candidates: list[str] = []
    for part in parts:
        normalized = " ".join((part or "").split())
        if not normalized:
            continue
        if len(normalized) < 18:
            continue
        if "?" in normalized or len(normalized.split()) >= 6:
            candidates.append(normalized)
        if len(candidates) >= limit:
            break
    if not candidates and text:
        candidates.append(f"Discuss what you know about: {text[:160]}...")
    return candidates


def _safe_tool_payload(tool: str, query: str, result: Any, *, confidence: float = 0.5, source_title: str | None = None) -> list[dict[str, str | float]]:
    if result is None:
        return []
    url = str(result.get("url", "") or "").strip()
    title = str(result.get("title", source_title or "") or "").strip()
    snippet = str(result.get("snippet", result.get("content", "") or "") or "").strip()
    entries = _coerce_questions_from_snippet(snippet, title)
    return [
        {
            "question": entry,
            "tool": tool,
            "query": query,
            "source_url": url,
            "source_title": title,
            "snippet": snippet[:2500],
            "confidence_score": confidence,
        }
        for entry in entries
    ]


async def search_interview_questions(role: str, company: str | None, *, max_items: int = DEFAULT_SEARXNG_RESULTS) -> list[dict[str, str | float]]:
    settings = get_settings()
    if not settings.searxng_enabled:
        return []
    query = build_interview_questions_query(role, company)
    client = SearXNGClient(
        settings.searxng_base_url,
        timeout_seconds=settings.searxng_timeout_seconds,
        client_ip=settings.searxng_client_ip,
    )
    results = await client.search(query, max_results=max_items)
    output: list[dict[str, str | float]] = []
    for idx, result in enumerate(results):
        mapped = _safe_tool_payload(
            "search_interview_questions",
            query,
            {"title": result.title, "url": result.url, "snippet": result.snippet},
            confidence=0.81 - (idx * 0.04),
            source_title=result.title,
        )
        output.extend(mapped)
    return output[:max_items]


async def search_role_skills(role: str, company: str | None, *, max_items: int = DEFAULT_SEARXNG_RESULTS) -> list[dict[str, str | float]]:
    settings = get_settings()
    if not settings.searxng_enabled:
        return []
    query = build_role_skills_query(role, company)
    client = SearXNGClient(
        settings.searxng_base_url,
        timeout_seconds=settings.searxng_timeout_seconds,
        client_ip=settings.searxng_client_ip,
    )
    results = await client.search(query, max_results=max_items)
    output: list[dict[str, str | float]] = []
    for idx, result in enumerate(results):
        mapped = _safe_tool_payload(
            "search_role_skills",
            query,
            {"title": result.title, "url": result.url, "snippet": result.snippet},
            confidence=0.78 - (idx * 0.03),
            source_title=result.title,
        )
        output.extend(mapped)
    return output[:max_items]


async def search_company_engineering_culture(company: str, role: str | None = None, *, max_items: int = DEFAULT_SEARXNG_RESULTS) -> list[dict[str, str | float]]:
    settings = get_settings()
    if not settings.searxng_enabled:
        return []
    query = build_company_culture_query(company, role)
    client = SearXNGClient(
        settings.searxng_base_url,
        timeout_seconds=settings.searxng_timeout_seconds,
        client_ip=settings.searxng_client_ip,
    )
    results = await client.search(query, max_results=max_items)
    output: list[dict[str, str | float]] = []
    for idx, result in enumerate(results):
        mapped = _safe_tool_payload(
            "search_company_engineering_culture",
            query,
            {"title": result.title, "url": result.url, "snippet": result.snippet},
            confidence=0.73 - (idx * 0.03),
            source_title=result.title,
        )
        output.extend(mapped)
    return output[:max_items]


async def query_vector_store(
    db: Session,
    query: str,
    job_id: int | None,
    *,
    max_items: int = 4,
    fetch_limit: int = 4,
) -> list[dict[str, str | float]]:
    if not query.strip():
        return []
    try:
        embeddings = await generate_embedding(query)
    except Exception as exc:
        logger.warning("query_vector_store skipped due to embedding failure. %s", exc)
        return []
    if not embeddings:
        return []

    rows = fetch_context_from_interview_documents(
        db,
        embeddings,
        job_id=job_id,
        scope_limit=max_items,
    )
    output: list[dict[str, str | float]] = []
    for row in rows:
        snippet = str(row.get("snippet", "") or "")
        title = str(row.get("filename", "") or "")
        if not snippet.strip():
            continue
        for question in _coerce_questions_from_snippet(snippet, title, limit=fetch_limit):
            output.append(
                {
                    "question": question,
                    "tool": "query_vector_store",
                    "query": query,
                    "source_url": f"vector://interview-doc/{row.get('chunk_id', '')}",
                    "source_title": title,
                    "snippet": snippet[:2500],
                    "confidence_score": 0.63,
                }
            )
    return output[:max_items * fetch_limit]


def _strip_html(content: str) -> str:
    if not content:
        return ""
    content = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", content)
    content = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", content)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"\s+", " ", content)
    return unescape(content).strip()


async def fetch_page(url: str, *, max_chars: int = DEFAULT_PAGE_SNIPPET_CHARS) -> dict[str, str]:
    if not url.lower().startswith(("http://", "https://")):
        return {"url": "", "title": "", "snippet": "", "error": "invalid_url"}
    settings = get_settings()
    timeout = max(2, settings.searxng_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
    except Exception as exc:
        return {"url": url, "title": "", "snippet": "", "error": str(exc)}
    if response.status_code >= 400:
        return {"url": url, "title": "", "snippet": "", "error": f"HTTP {response.status_code}"}
    html = response.text or ""
    text = _strip_html(html)
    if len(text) > max_chars:
        text = text[:max_chars]
    title = ""
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    if title_match:
        title = unescape(title_match.group(1)).strip()
    return {"url": url, "title": title, "snippet": text, "error": ""}
