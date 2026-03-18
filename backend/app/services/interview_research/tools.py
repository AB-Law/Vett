"""Interview research helper tools with prompt-driven safety controls."""

from __future__ import annotations

import asyncio
import logging
import re
from html import unescape
from typing import Any
from urllib.parse import urlparse

import httpx

from ..interview_docs import fetch_context_from_interview_documents
from ...config import get_settings
from ..llm import generate_embedding
from .searxng_client import SearXNGClient

logger = logging.getLogger(__name__)

DEFAULT_SEARXNG_RESULTS = 6
DEFAULT_PAGE_SNIPPET_CHARS = 900
DEFAULT_SEARCH_RETRIES = 2


def _clean_query_term(value: str | None) -> str:
    value = (value or "").strip()
    value = re.sub(r"\([^)]*\)", "", value)
    return " ".join(value.split())


def _candidate_queries(*queries: str) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for query in queries:
        normalized = " ".join((query or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def build_interview_questions_query(role: str, company: str | None) -> str:
    role_value = _clean_query_term(role)
    company_value = _clean_query_term(company)
    if company_value:
        return f'{role_value} "{company_value}" interview questions'
    if not role_value:
        return "interview questions"
    return f'{role_value} interview questions'


def build_role_skills_query(role: str, company: str | None) -> str:
    role_value = _clean_query_term(role)
    company_fragment = f'"{_clean_query_term(company)}"' if company else ""
    base_role = role_value or "software engineer"
    return f'{base_role} {company_fragment} commonly tested skills interview'.strip()


def build_company_culture_query(company: str, role: str | None = None) -> str:
    company_value = _clean_query_term(company)
    role_fragment = _clean_query_term(role)
    return f'{company_value} {role_fragment} engineering culture and architecture'.strip()


def build_distributed_systems_followup_query(role: str | None, company: str | None, focus: str | None = None) -> str:
    role_value = (role or "engineering").strip()
    company_value = (company or "").strip()
    followup = (focus or "distributed systems").strip()
    company_part = f'"{company_value}" ' if company_value else ""
    return f'{company_part}{role_value} {followup} interview questions'


def _coerce_questions_from_snippet(snippet: str, title: str, limit: int = 4) -> list[str]:
    text = unescape((snippet or "").strip())
    if not text:
        return []
    if title and title.lower() not in text.lower():
        text = f"{title}. {text}"
    parts = re.split(r"(?<=[.!?])\s+", text)
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


def _parse_domain_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = re.split(r"[,;\n]", raw)
    return [value.strip().lower() for value in values if value and value.strip()]


def _normalize_url_for_match(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower().strip()
    path = (parsed.path or "").lower()
    return f"{host}{path}"


def _matches_domain_pattern(value: str, pattern: str) -> bool:
    if not value or not pattern:
        return False
    candidate = value.lower()
    cleaned = pattern.strip().lower()
    if cleaned == "*":
        return True
    if cleaned.startswith("*."):
        cleaned = cleaned[2:]
        return candidate == cleaned or candidate.endswith(f".{cleaned}")
    return (
        candidate == cleaned
        or candidate.startswith(f"{cleaned}/")
        or candidate.endswith(f".{cleaned}")
        or f".{cleaned}/" in candidate
        or cleaned in candidate
    )


def _split_policy(settings: Any) -> tuple[list[str], list[str]]:
    blocked = _parse_domain_list(settings.interview_research_blocked_domains)
    allowed = _parse_domain_list(settings.interview_research_allowed_domains)
    return blocked, allowed


def _is_url_allowed(url: str, settings: Any) -> tuple[bool, str]:
    blocked, allowed = _split_policy(settings)
    if not url:
        return False, "invalid_url"
    normalized = _normalize_url_for_match(url)
    if allowed:
        if any(_matches_domain_pattern(normalized, pattern) for pattern in allowed):
            return True, ""
        return False, "not_in_allowed_domains"
    if any(_matches_domain_pattern(normalized, pattern) for pattern in blocked):
        return False, "blocked_domain"
    return True, ""


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return fallback
    return max(0.0, min(1.0, parsed))


def _safe_tool_payload(
    tool: str,
    query: str,
    result: Any,
    *,
    confidence: float = 0.5,
    source_title: str | None = None,
    source_type: str = "search",
    reason: str = "Derived from source snippet.",
    question_cap: int = 3,
) -> list[dict[str, Any]]:
    if result is None:
        return []
    url = str(result.get("url", "") or "").strip()
    title = str(result.get("title", source_title or "") or "").strip()
    snippet = str(result.get("snippet", result.get("content", "") or "") or "").strip()
    if not snippet:
        return []
    entries = _coerce_questions_from_snippet(snippet, title, limit=question_cap)
    return [
        {
            "question": entry,
            "question_text": entry,
            "category": "",
            "tool": tool,
            "query": query,
            "query_used": query,
            "source_url": url,
            "source_title": title,
            "source_type": source_type,
            "snippet": snippet[:2500],
            "confidence_score": _coerce_float(confidence, 0.5),
            "reason": reason,
        }
        for entry in entries
    ]


async def search_web(
    query: str,
    *,
    max_items: int = DEFAULT_SEARXNG_RESULTS,
    confidence: float = 0.78,
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.searxng_enabled:
        return []
    normalized_query = " ".join((query or "").split())
    if not normalized_query:
        return []
    attempts = max(1, DEFAULT_SEARCH_RETRIES)
    output: list[dict[str, Any]] = []
    dropped_domains_count = 0
    last_exception: Exception | None = None
    logger.info("search_web.start | query=%r max_items=%s", normalized_query, max_items)
    blocked_urls: list[str] = []

    for attempt in range(attempts):
        client = SearXNGClient(
            settings.searxng_base_url,
            timeout_seconds=settings.searxng_timeout_seconds,
            client_ip=settings.searxng_client_ip,
        )
        try:
            results = await client.search(normalized_query, max_results=max_items)
            logger.info(
                "search_web.engine_ok | attempt=%s query=%r raw_results=%s",
                attempt + 1,
                normalized_query,
                len(results),
            )
        except Exception as exc:
            last_exception = exc
            logger.warning("search_web query failed: %s", exc)
            await asyncio.sleep(0.1)
            continue

        for idx, result in enumerate(results):
            url = str(getattr(result, "url", "") or "")
            allowed, reason = _is_url_allowed(url, settings)
            if not allowed:
                dropped_domains_count += 1
                blocked_urls.append(url)
                logger.info("search_web.filtered_domain | query=%r reason=%s url=%r", normalized_query, reason, url)
                continue
            mapped = _safe_tool_payload(
                "search_web",
                normalized_query,
                {
                    "title": str(getattr(result, "title", "") or ""),
                    "url": url,
                    "snippet": str(getattr(result, "snippet", "") or ""),
                },
                confidence=confidence - (idx * 0.05),
                source_type="search",
                reason=f"Search result for query '{normalized_query}'.",
            )
            output.extend(mapped)
        if output:
            break

    if blocked_urls:
        logger.info(
            "search_web.blocked | query=%r blocked_count=%s sample=%s",
            normalized_query,
            len(blocked_urls),
            ", ".join(blocked_urls[:5]),
        )
    if metrics is not None:
        metrics["search_web_dropped_domains_count"] = (
            metrics.get("search_web_dropped_domains_count", 0) + dropped_domains_count
        )
    if not output:
        logger.warning(
            "search_web.no_results | query=%r attempts=%s error=%s",
            normalized_query,
            attempts,
            str(last_exception or ""),
        )
    else:
        logger.info("search_web.complete | query=%r results=%s", normalized_query, len(output))
    return output[:max_items]


async def _search_via_queries(
    queries: list[str],
    *,
    max_items: int = DEFAULT_SEARXNG_RESULTS,
    confidence: float = 0.78,
    metrics: dict[str, Any] | None = None,
    tool_name: str = "search_web",
) -> list[dict[str, Any]]:
    if not queries:
        return []
    settings = get_settings()
    if not settings.searxng_enabled:
        return []
    for query in queries:
        tool_results = await search_web(query, max_items=max_items, confidence=confidence, metrics=metrics)
        for row in tool_results:
            row["tool"] = tool_name
        if tool_results:
            return tool_results
    return []


async def search_interview_questions(
    role: str,
    company: str | None,
    *,
    max_items: int = DEFAULT_SEARXNG_RESULTS,
) -> list[dict[str, str | float]]:
    role_value = _clean_query_term(role)
    company_value = _clean_query_term(company)
    queries = _candidate_queries(
        build_interview_questions_query(role_value, company_value),
        f'{role_value} interview questions',
    )
    return await _search_via_queries(queries, max_items=max_items, confidence=0.8, tool_name="search_interview_questions")


async def search_role_skills(
    role: str,
    company: str | None,
    *,
    max_items: int = DEFAULT_SEARXNG_RESULTS,
) -> list[dict[str, str | float]]:
    role_value = _clean_query_term(role)
    company_value = _clean_query_term(company)
    queries = _candidate_queries(
        build_role_skills_query(role_value, company_value),
        f'{role_value} commonly tested skills',
        f'{company_value} role skills interview'.strip(),
    )
    return await _search_via_queries(queries, max_items=max_items, confidence=0.77, tool_name="search_role_skills")


async def search_company_engineering_culture(
    company: str,
    role: str | None = None,
    *,
    max_items: int = DEFAULT_SEARXNG_RESULTS,
) -> list[dict[str, str | float]]:
    company_value = _clean_query_term(company)
    role_value = _clean_query_term(role)
    queries = _candidate_queries(
        build_company_culture_query(company_value, role_value),
        f'{company_value} engineering culture',
        f'{company_value} architecture and technical culture',
        f'{company_value} engineering blog',
    )
    return await _search_via_queries(
        queries,
        max_items=max_items,
        confidence=0.73,
        tool_name="search_company_engineering_culture",
    )


async def query_vector_store(
    db,
    query: str,
    job_id: int | None,
    *,
    max_items: int = 4,
    fetch_limit: int = 4,
) -> list[dict[str, str | float]]:
    if not query.strip():
        return []
    logger.info("query_vector_store.start | query=%r job_id=%s", query, job_id)
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
    output: list[dict[str, Any]] = []
    for row in rows:
        snippet = str(row.get("snippet", "") or "")
        title = str(row.get("filename", "") or "")
        chunk_id = str(row.get("chunk_id", "") or "").strip()
        if not snippet.strip():
            continue
        for question in _coerce_questions_from_snippet(snippet, title, limit=fetch_limit):
            output.append(
                {
                    "question": question,
                    "question_text": question,
                    "category": "",
                    "tool": "query_vector_store",
                    "query": query,
                    "query_used": query,
                    "source_url": f"vector://interview-doc/{chunk_id}",
                    "source_title": title,
                    "source_type": "vector",
                    "snippet": snippet[:2500],
                    "confidence_score": 0.63,
                    "reason": "Vector evidence against job-linked interview documents.",
                }
            )
    logger.info("query_vector_store.complete | query=%r rows=%s", query, len(output))
    return output[: max_items * fetch_limit]


async def extract_jd_facts(
    job_title: str,
    company: str | None,
    job_description: str,
) -> dict[str, Any]:
    title = _clean_query_term(job_title)
    company_value = _clean_query_term(company)
    jd_text = (job_description or "").strip()
    if not jd_text and not title and not company_value:
        return {
            "role": title,
            "company": company_value,
            "seniority": "unknown",
            "stack_keywords": [],
            "must_test_themes": [],
            "responsibilities": [],
            "raw_text_summary": "",
            "policy_hint": "No JD context provided.",
        }

    lines = [line.strip() for line in jd_text.replace("\r\n", "\n").split("\n")]
    meaningful_lines = [line for line in lines if line and len(line) > 10]
    responsibilities = [
        line
        for line in meaningful_lines
        if any(key in line.lower() for key in ("responsib", "own", "build", "ship", "maintain", "lead"))
    ][:8]
    keywords = []
    for token in re.findall(r"[A-Za-z0-9+#.-]{2,}", jd_text.lower()):
        if len(token) > 1:
            keywords.append(token)
    must_test_themes = []
    for line in meaningful_lines:
        lower = line.lower()
        if any(term in lower for term in ("must", "required", "nice to have", "required skills", "experience")):
            candidate = re.sub(r"(^|:)\s*", "", line).strip()
            if candidate:
                must_test_themes.append(candidate)
        if len(must_test_themes) >= 8:
            break
    if not title and meaningful_lines:
        title = meaningful_lines[0][:60]
    return {
        "role": title,
        "company": company_value,
        "seniority": "senior" if any(word in title.lower() for word in ("senior", "lead", "staff", "principal", "manager")) else "unknown",
        "stack_keywords": sorted(set(keywords))[:30],
        "must_test_themes": must_test_themes[:8],
        "responsibilities": responsibilities[:10],
        "raw_text_summary": " ".join(meaningful_lines[:8])[:700],
        "policy_hint": "Prefer official company pages, github engineering posts, stack/medium/blog discussions.",
    }


async def extract_candidate_profile(profile_text: str) -> dict[str, Any]:
    text = (profile_text or "").strip()
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    highlights = [line for line in lines if 30 < len(line) < 240][:8]
    return {
        "summary": " ".join(highlights)[:1600] if highlights else text[:1600],
        "highlights": highlights[:8],
        "evidence_available": bool(text),
    }


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
