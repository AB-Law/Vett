"""LLM-driven interview research orchestrator with tool-call planning."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from ...config import get_settings
from ..llm import _get_llm_semaphore, _get_litellm_model
from ...models.score import Job
from ...models.cv import CV
from ...models.user_profile import UserProfile
from .models import InterviewResearchQuestion, InterviewResearchQuestionBank, InterviewResearchResult
from .tools import (
    extract_candidate_profile,
    extract_jd_facts,
    query_vector_store,
    search_company_engineering_culture,
    search_interview_questions,
    search_role_skills,
    search_web,
)

try:
    import litellm  # type: ignore
except Exception:  # pragma: no cover
    litellm = None  # type: ignore

logger = logging.getLogger(__name__)

PLANNER_PROMPT_VERSION = "interview-research-planner-v1"
CLASSIFIER_PROMPT_VERSION = "interview-research-classifier-v1"
PLANNER_TOOL_MAX = 4
PLANNER_MODEL_ATTEMPTS = 2
PLANNER_TIMEOUT_SECONDS = 14
CLASSIFIER_TIMEOUT_SECONDS = 10
CLASSIFIER_MODEL_ATTEMPTS = 2
CRITIC_PROMPT_VERSION = "interview-research-critic-v1"
CRITIC_TIMEOUT_SECONDS = 10
CRITIC_MODEL_ATTEMPTS = 2
CRITIC_RECOVERY_PROMPT_VERSION = "interview-research-critic-recovery-v1"
CRITIC_RECOVERY_QUERY_MAX = 4
QUESTION_SYNTH_PROMPT_VERSION = "interview-research-question-synth-v1"
QUESTION_SYNTH_TIMEOUT_SECONDS = 10
QUESTION_SYNTH_ATTEMPTS = 2
TARGET_WEB_SOURCE_PAGES = 10
MAX_SOURCE_EXPANSION_ROUNDS = 3
DEFAULT_WEB_MAX = 5
DEFAULT_VECTOR_MAX = 3
DEFAULT_VECTOR_FETCH_LIMIT = 2
MIN_COUNTS: dict[str, int] = {
    "behavioral": 2,
    "technical": 4,
    "system_design": 2,
    "company_specific": 3,
}
VALID_CATEGORIES = {"behavioral", "technical", "system_design", "company_specific"}
EMIT_FUNC = Callable[[dict[str, Any]], Awaitable[None]] | None


def _log_stage(stage_name: str, message: str, **fields: Any) -> None:
    if fields:
        extras = ", ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("interview_research.%s | %s | %s", stage_name, message, extras)
    else:
        logger.info("interview_research.%s | %s", stage_name, message)


@dataclass
class InterviewResearchRunContext:
    role: str
    company: str
    job: Job | None
    emit: EMIT_FUNC
    timeout_seconds: int


def _clean(value: object) -> str:
    return str(value or "").strip()


def _coerce_float(value: object, fallback: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return fallback


def _signature(text: str) -> str:
    normalized = _clean(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]", "", normalized)
    normalized = re.sub(r"\b(the|a|an|and|or|to|in|on|for|with|of)\b", "", normalized)
    return re.sub(r"\s+", "", normalized)


def _coerce_category(value: object) -> str:
    text = _clean(value).lower()
    if text in VALID_CATEGORIES:
        return text
    return "technical"


def _heuristic_category(question: str, company: str) -> str:
    value = _clean(question).lower()
    if not value:
        return "technical"
    if any(item in value for item in ["behavior", "feedback", "team", "conflict", "culture", "lead", "soft"]):
        return "behavioral"
    if any(item in value for item in ["architecture", "distributed", "scale", "reliability", "system design", "latency"]):
        return "system_design"
    if company and company.lower() in value:
        return "company_specific"
    return "technical"


def _tokenize_context(text: str, limit: int = 24) -> list[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "your",
        "have",
        "will",
        "into",
        "about",
        "role",
        "team",
        "work",
        "years",
        "experience",
        "internship",
        "intern",
    }
    output: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9+#.-]{3,}", text.lower()):
        if token in stop_words or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
        if len(output) >= limit:
            break
    return output


def _source_host(source_url: str) -> str:
    try:
        return (urlparse(source_url or "").hostname or "").lower().strip()
    except Exception:
        return ""


def _deterministic_critic_decision(
    candidate: dict[str, Any],
    *,
    role: str,
    company: str,
    job_text: str,
) -> tuple[bool, str]:
    source_type = _clean(candidate.get("source_type")).lower()
    if source_type in {"vector", "fallback"}:
        return True, "trusted_source_type"

    source_url = _clean(candidate.get("source_url"))
    host = _source_host(source_url)
    title = _clean(candidate.get("source_title")).lower()
    snippet = _clean(candidate.get("snippet")).lower()
    question_text = _clean(candidate.get("question_text") or candidate.get("question")).lower()
    combined = " ".join([host, title, snippet, question_text])

    low_signal_hosts = (
        "linkedin.com",
        "youtube.com",
        "quora.com",
        "letsintern.in",
        "tracxn.com",
        "internshala.com",
    )
    if any(pattern in host for pattern in low_signal_hosts):
        return False, f"low_signal_host:{host or 'unknown'}"

    if any(term in combined for term in ("/jobs", "job vacancy", "apply now", "recruitment", "internship listing")):
        return False, "job_listing_like_source"

    company_terms = [t for t in _tokenize_context(company, limit=4) if len(t) > 2]
    role_terms = [t for t in _tokenize_context(role, limit=6) if len(t) > 2]
    jd_terms = [t for t in _tokenize_context(job_text, limit=18) if len(t) > 2]
    context_terms = company_terms + role_terms + jd_terms
    context_hits = sum(1 for term in context_terms if term and term in combined)

    if context_terms and context_hits == 0:
        return False, "no_context_alignment"
    if context_hits == 1 and source_type == "search":
        return False, "weak_context_alignment"
    return True, "context_aligned"


def _dedupe_candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        question_text = _clean(row.get("question_text") or row.get("question"))
        source_url = _clean(row.get("source_url"))
        sig = f"{_signature(question_text)}|{source_url}"
        if not question_text or sig in seen:
            continue
        seen.add(sig)
        deduped.append(row)
    return deduped


def _count_web_sources(candidates: list[dict[str, Any]]) -> int:
    unique: set[str] = set()
    for row in candidates:
        source_url = _clean(row.get("source_url")).lower()
        if source_url.startswith(("http://", "https://")):
            unique.add(source_url)
    return len(unique)


def _looks_low_value_candidate_text(text: str) -> bool:
    value = _clean(text).lower()
    # Keep deterministic gating intentionally minimal. LLM stages decide quality.
    return not value or len(value) < 12


def _heuristic_question_from_candidate(
    row: dict[str, Any],
    *,
    role: str,
    company: str,
) -> str:
    base = _clean(row.get("question_text") or row.get("question") or row.get("snippet"))
    if _looks_low_value_candidate_text(base):
        return ""
    base = re.sub(r"\s+", " ", base).strip().rstrip(".")
    if "?" in base and len(base.split()) >= 6:
        return base
    if base.lower().startswith(("how ", "what ", "why ", "when ", "where ", "which ", "can ", "do ", "is ", "are ")):
        return base + "?" if not base.endswith("?") else base
    if len(base) > 170:
        base = base[:170].rstrip(" ,.;:")
    role_context = role or "this role"
    company_context = company or "the company"
    return f"How would you apply {base.lower()} as a {role_context} candidate interviewing at {company_context}?"


def _to_question(item: dict[str, Any]) -> InterviewResearchQuestion | None:
    question_text = _clean(item.get("question_text") or item.get("question") or "")
    if not question_text:
        return None
    return InterviewResearchQuestion(
        question=question_text,
        question_text=question_text,
        category=_coerce_category(item.get("category")),
        tool=_clean(item.get("tool")),
        query=_clean(item.get("query")),
        source_url=_clean(item.get("source_url")),
        source_title=_clean(item.get("source_title")),
        source_type=_clean(item.get("source_type", "search")),
        query_used=_clean(item.get("query_used") or item.get("query", "")),
        snippet=_clean(item.get("snippet")),
        confidence_score=_coerce_float(item.get("confidence_score"), 0.5),
        reason=_clean(item.get("reason")),
    )


def _safe_parse_json(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    cleaned = _clean(content)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_or_plan(plan: Any) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    used: list[str] = []
    if not isinstance(plan, dict):
        return calls, used
    raw = plan.get("tool_calls")
    if not isinstance(raw, list):
        return calls, used
    for entry in raw[:PLANNER_TOOL_MAX]:
        if not isinstance(entry, dict):
            continue
        name = _clean(entry.get("tool"))
        arguments = entry.get("arguments")
        if name not in {"search_web", "query_vector_store", "extract_jd_facts", "extract_candidate_profile"}:
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"tool": name, "arguments": arguments})
        used.append(name)
    return calls, used


def _fallback_plan(role: str, company: str, job_text: str) -> dict[str, Any]:
    return {
        "version": PLANNER_PROMPT_VERSION,
        "tool_calls": [
            {
                "tool": "extract_jd_facts",
                "arguments": {"job_title": role, "company": company, "job_description": job_text},
            },
            {
                "tool": "search_web",
                "arguments": {
                    "queries": [
                        f"{role} interview process and expectations".strip(),
                        f"{role} {company} hiring expectations".strip() if company else f"{role} interview process",
                    ],
                    "max_items": DEFAULT_WEB_MAX,
                },
            },
            {
                "tool": "query_vector_store",
                "arguments": {
                    "query": f"{role} technical interview themes",
                    "max_items": DEFAULT_VECTOR_MAX,
                    "fetch_limit": DEFAULT_VECTOR_FETCH_LIMIT,
                },
            },
        ],
        "reason": "Fallback plan.",
    }


def _build_planner_prompt(
    role: str,
    company: str,
    job_text: str,
    profile_text: str,
    jd_facts: dict[str, Any] | None = None,
) -> list[str]:
    facts = jd_facts or {}
    stack_keywords = ", ".join([str(item) for item in facts.get("stack_keywords", [])][:12])
    responsibilities = ", ".join([str(item) for item in facts.get("responsibilities", [])][:6])
    must_test = ", ".join([str(item) for item in facts.get("must_test_themes", [])][:6])
    seniority = facts.get("seniority", "unknown")
    schema_example = {
        "version": PLANNER_PROMPT_VERSION,
        "tool_calls": [
            {
                "tool": "...",
                "arguments": {
                    "query": "...",
                },
            }
        ],
        "reason": "...",
    }
    return [
        "You are an interview research planner.",
        "Return strict JSON only.",
        "Tools: search_web, query_vector_store, extract_jd_facts, extract_candidate_profile.",
        "Never emit final interview questions from the planner; only structured tool calls.",
        f"Schema: {json.dumps(schema_example)}",
        f"Role: {role}",
        f"Company: {company}",
        f"Role seniority: {seniority}",
        f"Stack keywords: {stack_keywords}",
        f"Responsibilities: {responsibilities}",
        f"Must-test themes: {must_test}",
        f"Job description: {job_text[:900]}",
        f"Profile text available: {'yes' if profile_text else 'no'}",
        "Use web and vector tools to find evidence first, then derive 1-3 interview questions per evidence source with rationale.",
        "Prioritize official company resources, engineering blogs, GitHub, StackOverflow, Reddit, then public discussion pages.",
        "Avoid recruiter/job-board pages unless explicitly overridden by allowed domains.",
        "Avoid generic question dump pages (e.g., GeeksforGeeks or similar listicles). Prefer evidence tied to this role/company.",
        "If choosing web results, favor sources that mention the target company or explicit stack keywords from the JD.",
    ]


async def _emit(event: EMIT_FUNC, payload: dict[str, Any]) -> None:
    if event is None:
        return None
    if payload.get("type") == "status":
        status_message = _clean(payload.get("message"))
        _log_stage(
            "sse_status",
            "emit status event",
            stage=payload.get("stage", ""),
            status_message=status_message[:240],
            query=_clean(payload.get("query")),
        )
    await event(payload)


async def _planner_call(
    role: str,
    company: str,
    job_text: str,
    profile_text: str,
    jd_facts: dict[str, Any] | None,
    emit: EMIT_FUNC,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    _log_stage(
        "planner_call",
        "starting planner",
        role=role,
        company=company,
        job_text_len=len(job_text),
        has_profile=bool(profile_text),
    )
    model_id, model_kwargs = _get_litellm_model()
    plan_prompt = _build_planner_prompt(role, company, job_text, profile_text, jd_facts)
    messages = [
        {"role": "system", "content": "You output strict JSON action plans."},
        {"role": "user", "content": "\n".join(plan_prompt)},
    ]

    if not litellm:
        _log_stage("planner_call", "planner unavailable; using fallback")
        await _emit(
            emit,
            {
                "type": "status",
                "stage": "planner",
                "message": "LLM unavailable. Using fallback plan.",
            },
        )
        return model_id, _fallback_plan(role, company, job_text), [{"tool": "fallback", "status": "planner_unavailable"}]

    parsed: dict[str, Any] | None = None
    planner_latency_ms = 0
    for attempt in range(PLANNER_MODEL_ATTEMPTS):
        _log_stage("planner_call", "attempting planner", attempt=attempt + 1)
        try:
            async with _get_llm_semaphore():
                start = time.perf_counter()
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model_id,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=700,
                        **model_kwargs,
                    ),
                    timeout=timeout_seconds,
                )
                planner_latency_ms = int((time.perf_counter() - start) * 1000)
                content = getattr(response.choices[0].message, "content", "") if response and response.choices else ""
                parsed = _safe_parse_json(str(content))
            if isinstance(parsed, dict):
                break
        except Exception as exc:
            _log_stage("planner_call", "attempt failed", attempt=attempt + 1, error=str(exc))
            await _emit(emit, {
                "type": "status",
                "stage": "planner",
                "message": "Planner attempt failed.",
                "error": str(exc),
                "attempt": attempt + 1,
            })
            await asyncio.sleep(0.2)

    if not isinstance(parsed, dict):
        _log_stage("planner_call", "planner output invalid; using fallback")
        await _emit(emit, {"type": "status", "stage": "planner", "message": "Planner output invalid. Using fallback plan."})
        return model_id, _fallback_plan(role, company, job_text), [{"tool": "planner", "status": "invalid"}]

    calls, used = _parse_or_plan(parsed)
    parsed["tool_calls"] = calls
    _log_stage(
        "planner_call",
        "planner parsed successfully",
        requested_tools=used,
        total_calls=len(calls),
        latency_ms=planner_latency_ms,
        model_id=model_id,
    )
    await _emit(
        emit,
        {
            "type": "status",
            "stage": "planner",
            "message": f"Planner emitted {len(calls)} calls.",
            "tools": used,
            "model_id": model_id,
            "model_latency_ms": planner_latency_ms,
            "prompt_version": PLANNER_PROMPT_VERSION,
        },
    )
    return model_id, parsed, [{"tool": c, "status": "ok"} for c in used]


async def _run_tool_calls(
    db: Session,
    role: str,
    company: str,
    job: Job | None,
    profile_text: str,
    plan: dict[str, Any],
    emit: EMIT_FUNC,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, bool, bool]:
    calls = plan.get("tool_calls") if isinstance(plan, dict) else []
    if not isinstance(calls, list):
        calls = []
    _log_stage(
        "run_tool_calls",
        "executing tool calls",
        call_count=len(calls),
        role=role,
        company=company,
    )
    candidates: list[dict[str, Any]] = []
    research_log: list[dict[str, Any]] = []
    dropped_domains_count = 0
    search_ok = False
    vector_ok = False

    for call in calls:
        if not isinstance(call, dict):
            continue
        tool = _clean(call.get("tool"))
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        _log_stage("run_tool_calls", "dispatching tool", tool=tool)

        if tool == "extract_jd_facts":
            query_start = time.perf_counter()
            try:
                await extract_jd_facts(role, company, _clean(arguments.get("job_description", "")))
                status = "ok"
                result_count = 1
            except Exception as exc:
                status = "error"
                result_count = 0
            _log_stage(
                "run_tool_calls",
                "extract_jd_facts completed",
                status=status,
                result_count=result_count,
                duration_ms=int((time.perf_counter() - query_start) * 1000),
            )
            research_log.append(
                {
                    "stage": "extract_jd_facts",
                    "tool": tool,
                    "query": "job_description",
                    "status": status,
                    "latency_ms": int((time.perf_counter() - query_start) * 1000),
                    "result_count": result_count,
                }
            )
            if status == "error":
                await _emit(emit, {"type": "status", "stage": "extract_jd_facts", "message": str(exc)})
            continue

        if tool == "extract_candidate_profile":
            if not profile_text:
                continue
            query_start = time.perf_counter()
            try:
                await extract_candidate_profile(profile_text)
                status = "ok"
                result_count = 1
            except Exception as exc:
                status = "error"
                result_count = 0
            _log_stage(
                "run_tool_calls",
                "extract_candidate_profile completed",
                status=status,
                result_count=result_count,
                duration_ms=int((time.perf_counter() - query_start) * 1000),
            )
            research_log.append(
                {
                    "stage": "extract_candidate_profile",
                    "tool": tool,
                    "query": "profile_text",
                    "status": status,
                    "latency_ms": int((time.perf_counter() - query_start) * 1000),
                    "result_count": result_count,
                }
            )
            if status == "error":
                await _emit(emit, {"type": "status", "stage": "extract_candidate_profile", "message": str(exc)})
            continue

        if tool == "search_web":
            raw_queries = arguments.get("queries")
            if isinstance(raw_queries, list):
                queries = [q for q in [str(q).strip() for q in raw_queries] if q]
            else:
                queries = [_clean(arguments.get("query", role))]
            max_items = arguments.get("max_items", DEFAULT_WEB_MAX)
            if not isinstance(max_items, int):
                max_items = DEFAULT_WEB_MAX
            _log_stage("run_tool_calls", "search_web batch", tool=tool, query_count=len(queries), max_items=max_items)
            for query in queries[:2]:
                query_start = time.perf_counter()
                local_metrics = {"search_web_dropped_domains_count": 0}
                status = "ok"
                error_message = ""
                try:
                    hits = await asyncio.wait_for(
                        search_web(query, max_items=max_items, metrics=local_metrics),
                        timeout=timeout_seconds,
                    )
                    result_count = len(hits)
                    search_ok = search_ok or bool(hits)
                except Exception as exc:
                    hits = []
                    status = "error"
                    error_message = str(exc)
                    result_count = 0
                _log_stage(
                    "run_tool_calls",
                    "search_web completed",
                    query=query,
                    status=status,
                    result_count=result_count,
                    dropped_domains=local_metrics.get("search_web_dropped_domains_count", 0),
                    duration_ms=int((time.perf_counter() - query_start) * 1000),
                )
                dropped_domains_count += int(local_metrics.get("search_web_dropped_domains_count", 0) or 0)
                research_log.append(
                    {
                        "stage": "search_web",
                        "tool": tool,
                        "query": query,
                        "status": status,
                        "latency_ms": int((time.perf_counter() - query_start) * 1000),
                        "result_count": result_count,
                    }
                )
                if status == "error":
                    research_log[-1]["error"] = error_message
                    await _emit(emit, {"type": "status", "stage": "search_web", "message": error_message, "query": query})
                for row in hits:
                    row["tool"] = tool
                    row["query"] = query
                    row["query_used"] = row.get("query_used") or query
                    row["source_type"] = _clean(row.get("source_type") or "search")
                    candidates.append(row)
            continue

        if tool == "query_vector_store":
            raw_query = _clean(arguments.get("query", role))
            if not raw_query:
                raw_query = role
            max_items = arguments.get("max_items", DEFAULT_VECTOR_MAX)
            fetch_limit = arguments.get("fetch_limit", DEFAULT_VECTOR_FETCH_LIMIT)
            if not isinstance(max_items, int):
                max_items = DEFAULT_VECTOR_MAX
            if not isinstance(fetch_limit, int):
                fetch_limit = DEFAULT_VECTOR_FETCH_LIMIT
            query_start = time.perf_counter()
            status = "ok"
            try:
                hits = await asyncio.wait_for(
                    query_vector_store(db, raw_query, job_id=job.id if job else None, max_items=max_items, fetch_limit=fetch_limit),
                    timeout=timeout_seconds,
                )
                result_count = len(hits)
                vector_ok = vector_ok or bool(hits)
            except Exception as exc:
                status = "error"
                result_count = 0
                hits = []
            _log_stage(
                "run_tool_calls",
                "query_vector_store completed",
                query=raw_query,
                status=status,
                result_count=result_count,
                duration_ms=int((time.perf_counter() - query_start) * 1000),
            )
            research_log.append(
                {
                    "stage": "query_vector_store",
                    "tool": tool,
                    "query": raw_query,
                    "status": status,
                    "latency_ms": int((time.perf_counter() - query_start) * 1000),
                    "result_count": result_count,
                }
            )
            if status == "error":
                research_log[-1]["error"] = str(exc)
                await _emit(emit, {"type": "status", "stage": "query_vector_store", "message": str(exc), "query": raw_query})
            for row in hits:
                row["tool"] = tool
                row["query"] = raw_query
                row["query_used"] = raw_query
                row["source_type"] = _clean(row.get("source_type") or "vector")
                candidates.append(row)

    return candidates, research_log, dropped_domains_count, search_ok, vector_ok


async def _critic_candidates(
    candidates: list[dict[str, Any]],
    *,
    role: str,
    company: str,
    job_text: str,
    emit: EMIT_FUNC,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    if not candidates:
        return [], [], []

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    crit_log: list[dict[str, Any]] = []

    deterministic_start = time.perf_counter()
    for row in candidates:
        allowed, reason = _deterministic_critic_decision(
            row,
            role=role,
            company=company,
            job_text=job_text,
        )
        if not allowed:
            rejected.append(
                {
                    "source_url": _clean(row.get("source_url")),
                    "source_title": _clean(row.get("source_title")),
                    "reason": reason,
                }
            )
            continue
        kept.append(row)
    crit_log.append(
        {
            "stage": "critic",
            "tool": "deterministic_critic",
            "query": "candidate_source_review",
            "status": "ok",
            "latency_ms": int((time.perf_counter() - deterministic_start) * 1000),
            "result_count": len(kept),
            "rejected_count": len(rejected),
        }
    )
    _log_stage(
        "critic",
        "deterministic critic completed",
        input_count=len(candidates),
        kept_count=len(kept),
        rejected_count=len(rejected),
    )

    if not kept or not litellm:
        return kept, crit_log, rejected

    review_entries = []
    for idx, row in enumerate(kept):
        review_entries.append(
            {
                "index": idx,
                "source_url": _clean(row.get("source_url"))[:220],
                "source_title": _clean(row.get("source_title"))[:180],
                "question_text": _clean(row.get("question_text") or row.get("question"))[:220],
                "snippet": _clean(row.get("snippet"))[:400],
                "source_type": _clean(row.get("source_type")),
            }
        )

    model_id, model_kwargs = _get_litellm_model()
    prompt = [
        "You are a strict relevance critic for interview research evidence.",
        "Return strict JSON only.",
        "Reject low-signal links: generic internship/job boards, social media posts, random city/government pages, generic directories, and off-topic pages.",
        "Keep sources that are strongly aligned with role/company/JD context and likely to yield interview signals.",
        "Schema: {\"version\": \"interview-research-critic-v1\", \"decisions\": [{\"index\": 0, \"keep\": true, \"reason\": \"...\"}]}",
        f"Role: {role}",
        f"Company: {company}",
        f"JD context: {job_text[:700]}",
        json.dumps(review_entries),
    ]

    parsed: dict[str, Any] | None = None
    critic_latency_ms = 0
    for attempt in range(CRITIC_MODEL_ATTEMPTS):
        try:
            async with _get_llm_semaphore():
                start = time.perf_counter()
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": "Critic only, strict JSON output."},
                            {"role": "user", "content": "\n".join(prompt)},
                        ],
                        temperature=0.0,
                        max_tokens=700,
                        **model_kwargs,
                    ),
                    timeout=timeout_seconds,
                )
                critic_latency_ms = int((time.perf_counter() - start) * 1000)
                content = getattr(response.choices[0].message, "content", "") if response and response.choices else ""
                parsed = _safe_parse_json(str(content))
            if isinstance(parsed, dict):
                break
        except Exception as exc:
            _log_stage("critic", "llm critic attempt failed", attempt=attempt + 1, error=str(exc))
            await _emit(
                emit,
                {
                    "type": "status",
                    "stage": "critic",
                    "message": "Critic attempt failed.",
                    "attempt": attempt + 1,
                    "error": str(exc),
                },
            )
            await asyncio.sleep(0.2)

    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list):
        crit_log.append(
            {
                "stage": "critic",
                "tool": "llm_critic",
                "query": "candidate_source_review",
                "status": "invalid_output",
                "latency_ms": critic_latency_ms,
                "result_count": len(kept),
                "rejected_count": 0,
            }
        )
        return kept, crit_log, rejected

    reviewed: list[dict[str, Any]] = []
    llm_rejected_count = 0
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        try:
            idx = int(decision.get("index"))
        except Exception:
            continue
        if not 0 <= idx < len(kept):
            continue
        keep = bool(decision.get("keep"))
        reason = _clean(decision.get("reason")) or "critic_decision"
        if keep:
            reviewed.append(kept[idx])
            continue
        llm_rejected_count += 1
        rejected.append(
            {
                "source_url": _clean(kept[idx].get("source_url")),
                "source_title": _clean(kept[idx].get("source_title")),
                "reason": f"llm_critic:{reason}",
            }
        )

    if reviewed:
        kept = reviewed
    crit_log.append(
        {
            "stage": "critic",
            "tool": "llm_critic",
            "query": "candidate_source_review",
            "status": "ok",
            "latency_ms": critic_latency_ms,
            "result_count": len(kept),
            "rejected_count": llm_rejected_count,
        }
    )
    _log_stage(
        "critic",
        "llm critic completed",
        latency_ms=critic_latency_ms,
        kept_count=len(kept),
        rejected_count=llm_rejected_count,
        model_id=model_id,
    )
    return kept, crit_log, rejected


def _fallback_recovery_queries(
    role: str,
    company: str,
    job_text: str,
    rejected_sources: list[dict[str, str]],
) -> tuple[list[str], str]:
    jd_terms = _tokenize_context(job_text, limit=8)
    jd_fragment = " ".join(jd_terms[:4]).strip()
    rejected_reasons = [item.get("reason", "") for item in rejected_sources[:6] if isinstance(item, dict)]
    reason_hint = "; ".join([r for r in rejected_reasons if r])[:220]

    queries = [
        f"{company} engineering blog {role}".strip(),
        f"{company} {role} interview process technical expectations".strip(),
        f"{role} {company} architecture stack responsibilities {jd_fragment}".strip(),
        f"{role} practical coding scenarios {jd_fragment}".strip(),
    ]
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = " ".join(query.split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    rationale = (
        "Deterministic recovery query set focused on company engineering sources and JD-aligned technical themes."
        if not reason_hint
        else f"Deterministic recovery after critic rejections ({reason_hint})."
    )
    return normalized[:CRITIC_RECOVERY_QUERY_MAX], rationale


async def _critic_recovery_candidates(
    db: Session,
    *,
    role: str,
    company: str,
    job: Job | None,
    job_text: str,
    rejected_sources: list[dict[str, str]],
    emit: EMIT_FUNC,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int]:
    queries, rationale = _fallback_recovery_queries(role, company, job_text, rejected_sources)
    model_id, model_kwargs = _get_litellm_model()
    if litellm and rejected_sources:
        prompt = [
            "You are improving search queries after a critic rejected low-quality sources.",
            "Return strict JSON only with keys: version, rationale, queries.",
            f"Version: {CRITIC_RECOVERY_PROMPT_VERSION}",
            f"Role: {role}",
            f"Company: {company}",
            f"JD snippet: {job_text[:700]}",
            f"Rejected summary: {json.dumps(rejected_sources[:8])}",
            "Generate 3-4 high-signal queries focused on official company engineering sources and role/JD-aligned themes.",
            "Avoid social posts, generic internship listings, and random directories.",
        ]
        parsed: dict[str, Any] | None = None
        for attempt in range(CRITIC_MODEL_ATTEMPTS):
            try:
                async with _get_llm_semaphore():
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": "JSON only."},
                                {"role": "user", "content": "\n".join(prompt)},
                            ],
                            temperature=0.0,
                            max_tokens=450,
                            **model_kwargs,
                        ),
                        timeout=timeout_seconds,
                    )
                content = getattr(response.choices[0].message, "content", "") if response and response.choices else ""
                parsed = _safe_parse_json(str(content))
                if isinstance(parsed, dict):
                    break
            except Exception as exc:
                _log_stage("critic_recovery", "llm query rewrite failed", attempt=attempt + 1, error=str(exc))
                await asyncio.sleep(0.2)

        raw_queries = parsed.get("queries") if isinstance(parsed, dict) else None
        if isinstance(raw_queries, list):
            rewritten: list[str] = []
            seen: set[str] = set()
            for item in raw_queries:
                candidate = " ".join(str(item or "").split())
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                rewritten.append(candidate)
            if rewritten:
                queries = rewritten[:CRITIC_RECOVERY_QUERY_MAX]
                rationale = _clean(parsed.get("rationale")) or rationale

    await _emit(
        emit,
        {
            "type": "status",
            "stage": "critic_recovery",
            "message": f"Critic recovery running {len(queries)} refined queries.",
        },
    )
    _log_stage(
        "critic_recovery",
        "running refined recovery queries",
        query_count=len(queries),
        rationale=rationale[:260],
    )

    recovered: list[dict[str, Any]] = []
    recovery_log: list[dict[str, Any]] = []
    dropped_domains_count = 0
    for query in queries:
        query_start = time.perf_counter()
        local_metrics = {"search_web_dropped_domains_count": 0}
        try:
            rows = await asyncio.wait_for(
                search_web(query, max_items=4, metrics=local_metrics),
                timeout=timeout_seconds,
            )
            status = "ok"
        except Exception as exc:
            rows = []
            status = "error"
            _log_stage("critic_recovery", "query failed", query=query, error=str(exc))
        dropped_domains_count += int(local_metrics.get("search_web_dropped_domains_count", 0) or 0)
        for row in rows:
            row["query"] = query
            row["query_used"] = row.get("query_used") or query
            row["reason"] = _clean(row.get("reason")) or f"Critic recovery search: {rationale}"
            recovered.append(row)
        recovery_log.append(
            {
                "stage": "critic_recovery",
                "tool": "search_web",
                "query": query,
                "status": status,
                "latency_ms": int((time.perf_counter() - query_start) * 1000),
                "result_count": len(rows),
            }
        )

    if job is not None:
        vector_query = f"{role} {company} {' '.join(_tokenize_context(job_text, limit=4))} interview themes".strip()
        query_start = time.perf_counter()
        try:
            rows = await asyncio.wait_for(
                query_vector_store(db, vector_query, job_id=job.id, max_items=3, fetch_limit=2),
                timeout=timeout_seconds,
            )
            for row in rows:
                row["reason"] = _clean(row.get("reason")) or f"Critic recovery vector query: {rationale}"
                recovered.append(row)
            status = "ok"
        except Exception as exc:
            rows = []
            status = "error"
            _log_stage("critic_recovery", "vector query failed", query=vector_query, error=str(exc))
        recovery_log.append(
            {
                "stage": "critic_recovery",
                "tool": "query_vector_store",
                "query": vector_query,
                "status": status,
                "latency_ms": int((time.perf_counter() - query_start) * 1000),
                "result_count": len(rows),
            }
        )

    return recovered, recovery_log, rationale, dropped_domains_count


async def _synthesize_questions_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    role: str,
    company: str,
    job_text: str,
    emit: EMIT_FUNC,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not candidates:
        return [], [], 0

    log_rows: list[dict[str, Any]] = []
    rejected_count = 0

    # Minimal deterministic sanity only; keep quality decisions in LLM.
    base_rows: list[dict[str, Any]] = []
    for row in candidates:
        seed_text = _clean(row.get("question_text") or row.get("question") or row.get("snippet"))
        if _looks_low_value_candidate_text(seed_text):
            rejected_count += 1
            continue
        cloned = dict(row)
        if not _clean(cloned.get("question_text")):
            cloned["question_text"] = seed_text
        if not _clean(cloned.get("question")):
            cloned["question"] = _clean(cloned.get("question_text"))
        base_rows.append(cloned)

    log_rows.append(
        {
            "stage": "question_synth",
            "tool": "deterministic_sanity",
            "query": "candidate_to_question",
            "status": "ok",
            "latency_ms": 0,
            "result_count": len(base_rows),
            "rejected_count": rejected_count,
        }
    )

    if not base_rows:
        return [], log_rows, rejected_count

    if not litellm:
        # Fallback path only when LLM is unavailable.
        heuristic_rows: list[dict[str, Any]] = []
        for row in base_rows:
            rewritten = _heuristic_question_from_candidate(row, role=role, company=company)
            if not rewritten:
                continue
            updated = dict(row)
            updated["question_text"] = rewritten
            updated["question"] = rewritten
            heuristic_rows.append(updated)
        log_rows.append(
            {
                "stage": "question_synth",
                "tool": "heuristic_synth",
                "query": "candidate_to_question",
                "status": "llm_unavailable",
                "latency_ms": 0,
                "result_count": len(heuristic_rows),
                "rejected_count": max(0, len(base_rows) - len(heuristic_rows)),
            }
        )
        return heuristic_rows, log_rows, rejected_count + max(0, len(base_rows) - len(heuristic_rows))

    entries = []
    for idx, row in enumerate(base_rows):
        entries.append(
            {
                "index": idx,
                "question_text": _clean(row.get("question_text") or row.get("question"))[:220],
                "source_title": _clean(row.get("source_title"))[:160],
                "source_url": _clean(row.get("source_url"))[:220],
                "snippet": _clean(row.get("snippet"))[:400],
                "source_type": _clean(row.get("source_type")),
            }
        )

    model_id, model_kwargs = _get_litellm_model()
    prompt = [
        "You are a strict interview-question writer.",
        "Return strict JSON only.",
        "For each item, either keep and rewrite into a high-quality interview question, or reject it.",
        "Rules:",
        "- The output MUST be an interview question a candidate can answer.",
        "- Do NOT output marketing slogans, API parameter docs, or product copy.",
        "- Avoid insider-only questions requiring confidential company knowledge.",
        "- Questions should be practical and interview-ready for this exact role.",
        "- Tailor to role + JD context.",
        "Schema: {\"version\":\"interview-research-question-synth-v1\",\"items\":[{\"index\":0,\"keep\":true,\"question_text\":\"...\",\"reason\":\"...\"}]}",
        f"Role: {role}",
        f"Company: {company}",
        f"JD context: {job_text[:700]}",
        json.dumps(entries),
    ]

    parsed: dict[str, Any] | None = None
    latency_ms = 0
    for attempt in range(QUESTION_SYNTH_ATTEMPTS):
        try:
            async with _get_llm_semaphore():
                start = time.perf_counter()
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": "Question synthesis only, strict JSON."},
                            {"role": "user", "content": "\n".join(prompt)},
                        ],
                        temperature=0.0,
                        max_tokens=900,
                        **model_kwargs,
                    ),
                    timeout=timeout_seconds,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
            content = getattr(response.choices[0].message, "content", "") if response and response.choices else ""
            parsed = _safe_parse_json(str(content))
            if isinstance(parsed, dict):
                break
        except Exception as exc:
            await _emit(
                emit,
                {
                    "type": "status",
                    "stage": "question_synth",
                    "message": "Question synthesis attempt failed.",
                    "attempt": attempt + 1,
                    "error": str(exc),
                },
            )
            await asyncio.sleep(0.2)

    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        log_rows.append(
            {
                "stage": "question_synth",
                "tool": "llm_question_synth",
                "query": "candidate_to_question",
                "status": "invalid_output",
                "latency_ms": latency_ms,
                "result_count": len(base_rows),
                "rejected_count": 0,
            }
        )
        # Graceful fallback: preserve rows as-is and let classifier/fallback continue.
        return base_rows, log_rows, rejected_count

    output: list[dict[str, Any]] = []
    llm_rejected = 0
    output_seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        if not 0 <= idx < len(base_rows):
            continue
        keep = bool(item.get("keep"))
        question_text = _clean(item.get("question_text"))
        reason = _clean(item.get("reason"))
        if not keep or not question_text:
            llm_rejected += 1
            continue
        if _looks_low_value_candidate_text(question_text):
            llm_rejected += 1
            continue
        sig = _signature(question_text)
        if sig in output_seen:
            continue
        output_seen.add(sig)
        row = dict(base_rows[idx])
        row["question_text"] = question_text
        row["question"] = question_text
        if reason:
            prior_reason = _clean(row.get("reason"))
            row["reason"] = f"{prior_reason} | synth: {reason}" if prior_reason else f"synth: {reason}"
        output.append(row)

    log_rows.append(
        {
            "stage": "question_synth",
            "tool": "llm_question_synth",
            "query": "candidate_to_question",
            "status": "ok",
            "latency_ms": latency_ms,
            "result_count": len(output),
            "rejected_count": llm_rejected,
        }
    )

    target_questions = min(max(10, sum(MIN_COUNTS.values())), 18)
    need_more = max(0, target_questions - len(output))
    regen_added = 0
    regen_rejected = 0
    regen_latency_ms = 0
    if need_more > 0:
        regen_prompt = [
            "You are generating NEW interview questions from available evidence snippets.",
            "Return strict JSON only.",
            "Generate additional high-quality questions to improve coverage.",
            "Rules:",
            "- Every output MUST be an interview question.",
            "- No marketing text, field documentation, product slogans, or copied snippet fragments.",
            "- Prefer practical backend/API/system-design and role-appropriate behavioral questions.",
            "- Make questions answerable by candidates without internal confidential company details.",
            "Schema: {\"version\":\"interview-research-question-synth-v1\",\"generated\":[{\"source_index\":0,\"question_text\":\"...\",\"reason\":\"...\"}]}",
            f"Need at least {need_more} additional questions.",
            f"Role: {role}",
            f"Company: {company}",
            f"JD context: {job_text[:700]}",
            json.dumps(entries),
        ]
        regen_parsed: dict[str, Any] | None = None
        for _ in range(QUESTION_SYNTH_ATTEMPTS):
            try:
                async with _get_llm_semaphore():
                    regen_start = time.perf_counter()
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": "Generate interview questions only. Strict JSON."},
                                {"role": "user", "content": "\n".join(regen_prompt)},
                            ],
                            temperature=0.1,
                            max_tokens=1200,
                            **model_kwargs,
                        ),
                        timeout=timeout_seconds,
                    )
                    regen_latency_ms = int((time.perf_counter() - regen_start) * 1000)
                content = getattr(response.choices[0].message, "content", "") if response and response.choices else ""
                regen_parsed = _safe_parse_json(str(content))
                if isinstance(regen_parsed, dict):
                    break
            except Exception as exc:
                await _emit(
                    emit,
                    {
                        "type": "status",
                        "stage": "question_synth",
                        "message": f"Question regeneration attempt failed: {exc}",
                    },
                )
                await asyncio.sleep(0.2)

        generated = regen_parsed.get("generated") if isinstance(regen_parsed, dict) else None
        if isinstance(generated, list):
            for item in generated:
                if not isinstance(item, dict):
                    continue
                try:
                    source_index = int(item.get("source_index"))
                except Exception:
                    source_index = 0
                if not 0 <= source_index < len(base_rows):
                    source_index = 0
                question_text = _clean(item.get("question_text"))
                reason = _clean(item.get("reason"))
                if not question_text or _looks_low_value_candidate_text(question_text):
                    regen_rejected += 1
                    continue
                sig = _signature(question_text)
                if sig in output_seen:
                    continue
                output_seen.add(sig)
                row = dict(base_rows[source_index])
                row["question_text"] = question_text
                row["question"] = question_text
                prior_reason = _clean(row.get("reason"))
                if reason:
                    row["reason"] = f"{prior_reason} | regen: {reason}" if prior_reason else f"regen: {reason}"
                else:
                    row["reason"] = f"{prior_reason} | regen: Additional question from accepted evidence.".strip()
                output.append(row)
                regen_added += 1
                if len(output) >= target_questions:
                    break

    log_rows.append(
        {
            "stage": "question_synth",
            "tool": "llm_question_regen",
            "query": "regenerate_from_evidence",
            "status": "ok" if regen_added > 0 else "skipped_or_empty",
            "latency_ms": regen_latency_ms,
            "result_count": regen_added,
            "rejected_count": regen_rejected,
        }
    )
    _log_stage(
        "question_synth",
        "question synthesis complete",
        model_id=model_id,
        kept_count=len(output),
        rejected_count=rejected_count + llm_rejected + regen_rejected,
        regenerated_count=regen_added,
    )
    return output or base_rows, log_rows, rejected_count + llm_rejected + regen_rejected


async def _classify_candidates(
    candidates: list[dict[str, Any]],
    role: str,
    company: str,
    job_text: str,
    emit: EMIT_FUNC,
    timeout_seconds: int,
) -> list[InterviewResearchQuestion]:
    if not candidates:
        _log_stage("classify", "no candidates to classify")
        return []

    model_classification: dict[int, dict[str, str]] = {}
    _log_stage("classify", "starting classification", candidate_count=len(candidates), role=role, company=company)

    if litellm:
        entries = []
        for idx, row in enumerate(candidates):
            entries.append(
                {
                    "index": idx,
                    "question_text": _clean(row.get("question_text") or row.get("question", "") or "")[:220],
                    "source_type": _clean(row.get("source_type", "search")),
                    "query_used": _clean(row.get("query_used", "")),
                    "source_url": _clean(row.get("source_url", "")),
                    "tool": _clean(row.get("tool", "")),
                }
            )
        model_id, model_kwargs = _get_litellm_model()
        prompt = [
            "Classify each candidate into one category.",
            "Return strict JSON with keys version and classified list.",
            "Categories: behavioral | technical | system_design | company_specific",
            f"Role: {role}",
            f"Company: {company}",
            f"Context snippet: {job_text[:700]}",
            json.dumps(entries),
        ]
        try:
            parsed: dict[str, Any] | None = None
            latency_ms = 0
            for _ in range(CLASSIFIER_MODEL_ATTEMPTS):
                try:
                    async with _get_llm_semaphore():
                        start = time.perf_counter()
                        response = await asyncio.wait_for(
                            litellm.acompletion(
                                model=model_id,
                                messages=[
                                    {"role": "system", "content": "Classify only, strict JSON output."},
                                    {"role": "user", "content": "\n".join(prompt)},
                                ],
                                temperature=0.0,
                                max_tokens=600,
                                **model_kwargs,
                            ),
                            timeout=timeout_seconds,
                        )
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        content = getattr(response.choices[0].message, "content", "") if response and response.choices else ""
                        parsed = _safe_parse_json(str(content))
                    if isinstance(parsed, dict):
                        break
                except Exception as exc:
                    _log_stage("classify", "attempt failed", error=str(exc))
                    await _emit(emit, {"type": "status", "stage": "classify", "message": f"Classifier attempt failed: {exc}"})
                    await asyncio.sleep(0.2)
            if not isinstance(parsed, dict):
                parsed = {}
            classified = parsed.get("classified") if isinstance(parsed, dict) else None
            _log_stage(
                "classify",
                "classification output parsed",
                parsed_count=0 if not isinstance(classified, list) else len(classified),
            )
            await _emit(
                emit,
                {
                    "type": "status",
                    "stage": "classify",
                    "message": "LLM classification complete.",
                    "model_latency_ms": latency_ms,
                },
            )
            entries = classified
            if isinstance(entries, list):
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    try:
                        idx = int(idx)
                    except Exception:
                        continue
                    if not 0 <= idx < len(candidates):
                        continue
                    model_classification[idx] = {
                        "category": _clean(item.get("category")),
                        "reason": _clean(item.get("reason", "")),
                    }
        except Exception as exc:
            await _emit(emit, {"type": "status", "stage": "classify", "message": f"Classification failed: {exc}"})

    output: list[InterviewResearchQuestion] = []
    for idx, row in enumerate(candidates):
        candidate = _to_question(row)
        if candidate is None:
            continue
        details = model_classification.get(idx, {})
        category = _coerce_category(details.get("category"))
        if category not in VALID_CATEGORIES:
            category = _heuristic_category(candidate.question_text, company)
        reason = _clean(details.get("reason")) or _clean(row.get("reason", ""))
        if not reason:
            reason = "Classified from model/heuristics."
        output.append(
            candidate.model_copy(
                update={
                    "category": category,
                    "reason": reason,
                }
            )
        )
    _log_stage("classify", "classification output ready", output_count=len(output))
    return output


def _append_to_category(bank: InterviewResearchQuestionBank, question: InterviewResearchQuestion) -> None:
    category = question.category if question.category in VALID_CATEGORIES else _heuristic_category(question.question_text, question.source_title)
    destination = {
        "behavioral": bank.behavioral,
        "technical": bank.technical,
        "system_design": bank.system_design,
        "company_specific": bank.company_specific,
    }.get(category, bank.technical)

    normalized = _signature(question.question_text)
    for existing in destination:
        if _signature(existing.question_text) == normalized:
            return
    destination.append(question)


def _dedupe(bank: InterviewResearchQuestionBank) -> None:
    seen: set[str] = set()
    for bucket in (bank.behavioral, bank.technical, bank.system_design, bank.company_specific):
        items: list[InterviewResearchQuestion] = []
        for item in bucket:
            sig = _signature(item.question_text)
            if sig in seen:
                continue
            seen.add(sig)
            items.append(item)
        if len(items) != len(bucket):
            bucket.clear()
            bucket.extend(items)


def _extract_candidate_profile_context(
    db: Session,
    profile_text: str,
) -> tuple[str, str]:
    profile = db.query(UserProfile).order_by(UserProfile.id.asc()).first()
    if profile is not None:
        candidate_company = _clean(profile.current_company or "")
        candidate_summary = _clean(profile.summary or "")
        if candidate_company or candidate_summary:
            return candidate_company, candidate_summary

    # Fallback only when there is no persisted profile row yet.
    text = _clean(profile_text)
    if not text:
        return "", ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pattern = re.compile(
        r"\bat\s+([A-Z][A-Za-z0-9&.,\- ]{2,60}?)(?:\s*\||\s*[-–]\s*|\s*\(|\s*$)"
    )
    company = ""
    for line in lines[:80]:
        match = pattern.search(line)
        if not match:
            continue
        candidate = _clean(match.group(1)).strip(" -|,.;")
        if len(candidate) >= 3:
            company = candidate
            break
    summary = _clean(" ".join(lines[:10]))
    return company, summary


def _fallback_questions_for_category(
    category: str,
    role: str,
    company: str,
    count: int,
    *,
    candidate_company: str = "",
) -> list[str]:
    context_company = candidate_company or company
    context_suffix = f" at {context_company}" if context_company else ""
    role_name = role or "this role"
    target_company = company or "this company"
    prior_company = candidate_company or "your previous company"
    templates = {
        "behavioral": [
            f"Tell us about handling tradeoffs under pressure as a {role_name}{context_suffix}.",
            f"Describe how you communicated a hard technical decision to your team{context_suffix}.",
            f"How do you recover when a critical release risks missing a deadline{context_suffix}?",
        ],
        "technical": [
            f"Design a debugging strategy for a hard failing service in a {role_name} context.",
            "How do you ensure testability of critical components?",
            "Walk through your approach to prioritizing engineering work under time pressure.",
            f"Explain how you would improve reliability in legacy systems{context_suffix}.",
        ],
        "system_design": [
            "Design a horizontally scalable API that tolerates partial failures.",
            "How would you design idempotency and retries for distributed jobs?",
            "Explain how you would model data consistency with eventual updates.",
        ],
        "company_specific": [
            f"Based on your backend experience at {prior_company}, what assumptions would you validate first before proposing changes at {target_company}?",
            f"Which fintech constraints from {prior_company} seem transferable to {target_company}'s onboarding and verification workflows, and which would you re-validate?",
            f"Using the role brief and public information, what 30-60 day engineering plan would you propose for {target_company}, and what unknowns would you flag early?",
        ],
    }
    return templates.get(category, [])[:count]


def _apply_minimums(
    bank: InterviewResearchQuestionBank,
    role: str,
    company: str,
    *,
    candidate_company: str = "",
) -> bool:
    fallback_used = False
    for category, minimum in MIN_COUNTS.items():
        current = {
            "behavioral": bank.behavioral,
            "technical": bank.technical,
            "system_design": bank.system_design,
            "company_specific": bank.company_specific,
        }[category]
        if len(current) >= minimum:
            continue
        missing = minimum - len(current)
        _log_stage("fallback_minimums", "adding fallback questions", category=category, missing=missing)
        for fallback_question in _fallback_questions_for_category(
            category,
            role,
            company,
            minimum - len(current),
            candidate_company=candidate_company,
        ):
            fallback_used = True
            fallback = InterviewResearchQuestion(
                question=fallback_question,
                question_text=fallback_question,
                category=category,
                tool="fallback",
                query="local://fallback",
                query_used="local://fallback",
                source_url="local://fallback",
                source_title="Fallback source",
                source_type="fallback",
                snippet=fallback_question,
                confidence_score=0.22,
                reason="Local deterministic fallback for category coverage.",
            )
            bank.register_source_url(fallback.source_url)
            _append_to_category(bank, fallback)
        _log_stage(
            "fallback_minimums",
            "category minimum reached",
            category=category,
            count=len(
                {
                    "behavioral": bank.behavioral,
                    "technical": bank.technical,
                    "system_design": bank.system_design,
                    "company_specific": bank.company_specific,
                }[category],
            ),
        )
    return fallback_used


async def run_llm_research_agent(
    db: Session,
    run_context: InterviewResearchRunContext,
) -> InterviewResearchResult:
    role = _clean(run_context.role) or "Interview role"
    company = _clean(run_context.company) or "this company"
    emit = run_context.emit
    timeout_seconds = max(2, int(run_context.timeout_seconds or 20))
    _log_stage(
        "run_llm_research_agent",
        "starting interview research",
        role=role,
        company=company,
        timeout_seconds=timeout_seconds,
    )

    settings = get_settings()
    tool_timeout_seconds = settings.interview_research_tool_timeout_seconds if hasattr(settings, "interview_research_tool_timeout_seconds") else 10
    job_text = _clean(run_context.job.description if run_context.job else "")
    latest_cv = db.query(CV).order_by(CV.id.desc()).first()
    profile_text = _clean(latest_cv.parsed_text if latest_cv else "")
    candidate_company, resolved_profile_summary = _extract_candidate_profile_context(db, profile_text)
    profile_text_for_planner = resolved_profile_summary or profile_text
    _log_stage(
        "run_llm_research_agent",
        "resolved context",
        job_text_len=len(job_text),
        has_profile=bool(profile_text_for_planner),
        tool_timeout_seconds=tool_timeout_seconds,
        overall_timeout_seconds=timeout_seconds,
        blocked_domains_count=len((settings.interview_research_blocked_domains or "").split(",")),
        allowed_domains_set=bool(settings.interview_research_allowed_domains),
        candidate_company=candidate_company,
    )

    question_bank = InterviewResearchQuestionBank()
    fallback_used = False
    all_log: list[dict[str, Any]] = []
    rejected_sources: list[dict[str, str]] = []
    critic_recovery_rationale = ""
    question_synth_rejected_count = 0
    model_id = ""

    await _emit(
        emit,
        {
            "type": "status",
            "stage": "initialized",
            "message": f"Researching interview questions for {role} at {company}.",
        },
    )

    try:
        try:
            jd_facts = await asyncio.wait_for(
                extract_jd_facts(role, company, job_text),
                timeout=tool_timeout_seconds,
            )
        except Exception:
            jd_facts = {}
            _log_stage("run_llm_research_agent", "jd_facts extraction failed", error="using_defaults")
        else:
            _log_stage(
                "run_llm_research_agent",
                "jd_facts extracted",
                stack_count=len(jd_facts.get("stack_keywords", [])),
                responsibilities_count=len(jd_facts.get("responsibilities", [])),
                must_test_count=len(jd_facts.get("must_test_themes", [])),
                policy_hint=jd_facts.get("policy_hint", ""),
            )

        model_id, plan, planner_log = await _planner_call(
            role=role,
            company=company,
            job_text=job_text,
            profile_text=profile_text_for_planner,
            jd_facts=jd_facts,
            emit=emit,
            timeout_seconds=PLANNER_TIMEOUT_SECONDS,
        )
        all_log.extend(planner_log)
        _log_stage("run_llm_research_agent", "planner completed", plan_calls=len(plan.get("tool_calls", [])))

        candidates, action_log, dropped_domains_count, search_ok, vector_ok = await _run_tool_calls(
            db=db,
            role=role,
            company=company,
            job=run_context.job,
            profile_text=profile_text_for_planner,
            plan=plan,
            emit=emit,
            timeout_seconds=tool_timeout_seconds,
        )
        all_log.extend(action_log)
        _log_stage(
            "run_llm_research_agent",
            "tool execution completed",
            candidate_count=len(candidates),
            search_ok=search_ok,
            vector_ok=vector_ok,
            dropped_domains=dropped_domains_count,
        )

        if not candidates:
            fallback_used = True
            _log_stage("run_llm_research_agent", "no candidates; fallback search engaged")
            fallback_hits: list[dict[str, Any]] = []
            fallback_queries = [
                f"{role} {company} interview process and engineering expectations".strip(),
                f"{company} engineering blog {role}".strip(),
                f"{role} {company} system design interview questions".strip(),
                f"{role} technical interview questions".strip(),
            ]
            for fallback_query in fallback_queries:
                try:
                    start = time.perf_counter()
                    fetched = await asyncio.wait_for(search_web(fallback_query, max_items=4), timeout=tool_timeout_seconds)
                    fallback_hits.extend(fetched)
                    count = len(fetched)
                    _log_stage(
                        "run_llm_research_agent",
                        "fallback query executed",
                        status="ok",
                        query=fallback_query,
                        result_count=count,
                    )
                    all_log.append(
                        {
                            "stage": "fallback",
                            "tool": "search_web",
                            "query": "fallback",
                            "status": "ok",
                            "latency_ms": int((time.perf_counter() - start) * 1000),
                            "result_count": count,
                        }
                    )
                except Exception:
                    _log_stage(
                        "run_llm_research_agent",
                        "fallback query failed",
                        status="error",
                    )
                    all_log.append(
                        {
                            "stage": "fallback",
                            "tool": "search_web",
                            "query": "fallback",
                            "status": "error",
                            "latency_ms": 0,
                            "result_count": 0,
                        }
                    )
            if run_context.job is not None:
                try:
                    start = time.perf_counter()
                    vector_start = len(fallback_hits)
                    fallback_hits.extend(
                        await asyncio.wait_for(
                            query_vector_store(db, f"{role} interview questions", job_id=run_context.job.id, max_items=3, fetch_limit=2),
                            timeout=tool_timeout_seconds,
                        )
                    )
                    vector_count = len(fallback_hits) - vector_start
                    _log_stage(
                        "run_llm_research_agent",
                        "fallback vector executed",
                        status="ok",
                        result_count=vector_count,
                    )
                    all_log.append(
                        {
                            "stage": "fallback",
                            "tool": "query_vector_store",
                            "query": f"{role} interview questions",
                            "status": "ok",
                            "latency_ms": int((time.perf_counter() - start) * 1000),
                            "result_count": vector_count,
                        }
                    )
                except Exception:
                    _log_stage("run_llm_research_agent", "fallback vector failed", status="error")
                    all_log.append(
                        {
                            "stage": "fallback",
                            "tool": "query_vector_store",
                            "query": f"{role} interview questions",
                            "status": "error",
                            "latency_ms": 0,
                            "result_count": 0,
                        }
                    )
            for row in fallback_hits:
                candidates.append(
                    {
                        "question": row.get("question", ""),
                        "question_text": row.get("question_text", row.get("question", "")),
                        "tool": row.get("tool", "fallback"),
                        "query": row.get("query", "local://fallback"),
                        "query_used": row.get("query_used", row.get("query", "local://fallback")),
                        "source_url": row.get("source_url", "local://fallback"),
                        "source_title": row.get("source_title", "Fallback"),
                        "source_type": row.get("source_type", "fallback"),
                        "snippet": row.get("snippet", row.get("question", "Fallback")),
                        "confidence_score": row.get("confidence_score", 0.22),
                        "reason": "Deterministic fallback stage.",
                    }
                )
            _log_stage(
                "run_llm_research_agent",
                "fallback candidates appended",
                count=len(fallback_hits),
            )

        candidates, critic_log, rejected_sources = await _critic_candidates(
            candidates,
            role=role,
            company=company,
            job_text=job_text,
            emit=emit,
            timeout_seconds=CRITIC_TIMEOUT_SECONDS,
        )
        all_log.extend(critic_log)
        await _emit(
            emit,
            {
                "type": "status",
                "stage": "critic",
                "message": f"Critic kept {len(candidates)} candidates, rejected {len(rejected_sources)}.",
            },
        )
        candidates = _dedupe_candidate_rows(candidates)
        if not candidates:
            await _emit(
                emit,
                {
                    "type": "status",
                    "stage": "critic_recovery",
                    "message": "Critic rejected all candidates. Running refined recovery pass.",
                },
            )
            recovered, recovery_log, recovery_rationale, recovery_dropped = await _critic_recovery_candidates(
                db,
                role=role,
                company=company,
                job=run_context.job,
                job_text=job_text,
                rejected_sources=rejected_sources,
                emit=emit,
                timeout_seconds=tool_timeout_seconds,
            )
            critic_recovery_rationale = recovery_rationale
            all_log.extend(recovery_log)
            dropped_domains_count += int(recovery_dropped)
            _log_stage(
                "run_llm_research_agent",
                "critic recovery generated candidates",
                generated_count=len(recovered),
                rationale=recovery_rationale[:220],
            )
            recovered, recovery_critic_log, recovery_rejected = await _critic_candidates(
                recovered,
                role=role,
                company=company,
                job_text=job_text,
                emit=emit,
                timeout_seconds=CRITIC_TIMEOUT_SECONDS,
            )
            all_log.extend(recovery_critic_log)
            rejected_sources.extend(recovery_rejected)
            candidates = recovered
            candidates = _dedupe_candidate_rows(candidates)
            await _emit(
                emit,
                {
                    "type": "status",
                    "stage": "critic_recovery",
                    "message": f"Recovery critic kept {len(candidates)} candidates, rejected {len(recovery_rejected)}.",
                },
            )

        source_expansion_rounds = 0
        web_source_count = _count_web_sources(candidates)
        while web_source_count < TARGET_WEB_SOURCE_PAGES and source_expansion_rounds < MAX_SOURCE_EXPANSION_ROUNDS:
            source_expansion_rounds += 1
            await _emit(
                emit,
                {
                    "type": "status",
                    "stage": "source_expansion",
                    "message": (
                        f"Source expansion round {source_expansion_rounds}: "
                        f"{web_source_count}/{TARGET_WEB_SOURCE_PAGES} web sources."
                    ),
                },
            )
            recovered, recovery_log, recovery_rationale, recovery_dropped = await _critic_recovery_candidates(
                db,
                role=role,
                company=company,
                job=run_context.job,
                job_text=job_text,
                rejected_sources=rejected_sources,
                emit=emit,
                timeout_seconds=tool_timeout_seconds,
            )
            critic_recovery_rationale = recovery_rationale or critic_recovery_rationale
            all_log.extend(recovery_log)
            dropped_domains_count += int(recovery_dropped)
            recovered, recovery_critic_log, recovery_rejected = await _critic_candidates(
                recovered,
                role=role,
                company=company,
                job_text=job_text,
                emit=emit,
                timeout_seconds=CRITIC_TIMEOUT_SECONDS,
            )
            all_log.extend(recovery_critic_log)
            rejected_sources.extend(recovery_rejected)
            if not recovered:
                _log_stage(
                    "source_expansion",
                    "no recovered candidates in round",
                    round=source_expansion_rounds,
                )
                break
            candidates = _dedupe_candidate_rows(candidates + recovered)
            web_source_count = _count_web_sources(candidates)
            _log_stage(
                "source_expansion",
                "round completed",
                round=source_expansion_rounds,
                web_source_count=web_source_count,
                candidate_count=len(candidates),
            )

        candidates, synth_log, synth_rejected = await _synthesize_questions_from_candidates(
            candidates,
            role=role,
            company=company,
            job_text=job_text,
            emit=emit,
            timeout_seconds=QUESTION_SYNTH_TIMEOUT_SECONDS,
        )
        all_log.extend(synth_log)
        question_synth_rejected_count += int(synth_rejected)
        await _emit(
            emit,
            {
                "type": "status",
                "stage": "question_synth",
                "message": f"Question synthesis kept {len(candidates)} candidates, rejected {synth_rejected}.",
            },
        )

        enriched = await _classify_candidates(
            candidates,
            role=role,
            company=company,
            job_text=job_text,
            emit=emit,
            timeout_seconds=CLASSIFIER_TIMEOUT_SECONDS,
        )
        for question in enriched:
            if not question.source_type:
                question.source_type = "search"
            if not question.reason:
                question.reason = "Classified with LLM-aware heuristics."
            if not question.confidence_score:
                question.confidence_score = 0.25
            if not question.query_used:
                question.query_used = question.query
            _append_to_category(question_bank, question)
            question_bank.register_source_url(question.source_url)

        _dedupe(question_bank)
        if _apply_minimums(
            question_bank,
            role=role,
            company=company,
            candidate_company=candidate_company,
        ):
            fallback_used = True

        if not search_ok and not vector_ok and len(question_bank.all_questions()) < sum(MIN_COUNTS.values()):
            fallback_used = True
            _log_stage(
                "run_llm_research_agent",
                "post-processing fallback marker",
                reason="insufficient primary results",
                question_count=len(question_bank.all_questions()),
            )

        _log_stage(
            "run_llm_research_agent",
            "research completed",
            total_questions=len(question_bank.all_questions()),
            fallback_used=fallback_used,
            unique_sources=len(question_bank.source_urls),
            sample_sources="|".join(question_bank.source_urls[:5]),
            web_source_pages=_count_web_sources(candidates),
        )

        return InterviewResearchResult(
            session_id="",
            role=role,
            company=company,
            status="completed",
            question_bank=question_bank,
            fallback_used=fallback_used,
            message="Interview questions generated.",
            metadata={
                "total_questions": len(question_bank.all_questions()),
                "research_log": all_log,
                "model_input_prompt_version": PLANNER_PROMPT_VERSION,
                "classification_version": CLASSIFIER_PROMPT_VERSION,
                "critic_version": CRITIC_PROMPT_VERSION,
                "critic_recovery_version": CRITIC_RECOVERY_PROMPT_VERSION,
                "critic_recovery_rationale": critic_recovery_rationale,
                "question_synth_version": QUESTION_SYNTH_PROMPT_VERSION,
                "question_synth_rejected_count": question_synth_rejected_count,
                "target_web_source_pages": TARGET_WEB_SOURCE_PAGES,
                "achieved_web_source_pages": _count_web_sources(candidates),
                "model_id": model_id,
                "tools_used": sorted({item.get("tool") for item in all_log if isinstance(item.get("tool"), str)}),
                "dropped_domains_count": int(dropped_domains_count),
                "rejected_sources": rejected_sources[:30],
                "rejected_source_count": len(rejected_sources),
            },
        )
    except Exception as exc:
        logger.exception("run_llm_research_agent failed")
        _log_stage("run_llm_research_agent", "failed and recovered", error=str(exc), fallback_used=True)
        return InterviewResearchResult(
            session_id="",
            role=role,
            company=company,
            status="completed",
            question_bank=question_bank,
            fallback_used=True,
            message=f"Recovered from error: {exc}",
            metadata={
                "total_questions": len(question_bank.all_questions()),
                "research_log": all_log,
                "model_input_prompt_version": PLANNER_PROMPT_VERSION,
                "critic_version": CRITIC_PROMPT_VERSION,
                "critic_recovery_version": CRITIC_RECOVERY_PROMPT_VERSION,
                "critic_recovery_rationale": critic_recovery_rationale,
                "question_synth_version": QUESTION_SYNTH_PROMPT_VERSION,
                "question_synth_rejected_count": question_synth_rejected_count,
                "target_web_source_pages": TARGET_WEB_SOURCE_PAGES,
                "achieved_web_source_pages": _count_web_sources(candidates),
                "model_id": model_id,
                "tools_used": ["fallback"],
                "rejected_sources": rejected_sources[:30],
                "rejected_source_count": len(rejected_sources),
            },
        )


async def run_interview_research(db: Session, run_context: InterviewResearchRunContext) -> InterviewResearchResult:
    return await run_llm_research_agent(db, run_context)
