from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Awaitable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ...models.score import Job
from .models import InterviewResearchQuestion, InterviewResearchQuestionBank, InterviewResearchResult
from .tools import (
    build_distributed_systems_followup_query,
    fetch_page,
    query_vector_store,
    search_company_engineering_culture,
    search_interview_questions,
    search_role_skills,
)
from .tools import _coerce_questions_from_snippet
from ...config import get_settings

logger = logging.getLogger(__name__)


MIN_COUNTS: dict[str, int] = {
    "behavioral": 2,
    "technical": 4,
    "system_design": 2,
    "company_specific": 3,
}

EMIT_FUNC = Callable[[dict[str, Any]], Awaitable[None]] | None


def _normalize_question(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    if value.endswith("."):
        return value[:-1].strip()
    return value


def _unique_append(items: list[InterviewResearchQuestion], item: InterviewResearchQuestion) -> None:
    token = item.question.lower().strip()
    if not token:
        return
    for existing in items:
        if existing.question.lower().strip() == token:
            return
    if item.confidence_score is None:
        item.confidence_score = 0.5
    items.append(item)


def _pick_category(question: str, tool: str, role: str, company: str, query: str) -> str:
    text = " ".join([question, tool, query, role, company]).lower()
    if "behavior" in text or "team" in text or "conflict" in text or "feedback" in text:
        return "behavioral"
    if "system design" in text or "architecture" in text or "distributed" in text or "scale" in text or "reliability" in text:
        return "system_design"
    if company and company.lower() in text:
        return "company_specific"
    return "technical"


def _to_question(item: dict[str, Any]) -> InterviewResearchQuestion:
    question = _normalize_question(str(item.get("question") or ""))
    snippet = str(item.get("snippet") or "")
    source_title = str(item.get("source_title") or "")
    source_url = str(item.get("source_url") or "")
    tool = str(item.get("tool") or "unknown")
    query = str(item.get("query") or "")
    confidence = float(item.get("confidence_score") or 0.5)
    confidence = min(1.0, max(0.0, confidence))
    return InterviewResearchQuestion(
        question=question,
        tool=tool,
        query=query,
        source_url=source_url,
        source_title=source_title,
        snippet=snippet,
        confidence_score=confidence,
    )


def _normalize_category_output(bank: InterviewResearchQuestionBank, role: str, company: str) -> None:
    for bucket in ("behavioral", "technical", "system_design", "company_specific"):
        bank_values = getattr(bank, bucket)
        if not bucket or len(bank_values) >= MIN_COUNTS[bucket]:
            continue

        missing = MIN_COUNTS[bucket] - len(bank_values)
        fallback_seed = _fallback_questions_for_category(bucket, role, company, count=missing)
        for fallback in fallback_seed:
            question = InterviewResearchQuestion(
                question=fallback,
                tool="fallback",
                query="local_fallback",
                source_url="local://fallback",
                source_title="Internal fallback",
                snippet=fallback,
                confidence_score=0.21,
            )
            _append_to_category(bank, bucket, question)
            bank.source_urls.append("local://fallback")
    # Unique + deterministic source URL list.
    bank.source_urls = list(dict.fromkeys([u for u in bank.source_urls if u]))


def _fallback_questions_for_category(category: str, role: str, company: str, *, count: int) -> list[str]:
    company_suffix = f" for {company}" if company else ""
    role_prefix = role or "the target role"
    templates = {
        "behavioral": [
            f"Tell us about a time you handled conflicting priorities in {role_prefix}{company_suffix}.",
            f"Describe a situation where you gave a difficult piece of feedback{company_suffix}.",
            f"Describe a time you had to recover from a production incident{company_suffix}.",
            f"What is your approach when a project milestone is at risk?{company_suffix}",
            f"Walk through a difficult collaboration moment{company_suffix}.",
        ],
        "technical": [
            f"How would you explain your most important project in {role_prefix}{company_suffix}?",
            f"What tradeoffs do you apply when designing data workflows{company_suffix}?",
            f"Which debugging process helps you isolate a latency issue quickly{company_suffix}?",
            f"How do you structure API ownership for stable deployment pipelines{company_suffix}?",
            f"What are your top checks for reviewing new code from an experienced teammate{company_suffix}?",
        ],
        "system_design": [
            f"Design a scalable event-driven architecture{company_suffix} for {role_prefix}.",
            f"How would you design a caching layer for high read-heavy services{company_suffix}?",
            f"How would you design idempotent background workflows for a core platform{company_suffix}?",
            f"What observability design would you choose for critical paths in {company_suffix}?",
        ],
        "company_specific": [
            f"What would you research first about {company or 'this company'}{company_suffix} before interviewing?",
            f"How would you prepare questions about product decisions at {company or 'this company'}?",
            f"How would you assess team norms and communication style when joining {company or 'a company'}?",
            f"How would you evaluate technical fit for a role at {company or 'this company'}?",
        ],
    }
    return templates.get(category, [])[:count]


def _append_to_category(bank: InterviewResearchQuestionBank, category: str, question: InterviewResearchQuestion) -> None:
    if category == "behavioral":
        _unique_append(bank.behavioral, question)
    elif category == "technical":
        _unique_append(bank.technical, question)
    elif category == "system_design":
        _unique_append(bank.system_design, question)
    elif category == "company_specific":
        _unique_append(bank.company_specific, question)


async def _emit(emit: EMIT_FUNC, payload: dict[str, Any]) -> None:
    if emit is None:
        return
    await emit(payload)


async def _add_from_tool_output(
    bank: InterviewResearchQuestionBank,
    role: str,
    company: str,
    emit: EMIT_FUNC,
    stage: str,
    tool_results: list[dict[str, Any]],
) -> None:
    for item in tool_results:
        question = _to_question(item)
        if not question.question:
            continue
        category = _pick_category(question.question, question.tool, role, company, question.query)
        _append_to_category(bank, category, question)
        bank.register_source_url(question.source_url)
    await _emit(
        emit,
        {
            "type": "status",
            "stage": stage,
            "message": f"{stage} added {len(tool_results)} candidates.",
            "count": len(tool_results),
        },
    )


def _collect_context_text(items: list[dict[str, Any]]) -> str:
    fragments = []
    for entry in items:
        if snippet := str(entry.get("snippet", "") or "").strip():
            fragments.append(snippet)
    return " ".join(fragments).lower()


async def _run_fetch_and_expand(
    bank: InterviewResearchQuestionBank,
    role: str,
    company: str,
    emit: EMIT_FUNC,
    urls: list[str],
    *,
    stage: str,
) -> None:
    for url in urls:
        details = await fetch_page(url)
        if details.get("error"):
            continue
        text = str(details.get("snippet", "") or "")
        if not text:
            continue
        source_title = str(details.get("title", "") or "")
        for question in _coerce_questions_from_snippet(text, source_title, limit=2):
            q_payload = {
                "question": question,
                "tool": "fetch_page",
                "query": f"fetch:{url}",
                "source_url": url,
                "source_title": source_title,
                "snippet": text[:2500],
                "confidence_score": 0.7,
            }
            question_record = _to_question(q_payload)
            category = _pick_category(question_record.question, question_record.tool, role, company, q_payload["query"])
            _append_to_category(bank, category, question_record)
            bank.register_source_url(url)
        await _emit(
            emit,
            {
                "type": "status",
                "stage": stage,
                "message": f"Fetched page: {source_title or url}",
                "source_url": url,
            },
        )


def _extract_expand_urls(items: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    urls: list[str] = []
    for item in items:
        url = str(item.get("source_url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _detect_distributed_topics(bank: InterviewResearchQuestionBank, role: str, company: str) -> list[str]:
    fragments: list[str] = [
        _collect_context_text(_coerce_as_payload(bank.behavioral)),
        _collect_context_text(_coerce_as_payload(bank.technical)),
        _collect_context_text(_coerce_as_payload(bank.system_design)),
        _collect_context_text(_coerce_as_payload(bank.company_specific)),
    ]
    if not any("kafka" in segment for segment in fragments):
        return []
    return [
        build_distributed_systems_followup_query(role, company, focus="Kafka"),
        build_distributed_systems_followup_query(role, company, focus="distributed systems"),
    ]


def _coerce_as_payload(items: list[InterviewResearchQuestion]) -> list[dict[str, Any]]:
    return [
        {
            "snippet": item.snippet,
            "source_url": item.source_url,
            "question": item.question,
            "tool": item.tool,
            "query": item.query,
        }
        for item in items
    ]


@dataclass
class InterviewResearchRunContext:
    role: str
    company: str
    job: Job | None
    emit: EMIT_FUNC
    timeout_seconds: int


async def run_interview_research(
    db: Session,
    run_context: InterviewResearchRunContext,
) -> InterviewResearchResult:
    role = run_context.role.strip() or "Interview role"
    company = run_context.company.strip() or "this company"
    bank = InterviewResearchQuestionBank()
    settings = get_settings()

    emit = run_context.emit
    message = ""
    fallback_used = False

    async def emit_stage(stage: str, payload_message: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "type": "status",
            "stage": stage,
            "message": payload_message,
            "role": role,
            "company": company,
        }
        if extra:
            payload.update(extra)
        await _emit(emit, payload)

    try:
        await emit_stage("initialized", f"Researching interview questions for {role} at {company}...")
        stage_timeout = max(2, run_context.timeout_seconds)

        async def _run_with_timeout(coro):
            return await asyncio.wait_for(coro, timeout=stage_timeout)

        await emit_stage("search_interview_questions", "Searching role-specific interview questions.")
        question_hits = await _run_with_timeout(search_interview_questions(role=role, company=company, max_items=4))
        await _add_from_tool_output(bank, role, company, emit, "search_interview_questions", question_hits)

        await emit_stage("search_role_skills", "Searching commonly tested role skills.")
        role_hits = await _run_with_timeout(search_role_skills(role=role, company=company, max_items=4))
        await _add_from_tool_output(bank, role, company, emit, "search_role_skills", role_hits)

        await emit_stage("search_company_engineering_culture", "Collecting company engineering context and culture clues.")
        culture_hits = await _run_with_timeout(search_company_engineering_culture(company=company, role=role, max_items=4))
        await _add_from_tool_output(bank, role, company, emit, "search_company_engineering_culture", culture_hits)

        expand_queries = _detect_distributed_topics(bank, role, company)
        if expand_queries and settings.searxng_enabled:
            for query in expand_queries:
                await emit_stage("search_company_engineering_culture", f"Detected distributed systems signal, running expansion query: {query}")
                try:
                    hits = await _run_with_timeout(search_interview_questions(role=f"{query}", company=company, max_items=2))
                    await _add_from_tool_output(bank, role, company, emit, "search_company_engineering_culture", hits)
                except Exception as exc:
                    logger.warning("Distributed systems expansion query failed: %s", exc)

        urls = _extract_expand_urls(question_hits + role_hits + culture_hits, limit=3)
        await _run_fetch_and_expand(bank, role, company, emit, "fetch_page", urls)

        await emit_stage("query_vector_store", "Querying local interview documents for role-specific context.")
        if run_context.job is not None:
            vector_queries = [
                f"{role} behavioral interview topics",
                f"{role} technical questions and patterns",
            ]
            for query in vector_queries:
                vector_hits = await _run_with_timeout(
                    query_vector_store(db, query, job_id=run_context.job.id, max_items=3, fetch_limit=2)
                )
                await _add_from_tool_output(bank, role, company, emit, "query_vector_store", vector_hits)
                await _run_fetch_and_expand(
                    bank,
                    role,
                    company,
                    emit,
                    "fetch_page",
                    _extract_expand_urls(vector_hits, limit=1),
                )

        await emit_stage("finalizing", "Merging question bank and applying quality gates.")

    except Exception as exc:
        fallback_used = True
        message = "Web research was partially unavailable; applied fallback question generation."
        if emit is not None:
            await _emit(emit, {
                "type": "status",
                "stage": "finalizing",
                "message": str(exc),
            })

    _normalize_category_output(bank, role, company)
    final_payload = InterviewResearchResult(
        session_id="",
        role=role,
        company=company,
        status="completed",
        question_bank=bank,
        fallback_used=fallback_used,
        message=message or "Research pipeline completed.",
        metadata={"total_questions": len(bank.all_questions())},
    )
    return final_payload
