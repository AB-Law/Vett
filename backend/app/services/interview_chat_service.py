from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, AsyncIterator, TypedDict, cast

from sqlalchemy.orm import Session

from ..models.cv import CV
from ..models.interview_chat import InterviewChatSession, InterviewChatTurn
from ..models.interview_research import InterviewResearchSession
from ..models.score import Job
from .interview_docs import fetch_context_from_interview_documents
from .llm import (
    classify_interview_answer_vagueness,
    classify_interview_repeat_intent,
    generate_adaptive_interview_question,
    generate_embedding,
)
from .scoring_orchestrator import execute_scoring_orchestrator

logger = logging.getLogger(__name__)

QUESTION_ORDER = ["behavioral", "technical", "system_design", "company_specific"]
QUESTION_TARGETS = {"behavioral": 3, "technical": 4, "system_design": 2, "company_specific": 2}
SPECIFIC_CONTEXT_TERMS = {
    "api",
    "queue",
    "pipeline",
    "schema",
    "latency",
    "throughput",
    "incident",
    "sla",
    "slo",
    "audit",
    "compliance",
    "rollback",
    "validation",
    "monitoring",
    "ingestion",
    "aggregation",
}
FOLLOW_UP_LIMIT = 5


class ThreadState(TypedDict):
    thread_id: str
    category: str
    question: str
    follow_up_count: int
    user_messages: list[str]
    assistant_messages: list[str]
    signature: str
    started_at: str


@dataclass
class AssistantReply:
    text: str
    turn_type: str
    tool_calls: list[dict[str, object]]
    context_sources: list[str]


@dataclass
class AssistantReplyChunk:
    delta: str
    full_text: str
    turn_type: str
    tool_calls: list[dict[str, object]]
    context_sources: list[str]
    is_final: bool


def _clean(value: object) -> str:
    return str(value or "").strip()


def _single_line(value: object) -> str:
    return re.sub(r"\s+", " ", _clean(value))


def _keyword_terms(text: str, *, limit: int = 6) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9+#.-]{3,}", text.lower()):
        if token in {"the", "and", "for", "with", "that", "this", "from", "you", "your", "into"}:
            continue
        if token in terms:
            continue
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _question_signature(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", " ", _single_line(text).lower())
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed[:220]


def _format_practice_round_label(round_index: int) -> str:
    today = datetime.utcnow()
    day = today.strftime("%d").lstrip("0") or "0"
    return f"Practice Round {round_index} - {today.strftime('%B')} {day}"


def _coerce_question(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return _single_line(item.get("question_text") or item.get("question"))


def _collect_research_bank(db: Session, job_id: int) -> dict[str, list[str]]:
    session = (
        db.query(InterviewResearchSession)
        .filter(InterviewResearchSession.job_id == job_id, InterviewResearchSession.status == "completed")
        .order_by(InterviewResearchSession.updated_at.desc(), InterviewResearchSession.id.desc())
        .first()
    )
    if not session or not isinstance(session.question_bank, dict):
        return {key: [] for key in QUESTION_ORDER}
    result: dict[str, list[str]] = {}
    for category in QUESTION_ORDER:
        raw = session.question_bank.get(category, [])
        values: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                question = _coerce_question(item)
                if question:
                    values.append(question)
        result[category] = values
    return result


def _extract_job_requirements(job: Job) -> list[str]:
    text = _clean(job.description)
    if not text:
        return []
    chunks: list[str] = []
    for line in text.splitlines():
        cleaned = _single_line(line.strip("-• ").strip())
        if not cleaned:
            continue
        if len(cleaned) <= 220:
            chunks.append(cleaned)
            continue
        for part in re.split(r"[.?!;]\s+", cleaned):
            part_clean = _single_line(part)
            if part_clean:
                chunks.append(part_clean)
    picked: list[str] = []
    picked_lower: set[str] = set()
    for chunk in chunks:
        if not (24 <= len(chunk) <= 170):
            continue
        if len(chunk.split()) > 28:
            continue
        lower = chunk.lower()
        if lower in picked_lower:
            continue
        picked.append(chunk)
        picked_lower.add(lower)
        if len(picked) >= 20:
            break
    if picked:
        return picked
    return [part for part in re.split(r"[.?!]\s+", _single_line(text)) if len(part) >= 24][:12]


def _extract_cv_signals(db: Session) -> list[str]:
    cv = db.query(CV).order_by(CV.created_at.desc(), CV.id.desc()).first()
    if not cv:
        return []
    content = _clean(cv.parsed_text)
    if not content:
        return []
    lines = [line.strip("-• ").strip() for line in content.splitlines() if len(line.strip()) > 12]
    return lines[:24]


def _fallback_question(category: str, *, role: str, company: str, requirements: list[str], cv_signals: list[str], index: int) -> str:
    req = requirements[index % len(requirements)] if requirements else f"{role} fundamentals"
    cv_hint = cv_signals[index % len(cv_signals)] if cv_signals else "a project that best represents your impact"
    if category == "behavioral":
        return (
            f"Tell me about a time you handled {req.lower()}. "
            "Use STAR and include measurable outcomes."
        )
    if category == "technical":
        return (
            f"Walk me through how your experience with '{cv_hint}' helps deliver '{req}' "
            f"in this {role} role."
        )
    if category == "system_design":
        return (
            f"Design a production-ready approach for {req.lower()} at {company}, "
            "including trade-offs, scaling assumptions, and failure handling."
        )
    return f"What should we know about engineering expectations at {company} when delivering {req.lower()}?"


def _is_dev_heavy(job: Job) -> bool:
    corpus = " ".join([_clean(job.title), _clean(job.description)]).lower()
    markers = ("backend", "software", "platform", "api", "developer", "engineering", "full stack", "distributed")
    return any(marker in corpus for marker in markers)


def _derive_limits(job: Job) -> dict[str, int]:
    raw = " ".join([_clean(getattr(job, "seniority", "")), _clean(job.title)]).lower()
    if any(k in raw for k in ("intern", "junior", "jr")):
        return {"min_questions": 6, "target_questions": 10, "max_questions": 16}
    if any(k in raw for k in ("senior", "lead", "staff", "principal", "manager")):
        return {"min_questions": 6, "target_questions": 16, "max_questions": 24}
    return {"min_questions": 6, "target_questions": 12, "max_questions": 20}


def _collect_prior_question_signatures(db: Session, job_id: int) -> list[str]:
    rows = (
        db.query(InterviewChatTurn.content)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(
            InterviewChatSession.job_id == job_id,
            InterviewChatTurn.speaker == "assistant",
            InterviewChatTurn.turn_type == "question",
        )
        .all()
    )
    signatures = {_question_signature(content or "") for (content,) in rows if _clean(content)}
    return sorted(signatures)


def _base_metadata(db: Session, job: Job) -> dict[str, object]:
    return {
        "role": _clean(job.title),
        "company": _clean(job.company),
        "created_from": "job_details",
        "preparation_status": "pending",
        "preparation_started_at": None,
        "preparation_completed_at": None,
        "focus_areas": [],
        "difficulty_level": "baseline",
        "memory_facts": [],
        "contradictions": [],
        "question_history": [],
        "asked_question_signatures": _collect_prior_question_signatures(db, job.id),
        "answer_scores": [],
        "rolling_score": None,
        "weak_streak": 0,
        "limits": _derive_limits(job),
        "thread_follow_up_limit": FOLLOW_UP_LIMIT,
        "thread_score_snapshot": None,
        "active_question_thread": None,
        "category_counts": {category: 0 for category in QUESTION_ORDER},
        "question_pools": {category: [] for category in QUESTION_ORDER},
        "dev_focus": _is_dev_heavy(job),
    }


def create_session(db: Session, job: Job) -> InterviewChatSession:
    existing_rounds = db.query(InterviewChatSession).filter(InterviewChatSession.job_id == job.id).count()
    session = InterviewChatSession(
        session_id=f"ichat_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        job_id=job.id,
        label=_format_practice_round_label(existing_rounds + 1),
        status="active",
        phase="preparing",
        current_question_index=0,
        is_waiting_for_candidate_question=False,
        question_plan=[],
        session_metadata=_base_metadata(db, job),
    )
    db.add(session)
    db.flush()
    return session


def get_or_create_active_session(db: Session, job: Job) -> InterviewChatSession:
    active = (
        db.query(InterviewChatSession)
        .filter(InterviewChatSession.job_id == job.id, InterviewChatSession.status == "active")
        .order_by(InterviewChatSession.updated_at.desc(), InterviewChatSession.id.desc())
        .first()
    )
    if active:
        return active
    return create_session(db, job)


def serialize_turn(turn: InterviewChatTurn) -> dict[str, object]:
    return {
        "id": turn.id,
        "turn_index": int(turn.turn_index),
        "speaker": turn.speaker,
        "turn_type": turn.turn_type,
        "content": turn.content,
        "tool_calls": list(turn.tool_calls or []),
        "context_sources": list(turn.context_sources or []),
        "created_at": str(turn.created_at),
    }


def list_turns(db: Session, session: InterviewChatSession) -> list[InterviewChatTurn]:
    return (
        db.query(InterviewChatTurn)
        .filter(InterviewChatTurn.session_id == session.id)
        .order_by(InterviewChatTurn.turn_index.asc(), InterviewChatTurn.id.asc())
        .all()
    )


def _next_turn_index(db: Session, session: InterviewChatSession) -> int:
    last = (
        db.query(InterviewChatTurn.turn_index)
        .filter(InterviewChatTurn.session_id == session.id)
        .order_by(InterviewChatTurn.turn_index.desc())
        .first()
    )
    return int(last[0]) + 1 if last else 1


def add_turn(
    db: Session,
    *,
    session: InterviewChatSession,
    speaker: str,
    turn_type: str,
    content: str,
    tool_calls: list[dict[str, object]] | None = None,
    context_sources: list[str] | None = None,
) -> InterviewChatTurn:
    row = InterviewChatTurn(
        session_id=session.id,
        turn_index=_next_turn_index(db, session),
        speaker=speaker,
        turn_type=turn_type,
        content=_single_line(content),
        tool_calls=tool_calls or [],
        context_sources=context_sources or [],
    )
    db.add(row)
    db.flush()
    return row


def _is_vague_answer(message: str) -> bool:
    text = _single_line(message).lower()
    words = re.findall(r"[a-z0-9+#.-]+", text)
    word_count = len(words)
    punctuation_count = len(re.findall(r"[.!?;,]", text))
    unique_ratio = (len(set(words)) / word_count) if word_count else 0.0
    has_specificity = any(term in text for term in SPECIFIC_CONTEXT_TERMS)

    if len(text) < 28:
        return True
    if len(_keyword_terms(text, limit=12)) <= 3:
        return True
    if _has_measurable_outcome(text) and word_count >= 20:
        return False
    if not has_specificity and word_count >= 28 and punctuation_count == 0:
        return True
    if not has_specificity and word_count >= 30 and punctuation_count == 0 and len(_keyword_terms(text, limit=10)) < 7:
        return True
    # Detect long, run-on answers that stay generic and repetitive.
    if word_count >= 40 and punctuation_count == 0 and not has_specificity:
        return True
    if word_count >= 36 and unique_ratio < 0.52 and not _has_measurable_outcome(text):
        return True
    return False


def _has_measurable_outcome(message: str) -> bool:
    text = _single_line(message).lower()
    has_number = any(char.isdigit() for char in text)
    if not has_number:
        return False
    if "%" in text:
        return True
    measurable_markers = (
        "x",
        "ms",
        "sec",
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
        "users",
        "requests",
        "tickets",
        "incidents",
        "sprints",
        "releases",
        "points",
    )
    return any(marker in text for marker in measurable_markers)


def _score_text_components(text: str) -> dict[str, float]:
    lowered = _single_line(text).lower()
    words = lowered.split()
    word_count = len(words)
    clarity = 80.0 if word_count >= 60 else 45.0 if word_count < 28 else 62.0
    if any(marker in lowered for marker in ("first", "second", "finally", "because", "therefore")):
        clarity += 8.0
    impact = 78.0 if _has_measurable_outcome(lowered) else 40.0
    depth = 72.0 if any(marker in lowered for marker in ("trade-off", "bottleneck", "latency", "availability", "consistency", "rollback")) else 48.0
    correctness = 74.0 if not _is_vague_answer(lowered) else 38.0
    role_fit = 74.0 if any(marker in lowered for marker in ("team", "stakeholder", "incident", "delivery", "ownership", "customer")) else 56.0
    return {
        "technical_correctness": max(0.0, min(correctness, 100.0)),
        "depth_tradeoffs": max(0.0, min(depth, 100.0)),
        "clarity_structure": max(0.0, min(clarity, 100.0)),
        "impact_metrics": max(0.0, min(impact, 100.0)),
        "role_fit_communication": max(0.0, min(role_fit, 100.0)),
    }


def _weighted_score(components: dict[str, float]) -> float:
    return (
        components["technical_correctness"] * 0.35
        + components["depth_tradeoffs"] * 0.25
        + components["clarity_structure"] * 0.15
        + components["impact_metrics"] * 0.15
        + components["role_fit_communication"] * 0.10
    )


def _build_intermediate_acknowledgement(message: str, *, category: str) -> str:
    terms = _keyword_terms(message, limit=3)
    if terms:
        joined = ", ".join(terms)
        return f"Thanks, that helps clarify your {category.replace('_', ' ')} depth around {joined}."
    return f"Thanks, that helps me understand your {category.replace('_', ' ')} approach."


def _build_interview_feedback(transcript_rows: list[InterviewChatTurn]) -> dict[str, object]:
    user_answers = [row for row in transcript_rows if row.speaker == "user" and row.turn_type == "answer"]
    assistant_follow_ups = [row for row in transcript_rows if row.speaker == "assistant" and row.turn_type == "follow_up"]
    if not user_answers:
        return {
            "overview": "The interview ended before enough candidate responses were captured for a full assessment.",
            "what_went_well": ["You started the session and engaged with the interviewer flow."],
            "what_to_improve": ["Complete more questions so we can evaluate communication, depth, and impact."],
            "next_steps": ["Run another round and answer at least 4-5 questions with concrete examples."],
        }
    measurable_count = 0
    vague_count = 0
    star_like_count = 0
    total_words = 0
    for row in user_answers:
        content = _single_line(row.content)
        total_words += len(content.split())
        if _has_measurable_outcome(content):
            measurable_count += 1
        if _is_vague_answer(content):
            vague_count += 1
        lowered = content.lower()
        if "situation" in lowered or "task" in lowered or "action" in lowered or "result" in lowered:
            star_like_count += 1
    answer_count = len(user_answers)
    avg_words = total_words / answer_count if answer_count else 0.0
    overview = (
        f"You answered {answer_count} question(s) with an average of {int(avg_words)} words per response. "
        f"{measurable_count} answer(s) included measurable outcomes."
    )
    went_well: list[str] = []
    if measurable_count > 0:
        went_well.append("You used quantifiable impact in parts of your answers, which improves credibility.")
    if avg_words >= 90:
        went_well.append("You provided reasonably detailed context instead of one-line answers.")
    if star_like_count > 0:
        went_well.append("You showed structured thinking (STAR-style framing) in at least some responses.")
    if not went_well:
        went_well.append("You maintained engagement and responded consistently across the interview.")
    to_improve: list[str] = []
    if vague_count > 0:
        to_improve.append("Some responses were too broad; add concrete actions, constraints, and your exact role.")
    if measurable_count < max(1, answer_count // 2):
        to_improve.append("Add more metrics (%, latency, throughput, incidents, time/cost impact) in each answer.")
    if avg_words < 60:
        to_improve.append("Increase depth by covering problem context, decision trade-offs, and final outcomes.")
    if not to_improve:
        to_improve.append("Keep tightening each answer to focus on your direct ownership and decision quality.")
    next_steps = [
        "Practice 3 STAR stories with explicit metrics and trade-offs.",
        "For technical answers, include architecture choices and why alternatives were rejected.",
        "Run another round and aim to reduce follow-up clarifications from the interviewer.",
    ]
    if assistant_follow_ups:
        next_steps[2] = f"Run another round and reduce follow-up clarifications (current round had {len(assistant_follow_ups)})."
    return {
        "overview": overview,
        "what_went_well": went_well[:3],
        "what_to_improve": to_improve[:3],
        "next_steps": next_steps[:3],
    }


async def _query_vector_context(db: Session, *, job_id: int, query: str) -> tuple[list[str], dict[str, object]]:
    if not _clean(query):
        return [], {"tool": "query_vector_store", "status": "skipped", "result_count": 0}
    try:
        vector = await generate_embedding(query)
        rows = fetch_context_from_interview_documents(db=db, query_vector=vector, job_id=job_id)
    except Exception as exc:
        return [], {"tool": "query_vector_store", "status": "error", "result_count": 0, "error": str(exc)}
    snippets: list[str] = []
    for row in rows[:2]:
        snippet = _single_line(row.get("snippet", ""))
        if not snippet:
            continue
        source_name = _single_line(row.get("filename", "interview-doc"))
        snippets.append(f"{source_name}: {snippet[:260]}")
    return snippets, {"tool": "query_vector_store", "status": "ok", "result_count": len(snippets)}


def _lookup_cv_details(db: Session, query: str) -> tuple[list[str], dict[str, object]]:
    cv = db.query(CV).order_by(CV.created_at.desc(), CV.id.desc()).first()
    if not cv or not _clean(cv.parsed_text):
        return [], {"tool": "get_cv_detail", "status": "empty", "result_count": 0}
    terms = _keyword_terms(query)
    if not terms:
        return [], {"tool": "get_cv_detail", "status": "skipped", "result_count": 0}
    lines = [_single_line(line) for line in cv.parsed_text.splitlines() if _clean(line)]
    matches = [line for line in lines if any(term in line.lower() for term in terms)][:2]
    return matches, {"tool": "get_cv_detail", "status": "ok", "result_count": len(matches)}


def _lookup_job_details(job: Job, query: str) -> tuple[list[str], dict[str, object]]:
    description = _clean(job.description)
    if not description:
        return [], {"tool": "get_job_detail", "status": "empty", "result_count": 0}
    terms = _keyword_terms(query)
    lines = [_single_line(line) for line in description.splitlines() if _clean(line)]
    matches = [line for line in lines if not terms or any(term in line.lower() for term in terms)][:2]
    return matches, {"tool": "get_job_detail", "status": "ok", "result_count": len(matches)}


async def build_context_tools(db: Session, *, job: Job, query: str) -> tuple[list[str], list[dict[str, object]]]:
    cv_matches, cv_call = _lookup_cv_details(db, query)
    job_matches, job_call = _lookup_job_details(job, query)
    vector_matches, vector_call = await _query_vector_context(db, job_id=job.id, query=query)
    context_sources = [*cv_matches, *job_matches, *vector_matches]
    tool_calls = [cv_call, job_call, vector_call]
    return context_sources[:4], tool_calls


def _opening_line(job: Job) -> str:
    role = _clean(job.title) or "this role"
    company = _clean(job.company) or "this company"
    return f"Hi, I'm your interviewer for the {role} role at {company}. I'll ask one question at a time and keep this realistic."


def _transition_line(previous_category: str, next_category: str) -> str:
    if previous_category == next_category:
        return ""
    transitions = {
        "behavioral": "Let's begin with behavioral context.",
        "technical": "Let's shift into technical depth.",
        "system_design": "Now let's move into system design.",
        "company_specific": "Great. I'll finish with company-specific context.",
        "closing": "Thanks. We'll close with one final question.",
    }
    return transitions.get(next_category, "Let's transition to the next area.")


def _ensure_metadata(session: InterviewChatSession) -> dict[str, Any]:
    metadata = dict(session.session_metadata or {})
    metadata.setdefault("limits", {"min_questions": 6, "target_questions": 12, "max_questions": 20})
    metadata.setdefault("asked_question_signatures", [])
    metadata.setdefault("question_history", [])
    metadata.setdefault("answer_scores", [])
    metadata.setdefault("rolling_score", None)
    metadata.setdefault("weak_streak", 0)
    metadata.setdefault("memory_facts", [])
    metadata.setdefault("contradictions", [])
    metadata.setdefault("focus_areas", [])
    metadata.setdefault("difficulty_level", "baseline")
    metadata.setdefault("thread_follow_up_limit", FOLLOW_UP_LIMIT)
    metadata.setdefault("thread_score_snapshot", None)
    metadata.setdefault("active_question_thread", None)
    metadata.setdefault("category_counts", {category: 0 for category in QUESTION_ORDER})
    metadata.setdefault("question_pools", {category: [] for category in QUESTION_ORDER})
    metadata.setdefault("dev_focus", False)
    metadata.setdefault("preparation_status", "pending")
    return metadata


def _prepare_question_pools(db: Session, job: Job) -> dict[str, list[str]]:
    bank = _collect_research_bank(db, job.id)
    requirements = _extract_job_requirements(job)
    cv_signals = _extract_cv_signals(db)
    role = _clean(job.title) or "this role"
    company = _clean(job.company) or "this company"
    pools: dict[str, list[str]] = {}
    for category in QUESTION_ORDER:
        category_pool = list(bank.get(category, []))
        target = QUESTION_TARGETS[category]
        while len(category_pool) < target + 3:
            category_pool.append(
                _fallback_question(
                    category,
                    role=role,
                    company=company,
                    requirements=requirements,
                    cv_signals=cv_signals,
                    index=len(category_pool),
                )
            )
        pools[category] = [_single_line(item) for item in category_pool if _clean(item)]
    return pools


def _prepare_session_context(db: Session, *, session: InterviewChatSession, job: Job) -> None:
    metadata = _ensure_metadata(session)
    if metadata.get("preparation_status") == "ready":
        session.session_metadata = metadata
        return
    metadata["preparation_status"] = "running"
    metadata["preparation_started_at"] = datetime.utcnow().isoformat()
    metadata["question_pools"] = _prepare_question_pools(db, job)
    metadata["job_requirements"] = _extract_job_requirements(job)[:20]
    metadata["cv_signals"] = _extract_cv_signals(db)[:20]
    focus_areas = []
    if bool(metadata.get("dev_focus")):
        focus_areas = ["system_design", "technical", "coding_drill"]
    else:
        focus_areas = ["behavioral", "technical", "company_specific"]
    metadata["focus_areas"] = focus_areas
    metadata["preparation_status"] = "ready"
    metadata["preparation_completed_at"] = datetime.utcnow().isoformat()
    session.phase = "opening"
    session.session_metadata = metadata


def _current_thread(metadata: dict[str, Any]) -> ThreadState | None:
    raw = metadata.get("active_question_thread")
    if not isinstance(raw, dict):
        return None
    thread_id = _clean(raw.get("thread_id"))
    question = _clean(raw.get("question"))
    category = _clean(raw.get("category")) or "technical"
    if not thread_id or not question:
        return None
    return {
        "thread_id": thread_id,
        "category": category,
        "question": question,
        "follow_up_count": int(raw.get("follow_up_count") or 0),
        "user_messages": [str(x) for x in (raw.get("user_messages") or []) if _clean(x)],
        "assistant_messages": [str(x) for x in (raw.get("assistant_messages") or []) if _clean(x)],
        "signature": _clean(raw.get("signature")) or _question_signature(question),
        "started_at": _clean(raw.get("started_at")) or datetime.utcnow().isoformat(),
    }


def _store_thread(metadata: dict[str, Any], thread: ThreadState | None) -> None:
    metadata["active_question_thread"] = thread


def _choose_category(metadata: dict[str, Any]) -> str:
    counts_raw = metadata.get("category_counts") or {}
    counts: dict[str, int] = {category: int(counts_raw.get(category) or 0) for category in QUESTION_ORDER}
    # Deterministic progression in QUESTION_ORDER; do not reorder by score/count sorting.
    for index, category in enumerate(QUESTION_ORDER[:-1]):
        later_categories = QUESTION_ORDER[index + 1 :]
        if later_categories and counts[category] < min(counts[name] for name in later_categories):
            return category
    min_count = min(counts.values()) if counts else 0
    for category in QUESTION_ORDER:
        if counts[category] == min_count:
            return category
    return QUESTION_ORDER[0]


async def _stream_assistant_reply(reply: AssistantReply) -> AsyncIterator[AssistantReplyChunk]:
    text = reply.text.strip()
    if not text:
        yield AssistantReplyChunk(
            delta="",
            full_text="",
            turn_type=reply.turn_type,
            tool_calls=reply.tool_calls,
            context_sources=reply.context_sources,
            is_final=True,
        )
        return

    pieces = [token for token in text.split(" ") if token]
    if not pieces:
        yield AssistantReplyChunk(
            delta=text,
            full_text=text,
            turn_type=reply.turn_type,
            tool_calls=reply.tool_calls,
            context_sources=reply.context_sources,
            is_final=True,
        )
        return

    assembled = ""
    last_index = len(pieces) - 1
    for index, token in enumerate(pieces):
        delta = f"{token} "
        assembled = f"{assembled}{delta}"
        yield AssistantReplyChunk(
            delta=delta,
            full_text=assembled.rstrip() if index == last_index else assembled,
            turn_type=reply.turn_type,
            tool_calls=reply.tool_calls,
            context_sources=reply.context_sources,
            is_final=index == last_index,
        )
        await asyncio.sleep(0)


async def _next_primary_question(
    db: Session,
    *,
    session: InterviewChatSession,
    job: Job,
    metadata: dict[str, Any],
) -> tuple[str, str, list[dict[str, object]], list[str]]:
    pools_raw = metadata.get("question_pools") or {}
    asked_signatures = {
        str(sig) for sig in cast(list[object], metadata.get("asked_question_signatures") or []) if _clean(sig)
    }
    category = _choose_category(metadata)
    role = _clean(job.title) or "this role"
    company = _clean(job.company) or "this company"
    rolling = metadata.get("rolling_score")
    weak_streak = int(metadata.get("weak_streak") or 0)
    performance_signal = (
        f"rolling_score={rolling if isinstance(rolling, (int, float)) else 'n/a'}, "
        f"weak_streak={weak_streak}, difficulty_level={_clean(metadata.get('difficulty_level')) or 'baseline'}"
    )
    memory_seed = " ".join([str(x) for x in cast(list[object], metadata.get("memory_facts") or [])[:6]])
    seed_query = (
        f"{role} {company} {category} interview question. "
        f"performance: {performance_signal}. "
        f"memory: {memory_seed}"
    )
    context_sources, context_tool_calls = await build_context_tools(db, job=job, query=seed_query)
    llm_tool_call: dict[str, object] = {"tool": "llm_primary_question_generation", "status": "fallback", "result_count": 0}
    llm_candidate = await generate_adaptive_interview_question(
        role=role,
        company=company,
        category=category,
        performance_signal=performance_signal,
        dev_focus=bool(metadata.get("dev_focus")),
        asked_questions=sorted(asked_signatures),
        job_requirements=[str(x) for x in cast(list[object], metadata.get("job_requirements") or []) if _clean(x)],
        cv_signals=[str(x) for x in cast(list[object], metadata.get("cv_signals") or []) if _clean(x)],
        memory_facts=[str(x) for x in cast(list[object], metadata.get("memory_facts") or []) if _clean(x)],
        context_snippets=context_sources,
    )
    llm_question = _single_line(llm_candidate.get("question") or "")
    if llm_question:
        llm_sig = _question_signature(llm_question)
        if llm_sig and llm_sig not in asked_signatures:
            asked_signatures.add(llm_sig)
            metadata["asked_question_signatures"] = sorted(asked_signatures)
            llm_tool_call["status"] = "ok"
            llm_tool_call["result_count"] = 1
            return category, llm_question, [*context_tool_calls, llm_tool_call], context_sources

    pool = [str(item) for item in cast(list[object], pools_raw.get(category, [])) if _clean(item)]
    for candidate in pool:
        sig = _question_signature(candidate)
        if sig in asked_signatures:
            continue
        if bool(metadata.get("dev_focus")) and category == "technical" and "window function" not in candidate.lower():
            if session.current_question_index % 4 == 0:
                candidate = (
                    "Quick coding drill: when would you choose a SQL window function over GROUP BY, "
                    "and how would you explain the trade-off?"
                )
                sig = _question_signature(candidate)
        asked_signatures.add(sig)
        metadata["asked_question_signatures"] = sorted(asked_signatures)
        return category, _single_line(candidate), [*context_tool_calls, llm_tool_call], context_sources
    role = _clean(job.title) or "this role"
    company = _clean(job.company) or "this company"
    requirements = _extract_job_requirements(job)
    cv_signals = _extract_cv_signals(db)
    fallback = ""
    sig = ""
    for attempt in range(6):
        candidate = _fallback_question(
            category,
            role=role,
            company=company,
            requirements=requirements,
            cv_signals=cv_signals,
            index=session.current_question_index + attempt,
        )
        candidate_sig = _question_signature(candidate)
        if candidate_sig not in asked_signatures:
            fallback = candidate
            sig = candidate_sig
            break
    if not fallback:
        fallback = (
            f"{_fallback_question(category, role=role, company=company, requirements=requirements, cv_signals=cv_signals, index=session.current_question_index)} "
            f"(variant {int(session.current_question_index) + 1})"
        )
        sig = _question_signature(fallback)
    asked_signatures.add(sig)
    metadata["asked_question_signatures"] = sorted(asked_signatures)
    return category, _single_line(fallback), [*context_tool_calls, llm_tool_call], context_sources


def _start_new_thread(metadata: dict[str, Any], *, category: str, question: str) -> ThreadState:
    thread: ThreadState = {
        "thread_id": f"thread_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "category": category,
        "question": question,
        "follow_up_count": 0,
        "user_messages": [],
        "assistant_messages": [],
        "signature": _question_signature(question),
        "started_at": datetime.utcnow().isoformat(),
    }
    _store_thread(metadata, thread)
    history = cast(list[dict[str, object]], metadata.get("question_history") or [])
    history.append(
        {
            "thread_id": thread["thread_id"],
            "category": category,
            "question": question,
            "asked_at": thread["started_at"],
            "signature": thread["signature"],
        }
    )
    metadata["question_history"] = history[-120:]
    category_counts = cast(dict[str, object], metadata.get("category_counts") or {})
    category_counts[category] = int(category_counts.get(category) or 0) + 1
    metadata["category_counts"] = category_counts
    return thread


def _append_memory(metadata: dict[str, Any], message: str) -> None:
    memory_facts = [str(item) for item in cast(list[object], metadata.get("memory_facts") or []) if _clean(item)]
    for term in _keyword_terms(message, limit=4):
        fact = f"candidate_mentioned:{term}"
        if fact not in memory_facts:
            memory_facts.append(fact)
    metadata["memory_facts"] = memory_facts[-24:]


def _append_thread_score(metadata: dict[str, Any], *, category: str, question: str, score: float, components: dict[str, float]) -> None:
    answer_scores = cast(list[dict[str, object]], metadata.get("answer_scores") or [])
    answer_scores.append(
        {
            "category": category,
            "question": question[:220],
            "score": round(score, 2),
            "components": {k: round(v, 2) for k, v in components.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    metadata["answer_scores"] = answer_scores[-200:]
    recent_scores = [float(item.get("score", 0)) for item in answer_scores[-3:] if isinstance(item, dict)]
    rolling = sum(recent_scores) / len(recent_scores) if recent_scores else score
    metadata["rolling_score"] = round(rolling, 2)
    weak_streak = int(metadata.get("weak_streak") or 0)
    if score < 45:
        weak_streak += 1
    else:
        weak_streak = 0
    metadata["weak_streak"] = weak_streak
    metadata["thread_score_snapshot"] = {
        "score": round(score, 2),
        "rolling_score": round(rolling, 2),
        "category": category,
        "question": question[:180],
    }
    metadata["difficulty_level"] = "increase" if rolling >= 75 else "reduce" if rolling < 45 else "baseline"


def _should_end_interview(session: InterviewChatSession, metadata: dict[str, Any]) -> bool:
    limits = cast(dict[str, object], metadata.get("limits") or {})
    min_q = int(limits.get("min_questions") or 6)
    max_q = int(limits.get("max_questions") or 20)
    asked_primary = int(session.current_question_index or 0)
    if asked_primary >= max_q:
        return True
    if asked_primary < min_q:
        return False
    weak_streak = int(metadata.get("weak_streak") or 0)
    rolling = float(metadata.get("rolling_score") or 0)
    return weak_streak >= 2 or rolling < 45


def _thread_followup_needed(thread: ThreadState, message: str, *, is_vague: bool | None = None) -> bool:
    if thread["follow_up_count"] >= FOLLOW_UP_LIMIT:
        return False
    if is_vague is None:
        is_vague = _is_vague_answer(message)
    if is_vague:
        return True
    if thread["category"] in {"behavioral", "technical"} and not _has_measurable_outcome(message):
        return True
    if thread["category"] == "system_design":
        lowered = message.lower()
        if not any(token in lowered for token in ("trade-off", "availability", "latency", "throughput", "failure")):
            return True
    return False


def _thread_followup_prompt(thread: ThreadState, message: str, *, is_vague: bool | None = None) -> str:
    if is_vague is None:
        is_vague = _is_vague_answer(message)
    if is_vague:
        if thread["follow_up_count"] >= 1:
            return (
                "Let's make this concrete before we continue: pick one real project and answer in 4 lines - "
                "context, what you personally changed, how you validated it, and one measurable outcome."
            )
        return (
            f"Let's stay on {thread['category'].replace('_', ' ')} for a moment: "
            "give one concrete example with your actions, constraints, and measurable outcome."
        )
    if thread["category"] == "system_design":
        return (
            "Good start. Go deeper on trade-offs: explain failure modes, scaling limits, "
            "and why you chose this architecture over alternatives."
        )
    return (
        "One more detail before we move on: what measurable outcome did you drive "
        "(for example %, latency, throughput, cost, or incidents)?"
    )


async def _classify_answer_clarity(thread: ThreadState, message: str) -> bool:
    normalized = _single_line(message)
    if not normalized:
        return True
    if _is_vague_answer(normalized):
        return True
    if _has_measurable_outcome(normalized):
        return False
    if len(_keyword_terms(normalized, limit=12)) >= 7 and len(re.findall(r"[.!?]", normalized)) >= 2:
        return False
    if len(normalized) > 240:
        return False
    try:
        result = await classify_interview_answer_vagueness(
            message=normalized,
            current_question=thread["question"],
            recent_assistant=thread["assistant_messages"],
            recent_user=thread["user_messages"],
        )
        is_vague = bool(result.get("is_vague"))
        confidence = float(result.get("confidence") or 0.0)
        return is_vague and confidence >= 0.45
    except Exception:
        return False


def _thread_repeat_or_clarify_prompt(thread: ThreadState, message: str) -> str:
    lowered = _single_line(message).lower()
    if any(token in lowered for token in ("rephrase", "clarify", "clearer", "what do you mean")):
        return (
            "Absolutely. Rephrased question: "
            f"{thread['question']} "
            "You can answer step-by-step and I can challenge specifics after."
        )
    return f"Sure — repeating the question: {thread['question']}"


def _post_closing_acknowledgement(message: str) -> str:
    if _clean(message):
        return (
            "Thanks for the follow-up. We've already wrapped this interview session, "
            "so I won't start new questions. You can end this round or start a new one."
        )
    return "This interview round is already wrapped. You can end this session or start a new round."


async def _answer_candidate_question(db: Session, *, session: InterviewChatSession, job: Job, message: str) -> AssistantReply:
    context_sources, tool_calls = await build_context_tools(db, job=job, query=message)
    answer = "Thanks for the question. "
    if context_sources:
        answer += "From the role context, " + context_sources[0].rstrip(".") + "."
    else:
        answer += "Based on the job description, collaboration, technical ownership, and measurable delivery are central expectations."
    session.is_waiting_for_candidate_question = False
    session.phase = "closing"
    session.updated_at = datetime.utcnow()
    return AssistantReply(text=answer, turn_type="follow_up", tool_calls=tool_calls, context_sources=context_sources)


async def produce_assistant_reply(
    db: Session,
    *,
    session: InterviewChatSession,
    job: Job,
    message: str,
) -> AsyncIterator[AssistantReplyChunk]:
    raw_message = _clean(message)
    normalized_message = _single_line(raw_message)
    metadata = _ensure_metadata(session)
    turns = list_turns(db, session)
    if raw_message:
        add_turn(
            db,
            session=session,
            speaker="user",
            turn_type="question" if session.is_waiting_for_candidate_question else "answer",
            content=raw_message,
        )
        _append_memory(metadata, normalized_message or raw_message)
    if session.is_waiting_for_candidate_question:
        reply = await _answer_candidate_question(db, session=session, job=job, message=raw_message)
        async for chunk in _stream_assistant_reply(reply):
            yield chunk
        session.session_metadata = metadata
        return
    if session.phase == "closing":
        # Terminal guard: never reopen primary-question flow after closing Q&A.
        reply = AssistantReply(
            text=_post_closing_acknowledgement(raw_message),
            turn_type="follow_up",
            tool_calls=[],
            context_sources=[],
        )
        async for chunk in _stream_assistant_reply(reply):
            yield chunk
        session.session_metadata = metadata
        session.updated_at = datetime.utcnow()
        return
    if metadata.get("preparation_status") != "ready":
        _prepare_session_context(db, session=session, job=job)
        metadata = _ensure_metadata(session)
    thread = _current_thread(metadata)
    if not turns and not thread:
        opening = _opening_line(job)
        first_category, first_question, first_tool_calls, first_context_sources = await _next_primary_question(
            db, session=session, job=job, metadata=metadata
        )
        _start_new_thread(metadata, category=first_category, question=first_question)
        session.current_question_index = int(session.current_question_index or 0) + 1
        session.phase = first_category
        opening_reply = AssistantReply(text=opening, turn_type="transition", tool_calls=[], context_sources=[])
        question_reply = AssistantReply(
            text=first_question,
            turn_type="question",
            tool_calls=first_tool_calls,
            context_sources=first_context_sources,
        )
        async for chunk in _stream_assistant_reply(opening_reply):
            yield chunk
        async for chunk in _stream_assistant_reply(question_reply):
            yield chunk
        session.session_metadata = metadata
        session.updated_at = datetime.utcnow()
        return
    if not thread:
        next_category, next_question, next_tool_calls, next_context_sources = await _next_primary_question(
            db, session=session, job=job, metadata=metadata
        )
        _start_new_thread(metadata, category=next_category, question=next_question)
        session.current_question_index = int(session.current_question_index or 0) + 1
        session.phase = next_category
        reply = AssistantReply(
            text=next_question,
            turn_type="question",
            tool_calls=next_tool_calls,
            context_sources=next_context_sources,
        )
        async for chunk in _stream_assistant_reply(reply):
            yield chunk
        session.session_metadata = metadata
        session.updated_at = datetime.utcnow()
        return
    if raw_message:
        thread["user_messages"].append(normalized_message or raw_message)
    if raw_message:
        repeat_intent = await classify_interview_repeat_intent(
            message=normalized_message or raw_message,
            current_question=thread["question"],
            recent_assistant=thread["assistant_messages"],
            recent_user=thread["user_messages"],
        )
        if bool(repeat_intent.get("is_repeat_intent")):
            repeat_reply = _thread_repeat_or_clarify_prompt(thread, normalized_message or raw_message)
            thread["assistant_messages"].append(repeat_reply)
            _store_thread(metadata, thread)
            reply = AssistantReply(text=repeat_reply, turn_type="transition", tool_calls=[], context_sources=[])
            async for chunk in _stream_assistant_reply(reply):
                yield chunk
            session.session_metadata = metadata
            session.updated_at = datetime.utcnow()
            return
    needs_follow_up = False
    if raw_message:
        needs_follow_up = await _classify_answer_clarity(thread, normalized_message or raw_message)
    if raw_message and _thread_followup_needed(thread, normalized_message or raw_message, is_vague=needs_follow_up):
        if thread["follow_up_count"] < int(metadata.get("thread_follow_up_limit") or FOLLOW_UP_LIMIT):
            context_sources, tool_calls = await build_context_tools(db, job=job, query=normalized_message or raw_message)
            follow_up = _thread_followup_prompt(thread, normalized_message or raw_message, is_vague=needs_follow_up)
            thread["follow_up_count"] += 1
            thread["assistant_messages"].append(follow_up)
            _store_thread(metadata, thread)
            reply = AssistantReply(text=follow_up, turn_type="follow_up", tool_calls=tool_calls, context_sources=context_sources)
            async for chunk in _stream_assistant_reply(reply):
                yield chunk
            session.session_metadata = metadata
            session.updated_at = datetime.utcnow()
            return
    thread_text = "\n".join(thread["user_messages"])
    components = _score_text_components(thread_text)
    thread_score = _weighted_score(components)
    _append_thread_score(metadata, category=thread["category"], question=thread["question"], score=thread_score, components=components)
    _store_thread(metadata, None)
    transition_reply = AssistantReply(
        text=_build_intermediate_acknowledgement(thread_text or normalized_message or raw_message, category=thread["category"]),
        turn_type="transition",
        tool_calls=[],
        context_sources=[],
    )
    async for chunk in _stream_assistant_reply(transition_reply):
        yield chunk
    if _should_end_interview(session, metadata):
        session.phase = "closing"
        session.is_waiting_for_candidate_question = True
        closing_reply = AssistantReply(
            text="We'll wrap up here. Do you have any questions for me about the role, team, or company?",
            turn_type="question",
            tool_calls=[],
            context_sources=[],
        )
        async for chunk in _stream_assistant_reply(closing_reply):
            yield chunk
        session.session_metadata = metadata
        session.updated_at = datetime.utcnow()
        return
    next_category, next_question, next_tool_calls, next_context_sources = await _next_primary_question(
        db, session=session, job=job, metadata=metadata
    )
    transition = _transition_line(thread["category"], next_category)
    _start_new_thread(metadata, category=next_category, question=next_question)
    session.current_question_index = int(session.current_question_index or 0) + 1
    session.phase = next_category
    if transition:
        transition_line_reply = AssistantReply(text=transition, turn_type="transition", tool_calls=[], context_sources=[])
        async for chunk in _stream_assistant_reply(transition_line_reply):
            yield chunk
    next_question_reply = AssistantReply(
        text=next_question,
        turn_type="question",
        tool_calls=next_tool_calls,
        context_sources=next_context_sources,
    )
    async for chunk in _stream_assistant_reply(next_question_reply):
        yield chunk
    session.session_metadata = metadata
    session.updated_at = datetime.utcnow()
    return


async def end_session_and_trigger_handoff(db: Session, *, session: InterviewChatSession, job: Job) -> dict[str, object]:
    if session.status == "completed":
        metadata = dict(session.session_metadata or {})
        existing_feedback = metadata.get("feedback")
        return {
            "status": "already_completed",
            "handoff_run_id": session.handoff_run_id,
            "feedback": existing_feedback if isinstance(existing_feedback, dict) else None,
        }
    transcript_rows = list_turns(db, session)
    feedback = _build_interview_feedback(transcript_rows)
    transcript_lines = [f"{row.speaker}:{row.turn_type}:{_single_line(row.content)}" for row in transcript_rows]
    transcript_text = "\n".join(transcript_lines)[-7000:]
    cv = db.query(CV).order_by(CV.created_at.desc(), CV.id.desc()).first()
    handoff_run_id: str | None = None
    handoff_status = "not_triggered"
    session.status = "completed"
    session.phase = "completed"
    session.completed_at = datetime.utcnow()
    session.updated_at = datetime.utcnow()
    session.handoff_run_id = handoff_run_id
    db.add(session)
    db.flush()
    if cv and _clean(job.description):
        scoring_context = (
            f"{_clean(job.description)}\n\n"
            "Interview transcript summary (for Story 4 handoff context):\n"
            f"{transcript_text}"
        )
        try:
            result = await execute_scoring_orchestrator(
                db,
                cv_id=cv.id,
                cv_text=cv.parsed_text or "",
                job_title=job.title,
                company=job.company,
                job_description=scoring_context,
                actor="interview-chat.end",
                source="interview_chat",
                idempotency_key=f"interview-chat:{session.session_id}:story4",
            )
            handoff_run_id = result.run_id
            handoff_status = "triggered"
        except Exception as exc:
            handoff_status = "failed"
            logger.exception("Story 4 scoring handoff failed for session=%s: %s", session.session_id, exc)

    session.handoff_run_id = handoff_run_id
    metadata = _ensure_metadata(session)
    metadata["story4_scoring_triggered"] = handoff_status == "triggered"
    metadata["transcript_turn_count"] = len(transcript_rows)
    metadata["feedback"] = feedback
    session.session_metadata = metadata
    db.add(session)
    db.flush()
    return {"status": handoff_status, "handoff_run_id": handoff_run_id, "feedback": feedback}
