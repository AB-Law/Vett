from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from sqlalchemy.orm import Session

from ..models.cv import CV
from ..models.interview_chat import InterviewChatSession, InterviewChatTurn
from ..models.interview_research import InterviewResearchSession
from ..models.score import Job
from .interview_docs import fetch_context_from_interview_documents
from .llm import generate_embedding
from .scoring_orchestrator import execute_scoring_orchestrator

QUESTION_ORDER = ["behavioral", "technical", "system_design", "company_specific"]
QUESTION_TARGETS = {
    "behavioral": 3,
    "technical": 4,
    "system_design": 2,
    "company_specific": 2,
}
VAGUE_MARKERS = {
    "not sure",
    "don't know",
    "idk",
    "maybe",
    "kind of",
    "sort of",
    "i think",
    "probably",
}


@dataclass
class AssistantReply:
    text: str
    turn_type: str
    tool_calls: list[dict[str, object]]
    context_sources: list[str]


def _clean(value: object) -> str:
    return str(value or "").strip()


def _single_line(value: object) -> str:
    return re.sub(r"\s+", " ", _clean(value))


def _keyword_terms(text: str, *, limit: int = 6) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9+#.-]{3,}", text.lower()):
        if token in {"the", "and", "for", "with", "that", "this", "from", "you", "your"}:
            continue
        if token in terms:
            continue
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _format_practice_round_label(round_index: int) -> str:
    today = datetime.utcnow()
    day = today.strftime("%d").lstrip("0") or "0"
    return f"Practice Round {round_index} — {today.strftime('%B')} {day}"


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
        # Keep requirement-like snippets and drop giant JD blobs that produce unusable prompts.
        if not (24 <= len(chunk) <= 170):
            continue
        if len(chunk.split()) > 26:
            continue
        lower = chunk.lower()
        if lower in picked_lower:
            continue
        picked.append(chunk)
        picked_lower.add(lower)
        if len(picked) >= 16:
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
            "Please structure your answer in STAR format and include measurable outcomes."
        )
    if category == "technical":
        return (
            f"Walk me through how you would apply your experience with '{cv_hint}' "
            f"to deliver '{req}' in this {role} role."
        )
    if category == "system_design":
        return (
            f"Design a production-ready approach for {req.lower()} at {company}, "
            "including trade-offs, scaling assumptions, and failure handling."
        )
    return (
        f"What should we know about working style and engineering expectations at {company} "
        f"when delivering {req.lower()}?"
    )


def build_question_plan(db: Session, job: Job) -> list[dict[str, str]]:
    research_bank = _collect_research_bank(db, job.id)
    requirements = _extract_job_requirements(job)
    cv_signals = _extract_cv_signals(db)

    plan: list[dict[str, str]] = []
    role = _clean(job.title) or "this role"
    company = _clean(job.company) or "this company"
    for category in QUESTION_ORDER:
        existing = research_bank.get(category, [])
        target = QUESTION_TARGETS[category]
        selected = existing[:target]
        while len(selected) < target:
            selected.append(
                _fallback_question(
                    category,
                    role=role,
                    company=company,
                    requirements=requirements,
                    cv_signals=cv_signals,
                    index=len(selected),
                )
            )
        for question_text in selected:
            plan.append({"category": category, "question": _single_line(question_text), "source": "deepagent_bank" if question_text in existing else "fallback"})
    return plan


def create_session(db: Session, job: Job) -> InterviewChatSession:
    existing_rounds = db.query(InterviewChatSession).filter(InterviewChatSession.job_id == job.id).count()
    session = InterviewChatSession(
        session_id=f"ichat_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        job_id=job.id,
        label=_format_practice_round_label(existing_rounds + 1),
        status="active",
        phase="opening",
        current_question_index=0,
        is_waiting_for_candidate_question=False,
        question_plan=build_question_plan(db, job),
        session_metadata={
            "role": _clean(job.title),
            "company": _clean(job.company),
            "created_from": "job_details",
        },
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
    if len(text) < 28:
        return True
    if len(_keyword_terms(text, limit=12)) <= 3:
        return True
    return any(marker in text for marker in VAGUE_MARKERS)


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


def _build_intermediate_acknowledgement(message: str, *, category: str) -> str:
    terms = _keyword_terms(message, limit=3)
    if terms:
        joined = ", ".join(terms)
        return f"Thanks, that gives good signal on your {category.replace('_', ' ')} depth, especially around {joined}."
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
    return (
        f"Hi, I'm your interviewer for the {role} role at {company}. "
        "I'll ask one question at a time and keep this realistic."
    )


def _question_text(session: InterviewChatSession) -> tuple[str, str]:
    plan = session.question_plan if isinstance(session.question_plan, list) else []
    if session.current_question_index >= len(plan):
        return "closing", "Do you have any questions for me about the role, team, or company?"
    row = plan[session.current_question_index]
    if not isinstance(row, dict):
        return "technical", "Walk me through how you would approach this role's key technical challenges."
    return _clean(row.get("category")) or "technical", _single_line(row.get("question"))


def _transition_line(previous_category: str, next_category: str) -> str:
    if previous_category == next_category:
        return ""
    transitions = {
        "technical": "Let's shift into technical depth.",
        "system_design": "Now let's move into system design.",
        "company_specific": "Great. I'll finish with company-specific context.",
        "closing": "Thanks. We'll close with one final question.",
    }
    return transitions.get(next_category, "Let's transition to the next area.")


async def produce_assistant_reply(
    db: Session,
    *,
    session: InterviewChatSession,
    job: Job,
    message: str,
) -> list[AssistantReply]:
    safe_message = _single_line(message)
    replies: list[AssistantReply] = []

    turns = list_turns(db, session)
    last_assistant_turn_type = next((turn.turn_type for turn in reversed(turns) if turn.speaker == "assistant"), "")
    if not turns and not safe_message:
        category, question = _question_text(session)
        opening = _opening_line(job)
        replies.append(AssistantReply(text=opening, turn_type="transition", tool_calls=[], context_sources=[]))
        replies.append(AssistantReply(text=question, turn_type="question", tool_calls=[], context_sources=[]))
        session.phase = category
        session.updated_at = datetime.utcnow()
        return replies

    if safe_message:
        add_turn(db, session=session, speaker="user", turn_type="answer", content=safe_message)

    if session.is_waiting_for_candidate_question:
        context_sources, tool_calls = await build_context_tools(db, job=job, query=safe_message)
        answer = "Thanks for the question. "
        if context_sources:
            answer += "From the role context, " + context_sources[0].rstrip(".") + "."
        else:
            answer += (
                "Based on the job description, collaboration, technical ownership, and measurable delivery "
                "are central expectations."
            )
        session.is_waiting_for_candidate_question = False
        session.phase = "closing"
        session.updated_at = datetime.utcnow()
        replies.append(AssistantReply(text=answer, turn_type="follow_up", tool_calls=tool_calls, context_sources=context_sources))
        return replies

    if _is_vague_answer(safe_message) and last_assistant_turn_type != "follow_up":
        category, _ = _question_text(session)
        context_sources, tool_calls = await build_context_tools(db, job=job, query=safe_message)
        probe = (
            f"Before we move on from {category.replace('_', ' ')}, "
            "give one concrete example with your actions, constraints, and measurable outcome."
        )
        replies.append(AssistantReply(text=probe, turn_type="follow_up", tool_calls=tool_calls, context_sources=context_sources))
        session.updated_at = datetime.utcnow()
        return replies

    current_category, _ = _question_text(session)
    if (
        safe_message
        and current_category in {"behavioral", "technical"}
        and not _has_measurable_outcome(safe_message)
        and last_assistant_turn_type != "follow_up"
    ):
        category, _ = _question_text(session)
        context_sources, tool_calls = await build_context_tools(db, job=job, query=safe_message)
        probe = (
            f"Thanks. One quick follow-up before we move on from {category.replace('_', ' ')}: "
            "what measurable outcome did you drive (for example %, latency, throughput, cost, or incidents)?"
        )
        replies.append(AssistantReply(text=probe, turn_type="follow_up", tool_calls=tool_calls, context_sources=context_sources))
        session.updated_at = datetime.utcnow()
        return replies

    previous_category, _ = _question_text(session)
    if safe_message:
        replies.append(
            AssistantReply(
                text=_build_intermediate_acknowledgement(safe_message, category=previous_category),
                turn_type="transition",
                tool_calls=[],
                context_sources=[],
            )
        )
    session.current_question_index += 1
    next_category, next_question = _question_text(session)
    if next_category == "closing":
        session.phase = "closing"
        session.is_waiting_for_candidate_question = True
        transition = _transition_line(previous_category, "closing")
        if transition:
            replies.append(AssistantReply(text=transition, turn_type="transition", tool_calls=[], context_sources=[]))
        replies.append(AssistantReply(text=next_question, turn_type="question", tool_calls=[], context_sources=[]))
        session.updated_at = datetime.utcnow()
        return replies

    session.phase = next_category
    transition = _transition_line(previous_category, next_category)
    if transition:
        replies.append(AssistantReply(text=transition, turn_type="transition", tool_calls=[], context_sources=[]))
    replies.append(AssistantReply(text=next_question, turn_type="question", tool_calls=[], context_sources=[]))
    session.updated_at = datetime.utcnow()
    return replies


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
    handoff_status = "skipped"
    if cv and _clean(job.description):
        scoring_context = (
            f"{_clean(job.description)}\n\n"
            "Interview transcript summary (for Story 4 handoff context):\n"
            f"{transcript_text}"
        )
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

    session.status = "completed"
    session.phase = "completed"
    session.completed_at = datetime.utcnow()
    session.updated_at = datetime.utcnow()
    session.handoff_run_id = handoff_run_id
    metadata = dict(session.session_metadata or {})
    metadata["story4_scoring_triggered"] = handoff_status == "triggered"
    metadata["transcript_turn_count"] = len(transcript_rows)
    metadata["feedback"] = feedback
    session.session_metadata = metadata
    db.add(session)
    db.flush()
    return {"status": handoff_status, "handoff_run_id": handoff_run_id, "feedback": feedback}
