"""Practice question retrieval and follow-up generation APIs."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.practice import PracticeGeneration, PracticeQuestion, PracticeSession, PracticeSessionQuestion
from ..models.score import Job
from ..services import llm as llm_service
from ..services.practice_vector import pick_best_candidate
from ..services.practice_sync import (
    get_questions_for_company,
    resolve_company_slug,
    normalize_company_slug,
    sync_all,
)

router = APIRouter(prefix="/practice", tags=["practice"])


class PracticeSyncRequest(BaseModel):
    company_slug: str | None = None
    preferred_window: str | None = Field(default=None, description="Prefer this window filename")


class PracticeSyncResponse(BaseModel):
    commit: str
    inserted: int
    updated: int
    retired: int
    companies: list[dict[str, int | str]]


class PracticeQuestionResponse(BaseModel):
    id: int
    title: str
    url: str | None = None
    difficulty: str | None = None
    acceptance: str | None = None
    frequency: str | None = None
    source_file: str | None = None
    source_window: str | None = None
    prompt: str | None = None
    is_ai_generated: bool = False
    is_solved: bool = False

    model_config = ConfigDict(from_attributes=True)


class PracticeQuestionsResponse(BaseModel):
    session_id: str
    company_slug: str
    questions: list[PracticeQuestionResponse]


class MarkSolvedRequest(BaseModel):
    session_id: str
    question_id: int


class UnmarkSolvedRequest(BaseModel):
    session_id: str
    question_id: int


class MarkSolvedResponse(BaseModel):
    session_id: str
    question_id: int
    status: str


class DiscardQuestionRequest(BaseModel):
    session_id: str
    question_id: int


class DiscardQuestionResponse(BaseModel):
    session_id: str
    question_id: int
    status: str


class PracticeNextRequest(BaseModel):
    session_id: str
    solved_question_id: int
    difficulty_delta: int | None = Field(default=None, ge=-2, le=2)
    language: str | None = None
    technique: str | None = None
    complexity: str | None = None
    time_pressure_minutes: int | None = None
    pattern: str | None = None


class ConstraintMetadata(BaseModel):
    difficulty_delta: int | None = None
    language: str | None = None
    technique: str | None = None
    complexity: str | None = None
    time_pressure_minutes: int | None = None
    pattern: str | None = None


class PracticeNextResponse(BaseModel):
    base_question_id: int
    base_question_link: str
    transformed_prompt: str
    constraint_metadata: ConstraintMetadata
    reason: str
    next_question: PracticeQuestionResponse | None = None


class PracticeChatMessage(BaseModel):
    role: str
    content: str


class PracticeInterviewChatRequest(BaseModel):
    session_id: str
    question_id: int
    message: str
    language: str | None = None
    interview_history: list[PracticeChatMessage] = Field(default_factory=list)
    solution_text: str | None = None


class PracticeInterviewChatResponse(BaseModel):
    session_id: str
    question_id: int
    interviewer_reply: str


class PracticeSessionQuestionResponse(BaseModel):
    session_id: str
    company_slug: str
    question_ids: list[int]
    status: str


def _get_or_create_session(db: Session, job_id: int | None, company_slug: str, session_id: str | None) -> PracticeSession:
    if session_id:
        session = (
            db.query(PracticeSession)
            .filter(PracticeSession.session_id == session_id, PracticeSession.is_active.is_(True))
            .first()
        )
        if session:
            if session.company_slug != company_slug:
                session.is_active = False
            else:
                return session

    resolved_slug = normalize_company_slug(company_slug)
    session = PracticeSession(
        session_id=uuid4().hex,
        job_id=job_id,
        company_slug=resolved_slug,
        is_active=True,
        last_constraint=None,
    )
    db.add(session)
    db.flush()
    return session


def _build_followup_question(source_question: PracticeQuestion) -> PracticeQuestion:
    q = PracticeQuestion(
        title=source_question.title,
        url=source_question.url,
        difficulty=source_question.difficulty,
        acceptance=source_question.acceptance,
        is_active=True,
        source_commit=getattr(source_question, 'source_commit', None),
    )
    q.frequency = None
    q.source_window = "generated"
    return q


def _normalize_interview_messages(messages: list[PracticeChatMessage] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not messages:
        return normalized
    for message in messages[-12:]:
        role = message.role.strip().lower()
        if role not in {"user", "assistant", "interviewer"}:
            role = "user"
        content = message.content.strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _coerce_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _coerce_non_empty_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


@router.post("/sync", response_model=PracticeSyncResponse)
def sync_questions(payload: PracticeSyncRequest, db: Session = Depends(get_db)) -> PracticeSyncResponse:
    try:
        result = sync_all(db, payload.company_slug, payload.preferred_window)
        db.commit()
        return PracticeSyncResponse(
            commit=str(result["commit"]),
            inserted=int(result["inserted"]),
            updated=int(result["updated"]),
            retired=int(result["retired"]),
            companies=result["companies"],  # type: ignore[arg-type]
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/company/{company_slug}/questions", response_model=PracticeQuestionsResponse)
def list_company_questions(
    company_slug: str,
    job_id: int,
    db: Session = Depends(get_db),
    session_id: str | None = None,
    limit: int = 8,
    difficulty: str | None = None,
    source_window: str | None = "all",
    include_solved: bool = False,
) -> PracticeQuestionsResponse:
    resolved_company = resolve_company_slug(company_slug, db=db)
    session = _get_or_create_session(db, job_id=job_id, company_slug=resolved_company, session_id=session_id)

    limit_value = max(1, int(limit))
    exclude_statuses = ["discarded"] if include_solved else ["solved", "discarded"]

    try:
        # If no questions loaded yet, sync this company once.
        questions = get_questions_for_company(
            db=db,
            company_slug=resolved_company,
            job_session_id=session.id,
            exclude_statuses=exclude_statuses,
            limit=limit_value,
            difficulty=difficulty,
            source_window=source_window,
            recent_window_minutes=get_settings().practice_restrict_dup_window_minutes,
        )
        if not questions:
            try:
                sync_all(db, company=resolved_company, preferred_window=source_window)
            except RuntimeError:
                # Missing company folders / renamed repos should not fail the whole page.
                # Surface an empty set instead so callers can show a graceful fallback.
                db.rollback()
                return PracticeQuestionsResponse(
                    session_id=session.session_id,
                    company_slug=resolved_company,
                    questions=[],
                )
            questions = get_questions_for_company(
                db=db,
                company_slug=resolved_company,
                job_session_id=session.id,
                exclude_statuses=exclude_statuses,
                limit=limit_value,
                difficulty=difficulty,
                source_window=source_window,
                recent_window_minutes=get_settings().practice_restrict_dup_window_minutes,
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))

    session_question_rows = {
        row.question_id: row.status
        for row in db.query(PracticeSessionQuestion.question_id, PracticeSessionQuestion.status)
        .filter(PracticeSessionQuestion.practice_session_id == session.id)
        .all()
    }

    return PracticeQuestionsResponse(
        session_id=session.session_id,
        company_slug=resolved_company,
        questions=[
            PracticeQuestionResponse(
                id=q.id,
                title=q.title,
                url=q.url,
                difficulty=q.difficulty,
                acceptance=q.acceptance,
                frequency=q.frequency,
                source_file=getattr(q, "source_file", q.source_window),
                source_window=q.source_window,
                is_ai_generated=session_question_rows.get(q.id) == "ai-generated",
                is_solved=session_question_rows.get(q.id) == "solved",
                prompt=None,
            )
            for q in questions
        ],
    )


@router.post("/mark-solved", response_model=MarkSolvedResponse)
def mark_solved(payload: MarkSolvedRequest, db: Session = Depends(get_db)) -> MarkSolvedResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == payload.question_id,
            )
        )
        .first()
    )
    if row is None:
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=question.id,
                status="solved",
                asked_at=datetime.utcnow(),
                solved_at=datetime.utcnow(),
            )
        )
    else:
        row.status = "solved"
        row.asked_at = datetime.utcnow()
        row.solved_at = datetime.utcnow()

    session.last_constraint = None
    session.updated_at = datetime.utcnow()
    db.commit()
    return MarkSolvedResponse(session_id=session.session_id, question_id=question.id, status="solved")


@router.post("/unmark-solved", response_model=MarkSolvedResponse)
def unmark_solved(payload: UnmarkSolvedRequest, db: Session = Depends(get_db)) -> MarkSolvedResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == payload.question_id,
            )
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=400, detail="Question is not marked as solved in this session")
    if row.status == "discarded":
        raise HTTPException(status_code=400, detail="Discarded questions cannot be restored here")

    row.status = "seen"
    row.solved_at = None
    row.asked_at = datetime.utcnow()
    session.updated_at = datetime.utcnow()
    db.commit()
    return MarkSolvedResponse(session_id=session.session_id, question_id=question.id, status="unsolved")


@router.post("/discard", response_model=DiscardQuestionResponse)
def discard_question(payload: DiscardQuestionRequest, db: Session = Depends(get_db)) -> DiscardQuestionResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == question.id,
            )
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=400, detail="Only generated questions can be discarded from this session")
    if row.status == "solved":
        raise HTTPException(status_code=400, detail="Solved questions cannot be discarded")
    if row.status != "ai-generated":
        raise HTTPException(status_code=400, detail="Only AI-generated questions can be discarded from this session")

    row.status = "discarded"
    row.asked_at = datetime.utcnow()

    db.commit()
    return DiscardQuestionResponse(
        session_id=session.session_id,
        question_id=question.id,
        status="discarded",
    )

@router.post("/next", response_model=PracticeNextResponse)
@router.post("/next/", response_model=PracticeNextResponse)
async def next_follow_up(payload: PracticeNextRequest, db: Session = Depends(get_db)) -> PracticeNextResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    base_question = (
        db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.solved_question_id).first()
    )
    if not base_question:
        raise HTTPException(status_code=404, detail="Solved question not found")
    if base_question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Solved question does not match session company")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == base_question.id,
            )
        )
        .first()
    )
    if row is None:
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=base_question.id,
                status="seen",
                asked_at=datetime.utcnow(),
            )
        )
    else:
        if row.status != "solved":
            row.status = "seen"
        row.asked_at = datetime.utcnow()
        if row.status == "solved":
            row.solved_at = datetime.utcnow()

    payload_constraints = {
        "difficulty_delta": payload.difficulty_delta,
        "language": payload.language,
        "technique": payload.technique,
        "complexity": payload.complexity,
        "time_pressure_minutes": payload.time_pressure_minutes,
        "pattern": payload.pattern,
    }

    settings = get_settings()
    candidate_limit = max(settings.practice_default_limit, 40)
    pool = get_questions_for_company(
        db=db,
        company_slug=session.company_slug,
        job_session_id=session.id,
        exclude_statuses=["solved", "discarded", "ai-generated", "seen"],
        limit=candidate_limit,
        difficulty=None,
        source_window=None,
        recent_window_minutes=settings.practice_restrict_dup_window_minutes,
    )
    candidate_list = [question for question in pool if question.id != base_question.id]
    if not candidate_list:
        # Try one final retrieval without the duplicate window to avoid dead-ends.
        fallback_pool = get_questions_for_company(
            db=db,
            company_slug=session.company_slug,
            job_session_id=None,
            exclude_statuses=["solved", "discarded", "ai-generated", "seen"],
            limit=max(settings.practice_default_limit, 80),
            difficulty=None,
            source_window=None,
            recent_window_minutes=0,
        )
        candidate_list = [question for question in fallback_pool if question.id != base_question.id]

    if not candidate_list:
        llm_result = await llm_service.generate_constrained_followup(
            title=base_question.title,
            url=base_question.url or "",
            company=session.company_slug,
            difficulty=base_question.difficulty,
            acceptance=base_question.acceptance,
            frequency=base_question.frequency,
            constraints=payload_constraints,
        )
        llm_model = str(llm_result.get("llm_model") or "unknown")
        llm_provider = str(llm_result.get("llm_provider") or get_settings().active_llm_provider)
        transformed_prompt = _coerce_non_empty_string(llm_result.get("transformed_prompt"))
        if not transformed_prompt:
            transformed_prompt = (
                "Take this base problem and apply the requested constraints to design a follow-up variant with similar skills."
            )
        base_link = str(llm_result.get("base_question_link") or base_question.url or "")
        reason = _coerce_non_empty_string(llm_result.get("reason")) or "No additional questions available for this session."

        generated_question = _build_followup_question(source_question=base_question)
        generated_question.title = f"{base_question.title} — Follow-up"
        db.add(generated_question)
        db.flush()
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=generated_question.id,
                status="ai-generated",
                asked_at=datetime.utcnow(),
            )
        )
        db.add(
            PracticeGeneration(
                practice_session_id=session.id,
                source_question_id=base_question.id,
                generated_text=transformed_prompt,
                constraint_type="custom",
                applied_constraints=payload_constraints,
                reason=reason,
                base_question_link=base_question.url,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        )
        session.last_constraint = payload_constraints
        session.updated_at = datetime.utcnow()
        db.commit()

        return PracticeNextResponse(
            base_question_id=base_question.id,
            base_question_link=base_link,
            transformed_prompt=transformed_prompt,
            constraint_metadata=ConstraintMetadata(**payload_constraints),
            reason=reason,
            next_question=PracticeQuestionResponse(
                id=generated_question.id,
                title=generated_question.title,
                url=generated_question.url,
                difficulty=generated_question.difficulty,
                acceptance=generated_question.acceptance,
                frequency=generated_question.frequency,
                source_file=getattr(generated_question, "source_file", generated_question.source_window),
                source_window=generated_question.source_window,
                prompt=transformed_prompt,
                is_ai_generated=True,
            ),
        )

    job_description: str | None = None
    if session.job_id:
        job = db.query(Job).filter(Job.id == session.job_id).first()
        if job and job.description:
            job_description = job.description

    candidate, _ = await pick_best_candidate(
        db=db,
        base_question=base_question,
        candidate_questions=candidate_list,
        difficulty_delta=payload.difficulty_delta,
        job_description=job_description,
        language=payload.language,
        technique=payload.technique,
        complexity=payload.complexity,
        time_pressure_minutes=payload.time_pressure_minutes,
        pattern=payload.pattern,
    )
    if candidate is None:
        candidate = candidate_list[0]

    generated_question = _build_followup_question(source_question=candidate)
    db.add(generated_question)
    db.flush()
    db.add(
        PracticeSessionQuestion(
            practice_session_id=session.id,
            question_id=generated_question.id,
            status="ai-generated",
            asked_at=datetime.utcnow(),
        )
    )
    llm_result = await llm_service.generate_constrained_followup(
        title=base_question.title,
        url=base_question.url or "",
        company=session.company_slug,
        difficulty=base_question.difficulty,
        acceptance=base_question.acceptance,
        frequency=base_question.frequency,
        constraints=payload_constraints,
    )
    llm_model = str(llm_result.get("llm_model") or "unknown")
    llm_provider = str(llm_result.get("llm_provider") or get_settings().active_llm_provider)

    db.add(
        PracticeGeneration(
            practice_session_id=session.id,
            source_question_id=base_question.id,
            generated_text=llm_result["transformed_prompt"],
            constraint_type="custom",
            applied_constraints=payload_constraints,
            reason=llm_result.get("reason"),
            base_question_link=base_question.url,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    )
    session.last_constraint = payload_constraints
    session.updated_at = datetime.utcnow()
    db.commit()

    return PracticeNextResponse(
        base_question_id=base_question.id,
        base_question_link=str(llm_result.get("base_question_link") or base_question.url or ""),
        transformed_prompt=str(llm_result.get("transformed_prompt", "")),
        constraint_metadata=ConstraintMetadata(**payload_constraints),
        reason=str(llm_result.get("reason", "Generated from solved question")),
        next_question=PracticeQuestionResponse(
            id=generated_question.id,
            title=generated_question.title,
            url=generated_question.url,
            difficulty=generated_question.difficulty,
            acceptance=generated_question.acceptance,
            frequency=generated_question.frequency,
                source_file=getattr(generated_question, "source_file", generated_question.source_window),
            source_window=generated_question.source_window,
            prompt=str(llm_result.get("transformed_prompt", "")) or None,
            is_ai_generated=True,
        ),
    )


@router.post("/interview-chat", response_model=PracticeInterviewChatResponse)
async def interview_chat(payload: PracticeInterviewChatRequest, db: Session = Depends(get_db)) -> PracticeInterviewChatResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Question does not match session company")

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = _normalize_interview_messages(payload.interview_history)
    reply = await llm_service.simulate_interviewer_chat(
        title=question.title,
        url=question.url or "",
        company=session.company_slug,
        difficulty=question.difficulty,
        acceptance=question.acceptance,
        frequency=question.frequency,
        language=payload.language,
        message=message,
        conversation=history,
        solution_text=payload.solution_text,
        db=db,
        job_id=session.job_id,
    )

    return PracticeInterviewChatResponse(session_id=session.session_id, question_id=question.id, interviewer_reply=reply)


@router.get("/sessions/{session_id}", response_model=PracticeSessionQuestionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)) -> PracticeSessionQuestionResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question_ids = [
        qid
        for (qid,) in db.query(PracticeSessionQuestion.question_id)
        .filter(PracticeSessionQuestion.practice_session_id == session.id)
        .all()
    ]
    return PracticeSessionQuestionResponse(
        session_id=session.session_id,
        company_slug=session.company_slug,
        question_ids=question_ids,
        status="active",
    )
