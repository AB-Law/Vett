from __future__ import annotations

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.interview_chat import InterviewChatSession, InterviewChatTurn
from ..models.score import Job
from ..services.interview_chat_service import (
    add_turn,
    end_session_and_trigger_handoff,
    get_or_create_active_session,
    list_turns,
    produce_assistant_reply,
    serialize_turn,
)

router = APIRouter(prefix="/interview-chat", tags=["interview-chat"])


class InterviewChatSessionSummary(BaseModel):
    session_id: str
    label: str
    status: str
    phase: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    turn_count: int
    handoff_run_id: str | None = None


class InterviewChatSessionDetail(InterviewChatSessionSummary):
    job_id: int
    feedback: dict[str, object] | None = None
    turns: list[dict[str, object]] = Field(default_factory=list)


class InterviewChatCreateResponse(BaseModel):
    session: InterviewChatSessionDetail


class InterviewChatStreamRequest(BaseModel):
    message: str | None = None


class InterviewChatEndResponse(BaseModel):
    session_id: str
    status: str
    handoff_status: str
    handoff_run_id: str | None = None
    feedback: dict[str, object] | None = None


class InterviewChatDeleteResponse(BaseModel):
    session_id: str
    status: str


def _serialize_session(db: Session, session: InterviewChatSession, include_turns: bool) -> InterviewChatSessionDetail:
    turns = list_turns(db, session)
    metadata = dict(session.session_metadata or {})
    feedback = metadata.get("feedback")
    return InterviewChatSessionDetail(
        session_id=session.session_id,
        label=session.label,
        status=session.status,
        phase=session.phase,
        created_at=str(session.created_at),
        updated_at=str(session.updated_at),
        completed_at=str(session.completed_at) if session.completed_at else None,
        turn_count=len(turns),
        job_id=session.job_id,
        handoff_run_id=session.handoff_run_id,
        feedback=feedback if isinstance(feedback, dict) else None,
        turns=[serialize_turn(turn) for turn in turns] if include_turns else [],
    )


def _load_job_or_404(db: Session, job_id: int) -> Job:
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _load_session_or_404(db: Session, job_id: int, session_id: str) -> InterviewChatSession:
    row = (
        db.query(InterviewChatSession)
        .filter(InterviewChatSession.job_id == job_id, InterviewChatSession.session_id == session_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return row


@router.get("/jobs/{job_id}/sessions", response_model=list[InterviewChatSessionSummary])
def list_interview_sessions(job_id: int, db: Session = Depends(get_db)) -> list[InterviewChatSessionSummary]:
    _load_job_or_404(db, job_id)
    rows = (
        db.query(InterviewChatSession)
        .filter(InterviewChatSession.job_id == job_id)
        .order_by(InterviewChatSession.updated_at.desc(), InterviewChatSession.id.desc())
        .all()
    )
    response: list[InterviewChatSessionSummary] = []
    for row in rows:
        turn_count = db.query(InterviewChatTurn).filter(InterviewChatTurn.session_id == row.id).count()
        response.append(
            InterviewChatSessionSummary(
                session_id=row.session_id,
                label=row.label,
                status=row.status,
                phase=row.phase,
                created_at=str(row.created_at),
                updated_at=str(row.updated_at),
                completed_at=str(row.completed_at) if row.completed_at else None,
                turn_count=turn_count,
                handoff_run_id=row.handoff_run_id,
            )
        )
    return response


@router.post("/jobs/{job_id}/sessions", response_model=InterviewChatCreateResponse)
def create_or_resume_interview_session(job_id: int, db: Session = Depends(get_db)) -> InterviewChatCreateResponse:
    job = _load_job_or_404(db, job_id)
    session = get_or_create_active_session(db, job)
    db.commit()
    db.refresh(session)
    return InterviewChatCreateResponse(session=_serialize_session(db, session, include_turns=True))


@router.get("/jobs/{job_id}/sessions/{session_id}", response_model=InterviewChatSessionDetail)
def get_interview_session(job_id: int, session_id: str, db: Session = Depends(get_db)) -> InterviewChatSessionDetail:
    row = _load_session_or_404(db, job_id, session_id)
    return _serialize_session(db, row, include_turns=True)


@router.post("/jobs/{job_id}/sessions/{session_id}/stream")
async def stream_interview_turn(
    job_id: int,
    session_id: str,
    payload: InterviewChatStreamRequest,
    db: Session = Depends(get_db),
):
    job = _load_job_or_404(db, job_id)
    session = _load_session_or_404(db, job_id, session_id)
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Interview session is not active")

    replies = await produce_assistant_reply(
        db,
        session=session,
        job=job,
        message=payload.message or "",
    )
    emitted_messages: list[str] = []
    emitted_turn_types: list[str] = []
    emitted_tool_calls: list[dict[str, object]] = []
    emitted_context_sources: list[str] = []

    for reply in replies:
        add_turn(
            db,
            session=session,
            speaker="assistant",
            turn_type=reply.turn_type,
            content=reply.text,
            tool_calls=reply.tool_calls,
            context_sources=reply.context_sources,
        )
        emitted_messages.append(reply.text)
        emitted_turn_types.append(reply.turn_type)
        emitted_tool_calls.extend(reply.tool_calls)
        emitted_context_sources.extend(reply.context_sources)

    session.updated_at = datetime.utcnow()
    db.add(session)
    db.commit()

    full_message = "\n\n".join([msg for msg in emitted_messages if msg.strip()]).strip()

    async def event_generator():
        if full_message:
            for token in full_message.split(" "):
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'delta': f'{token} '})}\n\n"
                await asyncio.sleep(0)
        done_payload = {
            "type": "done",
            "session_id": session.session_id,
            "message": full_message,
            "turn_types": emitted_turn_types,
            "tool_calls": emitted_tool_calls,
            "context_sources": emitted_context_sources,
            "phase": session.phase,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/sessions/{session_id}/end", response_model=InterviewChatEndResponse)
async def end_interview_session(job_id: int, session_id: str, db: Session = Depends(get_db)) -> InterviewChatEndResponse:
    job = _load_job_or_404(db, job_id)
    session = _load_session_or_404(db, job_id, session_id)
    handoff = await end_session_and_trigger_handoff(db, session=session, job=job)
    db.commit()
    return InterviewChatEndResponse(
        session_id=session.session_id,
        status=session.status,
        handoff_status=str(handoff.get("status", "skipped")),
        handoff_run_id=handoff.get("handoff_run_id") if isinstance(handoff.get("handoff_run_id"), str) else None,
        feedback=handoff.get("feedback") if isinstance(handoff.get("feedback"), dict) else None,
    )


@router.delete("/jobs/{job_id}/sessions/{session_id}", response_model=InterviewChatDeleteResponse)
def delete_interview_session(job_id: int, session_id: str, db: Session = Depends(get_db)) -> InterviewChatDeleteResponse:
    _load_job_or_404(db, job_id)
    session = _load_session_or_404(db, job_id, session_id)
    db.query(InterviewChatTurn).filter(InterviewChatTurn.session_id == session.id).delete(synchronize_session=False)
    db.delete(session)
    db.commit()
    return InterviewChatDeleteResponse(session_id=session_id, status="deleted")
