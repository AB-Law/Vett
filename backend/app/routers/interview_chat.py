from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
from ..services.stt_service import (
    SttError,
    SttNoSpeechError,
    SttUnavailableError,
    transcribe_audio_bytes,
)

router = APIRouter(prefix="/interview-chat", tags=["interview-chat"])
logger = logging.getLogger(__name__)


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
    preparation_status: str | None = None
    rolling_score: float | None = None
    limits: dict[str, int] | None = None
    primary_question_count: int = 0


class InterviewChatSessionDetail(InterviewChatSessionSummary):
    job_id: int
    feedback: dict[str, object] | None = None
    thread_score_snapshot: dict[str, object] | None = None
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


class InterviewChatTranscriptionResponse(BaseModel):
    transcript: str
    latency_ms: int


def _serialize_session(db: Session, session: InterviewChatSession, include_turns: bool) -> InterviewChatSessionDetail:
    turns = list_turns(db, session)
    metadata = dict(session.session_metadata or {})
    feedback = metadata.get("feedback")
    limits = metadata.get("limits") if isinstance(metadata.get("limits"), dict) else None
    raw_rolling = metadata.get("rolling_score")
    rolling_score = float(raw_rolling) if isinstance(raw_rolling, (float, int)) else None
    preparation_status = _normalize_preparation_status(metadata)
    thread_score_snapshot = metadata.get("thread_score_snapshot") if isinstance(metadata.get("thread_score_snapshot"), dict) else None
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
        preparation_status=preparation_status,
        rolling_score=rolling_score,
        limits=limits,
        primary_question_count=int(session.current_question_index or 0),
        feedback=feedback if isinstance(feedback, dict) else None,
        thread_score_snapshot=thread_score_snapshot,
        turns=[serialize_turn(turn) for turn in turns] if include_turns else [],
    )


def _normalize_preparation_status(metadata: dict[str, object]) -> str | None:
    raw = metadata.get("preparation_status")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


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
                preparation_status=_normalize_preparation_status(dict(row.session_metadata or {})),
                rolling_score=float((row.session_metadata or {}).get("rolling_score"))
                if isinstance((row.session_metadata or {}).get("rolling_score"), (float, int))
                else None,
                limits=(row.session_metadata or {}).get("limits")
                if isinstance((row.session_metadata or {}).get("limits"), dict)
                else None,
                primary_question_count=int(row.current_question_index or 0),
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

    queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

    async def produce_events() -> None:
        emitted_messages: list[str] = []
        emitted_turn_types: list[str] = []
        emitted_tool_calls: list[dict[str, object]] = []
        emitted_context_sources: list[str] = []
        try:
            async for chunk in produce_assistant_reply(
                db,
                session=session,
                job=job,
                message=payload.message or "",
            ):
                if chunk.delta:
                    await queue.put({"type": "token", "delta": chunk.delta})
                if not chunk.is_final:
                    continue

                completed_message = chunk.full_text.strip()
                if completed_message:
                    add_turn(
                        db,
                        session=session,
                        speaker="assistant",
                        turn_type=chunk.turn_type,
                        content=completed_message,
                        tool_calls=chunk.tool_calls,
                        context_sources=chunk.context_sources,
                    )
                    emitted_messages.append(completed_message)
                    emitted_turn_types.append(chunk.turn_type)
                    emitted_tool_calls.extend(chunk.tool_calls)
                    emitted_context_sources.extend(chunk.context_sources)

            session.updated_at = datetime.utcnow()
            db.add(session)
            db.commit()
        except Exception as exc:
            db.rollback()
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            full_message = "\n\n".join([msg for msg in emitted_messages if msg.strip()]).strip()
            done_payload = {
                "type": "done",
                "session_id": session.session_id,
                "message": full_message,
                "turn_types": emitted_turn_types,
                "tool_calls": emitted_tool_calls,
                "context_sources": emitted_context_sources,
                "phase": session.phase,
                "preparation_status": (session.session_metadata or {}).get("preparation_status"),
                "rolling_score": (session.session_metadata or {}).get("rolling_score"),
                "thread_score_snapshot": (session.session_metadata or {}).get("thread_score_snapshot"),
                "primary_question_count": int(session.current_question_index or 0),
                "limits": (session.session_metadata or {}).get("limits"),
            }
            await queue.put(done_payload)
            await queue.put(None)

    producer_task = asyncio.create_task(produce_events())

    async def event_generator():
        while True:
            payload_item = await queue.get()
            if payload_item is None:
                break
            yield f"data: {json.dumps(payload_item)}\n\n"
            await asyncio.sleep(0)
        await producer_task

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


@router.post("/jobs/{job_id}/sessions/{session_id}/transcribe", response_model=InterviewChatTranscriptionResponse)
async def transcribe_interview_audio(
    job_id: int,
    session_id: str,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InterviewChatTranscriptionResponse:
    _load_job_or_404(db, job_id)
    session = _load_session_or_404(db, job_id, session_id)
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Interview session is not active")

    request_started = perf_counter()
    logger.info(
        "stt.transcription_started session_id=%s filename=%s content_type=%s",
        session.session_id,
        audio_file.filename or "",
        audio_file.content_type or "",
    )

    try:
        payload = await audio_file.read()
        suffix = ".webm"
        if audio_file.filename and "." in audio_file.filename:
            suffix = f".{audio_file.filename.rsplit('.', 1)[-1]}"
        transcript, stt_latency_ms = transcribe_audio_bytes(audio_bytes=payload, suffix=suffix, language="en")
        total_latency_ms = int((perf_counter() - request_started) * 1000)
        logger.info(
            "stt.transcription_success session_id=%s transcript_len=%s stt_latency_ms=%s total_latency_ms=%s",
            session.session_id,
            len(transcript),
            stt_latency_ms,
            total_latency_ms,
        )
        return InterviewChatTranscriptionResponse(transcript=transcript, latency_ms=stt_latency_ms)
    except SttNoSpeechError as exc:
        total_latency_ms = int((perf_counter() - request_started) * 1000)
        logger.warning(
            "stt.transcription_no_speech session_id=%s total_latency_ms=%s error=%s",
            session.session_id,
            total_latency_ms,
            str(exc),
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SttUnavailableError as exc:
        logger.error("stt.transcription_unavailable session_id=%s error=%s", session.session_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SttError as exc:
        total_latency_ms = int((perf_counter() - request_started) * 1000)
        logger.exception(
            "stt.transcription_failed session_id=%s total_latency_ms=%s error=%s",
            session.session_id,
            total_latency_ms,
            str(exc),
        )
        raise HTTPException(status_code=500, detail="Audio transcription failed") from exc
