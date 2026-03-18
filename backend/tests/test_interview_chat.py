import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.cv import CV
from app.models.interview_chat import InterviewChatSession, InterviewChatTurn
from app.models.interview_research import InterviewResearchSession
from app.models.score import Job
from app.routers import interview_chat as interview_chat_router
from app.services import interview_chat_service


def _new_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _parse_sse(payload: bytes | str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    text_payload = payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
    for block in text_payload.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line.removeprefix("data: ")))
    return events


@pytest.mark.asyncio
async def test_interview_chat_opening_and_category_order():
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="Acme", description="Build resilient APIs and distributed systems.")
    db.add(job)
    db.commit()
    db.refresh(job)

    db.add(
        InterviewResearchSession(
            session_id="research_1",
            job_id=job.id,
            role=job.title,
            company=job.company,
            status="completed",
            stage="finalizing",
            question_bank={
                "behavioral": [{"question": "Tell me about a conflict you resolved on a team."}],
                "technical": [{"question": "How would you optimize a high-traffic REST API?"}],
                "system_design": [{"question": "Design a rate-limited notification service."}],
                "company_specific": [{"question": "How would you align with Acme's product quality bar?"}],
                "source_urls": [],
            },
            source_urls=[],
            fallback_used=False,
        )
    )
    db.commit()

    create_result = interview_chat_router.create_or_resume_interview_session(job.id, db=db)
    assert create_result.session.job_id == job.id

    stream_response = await interview_chat_router.stream_interview_turn(
        job.id,
        create_result.session.session_id,
        interview_chat_router.InterviewChatStreamRequest(message=None),
        db=db,
    )
    stream_events: list[dict[str, object]] = []
    async for chunk in stream_response.body_iterator:
        stream_events.extend(_parse_sse(chunk))
    done_event = next(event for event in stream_events if event.get("type") == "done")
    assert "Backend Engineer" in str(done_event.get("message"))
    assert "Acme" in str(done_event.get("message"))
    assert "Tell me about a conflict you resolved on a team." in str(done_event.get("message"))

    detail = interview_chat_router.get_interview_session(job.id, create_result.session.session_id, db=db)
    assistant_turns = [turn for turn in detail.turns if turn["speaker"] == "assistant"]
    assert assistant_turns[0]["turn_type"] == "transition"
    assert assistant_turns[1]["turn_type"] == "question"

    await interview_chat_router.stream_interview_turn(
        job.id,
        create_result.session.session_id,
        interview_chat_router.InterviewChatStreamRequest(message="I clarified goals, delegated owners, and reduced incident count by 40%."),
        db=db,
    )
    session_row = (
        db.query(InterviewChatSession)
        .filter(InterviewChatSession.session_id == create_result.session.session_id)
        .first()
    )
    assert session_row is not None
    assert session_row.phase == "behavioral"


@pytest.mark.asyncio
async def test_interview_chat_vague_answer_triggers_follow_up():
    db = _new_db_session()
    job = Job(title="Data Engineer", company="Northwind", description="Own ETL pipelines and quality controls.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await interview_chat_router.stream_interview_turn(
        job.id,
        session.session_id,
        interview_chat_router.InterviewChatStreamRequest(message=None),
        db=db,
    )

    await interview_chat_router.stream_interview_turn(
        job.id,
        session.session_id,
        interview_chat_router.InterviewChatStreamRequest(message="Not sure, maybe."),
        db=db,
    )

    turns = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(InterviewChatSession.session_id == session.session_id, InterviewChatTurn.speaker == "assistant")
        .order_by(InterviewChatTurn.turn_index.desc())
        .all()
    )
    assert turns
    assert turns[0].turn_type == "follow_up"


@pytest.mark.asyncio
async def test_end_interview_triggers_story4_handoff(monkeypatch):
    db = _new_db_session()
    job = Job(title="Platform Engineer", company="Waynetics", description="Lead platform reliability work.")
    cv = CV(filename="resume.pdf", file_size=100, file_type="pdf", parsed_text="Built platform systems and led incident response.")
    db.add(job)
    db.add(cv)
    db.commit()
    db.refresh(job)

    async def fake_execute_scoring_orchestrator(*_args, **_kwargs):
        class Result:
            run_id = "run_story4_123"

        return Result()

    monkeypatch.setattr(interview_chat_service, "execute_scoring_orchestrator", fake_execute_scoring_orchestrator)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await interview_chat_router.stream_interview_turn(
        job.id,
        session.session_id,
        interview_chat_router.InterviewChatStreamRequest(message=None),
        db=db,
    )

    result = await interview_chat_router.end_interview_session(job.id, session.session_id, db=db)
    assert result.status == "completed"
    assert result.handoff_status == "triggered"
    assert result.handoff_run_id == "run_story4_123"
    assert isinstance(result.feedback, dict)
    assert "overview" in result.feedback


def test_delete_interview_session_removes_transcript_rows():
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="Acme", description="Build APIs.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    session_row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    assert session_row is not None
    db.add(
        InterviewChatTurn(
            session_id=session_row.id,
            turn_index=1,
            speaker="assistant",
            turn_type="question",
            content="Tell me about an API you built.",
            tool_calls=[],
            context_sources=[],
        )
    )
    db.commit()

    delete_result = interview_chat_router.delete_interview_session(job.id, session.session_id, db=db)
    assert delete_result.status == "deleted"

    remaining_session = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    remaining_turns = db.query(InterviewChatTurn).all()
    assert remaining_session is None
    assert remaining_turns == []
