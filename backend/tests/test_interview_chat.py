import json
from io import BytesIO

import pytest
from fastapi import UploadFile
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from backend.app.database import Base
    from backend.app.models.cv import CV
    from backend.app.models.interview_chat import InterviewChatSession, InterviewChatTurn
    from backend.app.models.interview_research import InterviewResearchSession
    from backend.app.models.score import Job
    from backend.app.routers import interview_chat as interview_chat_router
    from backend.app.services import interview_chat_service
except ModuleNotFoundError:
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


async def _stream_turn(db, job_id: int, session_id: str, message: str | None) -> list[dict[str, object]]:
    response = await interview_chat_router.stream_interview_turn(
        job_id,
        session_id,
        interview_chat_router.InterviewChatStreamRequest(message=message),
        db=db,
    )
    events: list[dict[str, object]] = []
    async for chunk in response.body_iterator:
        events.extend(_parse_sse(chunk))
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

    stream_events = await _stream_turn(db, job.id, create_result.session.session_id, None)
    done_event = next(event for event in stream_events if event.get("type") == "done")
    assert "Backend Engineer" in str(done_event.get("message"))
    assert "Acme" in str(done_event.get("message"))
    assert "Tell me about a conflict you resolved on a team." in str(done_event.get("message"))
    assert done_event.get("preparation_status") == "ready"

    detail = interview_chat_router.get_interview_session(job.id, create_result.session.session_id, db=db)
    assistant_turns = [turn for turn in detail.turns if turn["speaker"] == "assistant"]
    assert assistant_turns[0]["turn_type"] == "transition"
    assert assistant_turns[1]["turn_type"] == "question"
    assert detail.primary_question_count == 1
    assert detail.preparation_status == "ready"

    await _stream_turn(
        db,
        job.id,
        create_result.session.session_id,
        "I clarified goals, delegated owners, and reduced incident count by 40%.",
    )
    session_row = (
        db.query(InterviewChatSession)
        .filter(InterviewChatSession.session_id == create_result.session.session_id)
        .first()
    )
    assert session_row is not None
    assert session_row.phase in {"system_design", "behavioral", "technical", "company_specific"}


@pytest.mark.asyncio
async def test_interview_chat_vague_answer_triggers_follow_up():
    db = _new_db_session()
    job = Job(title="Data Engineer", company="Northwind", description="Own ETL pipelines and quality controls.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, session.session_id, None)

    await _stream_turn(db, job.id, session.session_id, "Not sure, maybe.")

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
async def test_interview_chat_run_on_generic_answer_triggers_follow_up():
    db = _new_db_session()
    job = Job(title="ML Engineer", company="City Greens", description="Build AI features with strong reliability.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, session.session_id, None)
    await _stream_turn(
        db,
        job.id,
        session.session_id,
        (
            "hey so my approach is i would first take a look at data and then maybe try to look into all sources "
            "and kind of compare things and then probably have planning and then another agent checks and gives "
            "feedback and i think that should work"
        ),
    )

    last_assistant_turn = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(InterviewChatSession.session_id == session.session_id, InterviewChatTurn.speaker == "assistant")
        .order_by(InterviewChatTurn.turn_index.desc())
        .first()
    )
    assert last_assistant_turn is not None
    assert last_assistant_turn.turn_type == "follow_up"
    assert "concrete example" in (last_assistant_turn.content or "").lower()


@pytest.mark.asyncio
async def test_interview_chat_escalates_follow_up_after_repeated_vague_answers():
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="ProbeCo", description="Build resilient APIs.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, session.session_id, None)
    await _stream_turn(db, job.id, session.session_id, "Not sure, maybe we just optimize a bit.")
    await _stream_turn(db, job.id, session.session_id, "i think we can kind of improve things probably")

    assistant_followups = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(
            InterviewChatSession.session_id == session.session_id,
            InterviewChatTurn.speaker == "assistant",
            InterviewChatTurn.turn_type == "follow_up",
        )
        .order_by(InterviewChatTurn.turn_index.asc())
        .all()
    )
    assert len(assistant_followups) >= 2
    assert "pick one real project" in (assistant_followups[-1].content or "").lower()


@pytest.mark.asyncio
async def test_thread_guardrail_does_not_increment_primary_question_count_until_thread_closes():
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="GuardrailCo", description="Build and scale APIs.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, session.session_id, None)

    for _ in range(5):
        await _stream_turn(db, job.id, session.session_id, "Not sure, maybe.")
        row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
        assert row is not None
        assert row.current_question_index == 1

    await _stream_turn(db, job.id, session.session_id, "Not sure, maybe.")
    row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    assert row is not None
    assert row.current_question_index == 2


@pytest.mark.asyncio
async def test_thread_scoring_aggregates_whole_conversation():
    db = _new_db_session()
    job = Job(title="Data Engineer", company="ScoringCo", description="Own ETL quality and platform reliability.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, session.session_id, None)
    await _stream_turn(
        db,
        job.id,
        session.session_id,
        "I led the migration, reduced latency by 32%, and handled rollback trade-offs with incident metrics.",
    )

    row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    assert row is not None
    metadata = dict(row.session_metadata or {})
    assert isinstance(metadata.get("thread_score_snapshot"), dict)
    answer_scores = metadata.get("answer_scores")
    assert isinstance(answer_scores, list)
    assert len(answer_scores) >= 1


@pytest.mark.asyncio
async def test_post_closing_message_does_not_reopen_primary_question_flow(monkeypatch):
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="CloseGuard", description="Build resilient backend systems.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, session.session_id, None)

    monkeypatch.setattr(interview_chat_service, "_should_end_interview", lambda *_args, **_kwargs: True)
    await _stream_turn(
        db,
        job.id,
        session.session_id,
        "I led an incident review and reduced recurring failures by 35% through ownership and metrics.",
    )

    row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    assert row is not None
    assert row.phase == "closing"
    assert row.is_waiting_for_candidate_question is True
    assert row.current_question_index == 1

    await _stream_turn(
        db,
        job.id,
        session.session_id,
        "How does the team measure success for this role after onboarding?",
    )
    row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    assert row is not None
    assert row.phase == "closing"
    assert row.is_waiting_for_candidate_question is False
    assert row.current_question_index == 1

    question_turn_count_before_extra = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(
            InterviewChatSession.session_id == session.session_id,
            InterviewChatTurn.speaker == "assistant",
            InterviewChatTurn.turn_type == "question",
        )
        .count()
    )

    await _stream_turn(db, job.id, session.session_id, "One more thing, thanks.")

    row = db.query(InterviewChatSession).filter(InterviewChatSession.session_id == session.session_id).first()
    assert row is not None
    assert row.phase == "closing"
    assert row.current_question_index == 1
    metadata = dict(row.session_metadata or {})
    assert metadata.get("active_question_thread") is None

    question_turn_count_after_extra = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(
            InterviewChatSession.session_id == session.session_id,
            InterviewChatTurn.speaker == "assistant",
            InterviewChatTurn.turn_type == "question",
        )
        .count()
    )
    assert question_turn_count_after_extra == question_turn_count_before_extra

    last_assistant_turn = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(InterviewChatSession.session_id == session.session_id, InterviewChatTurn.speaker == "assistant")
        .order_by(InterviewChatTurn.turn_index.desc())
        .first()
    )
    assert last_assistant_turn is not None
    assert last_assistant_turn.turn_type == "follow_up"


@pytest.mark.asyncio
async def test_cross_session_question_dedup_avoids_repeat():
    db = _new_db_session()
    job = Job(title="Platform Engineer", company="NoRepeat", description="Own distributed backend systems.")
    db.add(job)
    db.commit()
    db.refresh(job)

    first_session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, first_session.session_id, None)
    first_question = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(
            InterviewChatSession.session_id == first_session.session_id,
            InterviewChatTurn.speaker == "assistant",
            InterviewChatTurn.turn_type == "question",
        )
        .order_by(InterviewChatTurn.turn_index.asc())
        .first()
    )
    assert first_question is not None

    await interview_chat_router.end_interview_session(job.id, first_session.session_id, db=db)

    second_session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session
    await _stream_turn(db, job.id, second_session.session_id, None)
    second_question = (
        db.query(InterviewChatTurn)
        .join(InterviewChatSession, InterviewChatSession.id == InterviewChatTurn.session_id)
        .filter(
            InterviewChatSession.session_id == second_session.session_id,
            InterviewChatTurn.speaker == "assistant",
            InterviewChatTurn.turn_type == "question",
        )
        .order_by(InterviewChatTurn.turn_index.asc())
        .first()
    )
    assert second_question is not None
    assert first_question.content != second_question.content


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
    await _stream_turn(db, job.id, session.session_id, None)

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


@pytest.mark.asyncio
async def test_transcribe_interview_audio_success(monkeypatch):
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="Acme", description="Build APIs.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session

    def _fake_transcribe(**_kwargs):
        return "I improved API latency by 30 percent.", 210

    monkeypatch.setattr(interview_chat_router, "transcribe_audio_bytes", _fake_transcribe)

    audio_file = UploadFile(filename="input.webm", file=BytesIO(b"dummy-audio"))
    result = await interview_chat_router.transcribe_interview_audio(job.id, session.session_id, audio_file=audio_file, db=db)

    assert result.transcript == "I improved API latency by 30 percent."
    assert result.latency_ms == 210


@pytest.mark.asyncio
async def test_transcribe_interview_audio_no_speech_maps_to_422(monkeypatch):
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="Acme", description="Build APIs.")
    db.add(job)
    db.commit()
    db.refresh(job)

    session = interview_chat_router.create_or_resume_interview_session(job.id, db=db).session

    def _fake_transcribe(**_kwargs):
        raise interview_chat_router.SttNoSpeechError("No speech detected in recording")

    monkeypatch.setattr(interview_chat_router, "transcribe_audio_bytes", _fake_transcribe)

    audio_file = UploadFile(filename="input.webm", file=BytesIO(b"dummy-audio"))
    with pytest.raises(HTTPException) as exc_info:
        await interview_chat_router.transcribe_interview_audio(job.id, session.session_id, audio_file=audio_file, db=db)
    assert exc_info.value.status_code == 422
    assert "No speech detected" in str(exc_info.value.detail)
