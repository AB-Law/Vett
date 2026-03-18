import json
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.interview_research import InterviewResearchSession
from app.models.score import Job
from app.routers import jobs as jobs_router
from app.services.interview_research import InterviewResearchQuestion, InterviewResearchQuestionBank, InterviewResearchResult


def _new_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _parse_sse_data(payload: bytes | str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    text_payload = payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
    for block in text_payload.split("\n\n"):
        text = block.strip()
        if not text.startswith("data: "):
            continue
        events.append(json.loads(text.removeprefix("data: ")))
    return events


def test_docker_compose_exposes_searxng_internally_only():
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    content = compose_path.read_text()
    lines = content.splitlines()
    in_service = False
    service_indent = 0
    service_lines: list[str] = []
    for line in lines:
        if re.match(r"^\s*searxng:\s*$", line):
            in_service = True
            service_indent = len(line) - len(line.lstrip())
            service_lines = []
            continue
        if in_service:
            if not line.strip():
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= service_indent:
                break
            service_lines.append(line)
    service_block = "\n".join(service_lines)
    assert "ports:" not in service_block
    assert "expose:" in service_block and "8080" in service_block


@pytest.mark.asyncio
async def test_stream_interview_research_persists_session_and_returns_fetchable_payload(monkeypatch):
    db = _new_db_session()
    job = Job(title="Backend Engineer", company="Acme")
    db.add(job)
    db.commit()
    db.refresh(job)

    bank = InterviewResearchQuestionBank(
        behavioral=[
            InterviewResearchQuestion(
                question="How do you handle ambiguous requirements?",
                tool="interview_documents",
                query="behavioral",
                source_url="vector://interview-doc/doc-1",
                source_title="Interview notes",
                timestamp="2026-01-01T00:00:00Z",
                snippet="Behavioral evidence",
                confidence_score=0.8,
            )
        ],
        technical=[
            InterviewResearchQuestion(
                question="Explain eventual consistency in distributed systems.",
                tool="interview_documents",
                query="technical",
                source_url="vector://interview-doc/doc-2",
                source_title="Interview notes",
                timestamp="2026-01-01T00:00:00Z",
                snippet="Technical evidence",
                confidence_score=0.8,
            )
        ],
        system_design=[
            InterviewResearchQuestion(
                question="Design a high-throughput job queue.",
                tool="interview_documents",
                query="design",
                source_url="vector://interview-doc/doc-3",
                source_title="Interview notes",
                timestamp="2026-01-01T00:00:00Z",
                snippet="System design evidence",
                confidence_score=0.7,
            )
        ],
        company_specific=[],
        source_urls=[
            "local://fallback",
            "https://example.com/culture",
        ],
    )

    async def fake_run_interview_research(_, context):
        await context.emit({
            "type": "status",
            "stage": "search_interview_questions",
            "message": "ok",
            "role": context.role,
            "company": context.company,
        })
        await context.emit({
            "type": "status",
            "stage": "search_role_skills",
            "message": "ok",
            "role": context.role,
            "company": context.company,
        })
        await context.emit({
            "type": "status",
            "stage": "search_company_engineering_culture",
            "message": "ok",
            "role": context.role,
            "company": context.company,
        })
        await context.emit({
            "type": "status",
            "stage": "fetch_page",
            "message": "ok",
            "role": context.role,
            "company": context.company,
        })
        await context.emit({
            "type": "status",
            "stage": "query_vector_store",
            "message": "ok",
            "role": context.role,
            "company": context.company,
        })
        return InterviewResearchResult(
            session_id="",
            role=context.role,
            company=context.company,
            status="completed",
            question_bank=bank,
            fallback_used=False,
            message="Research pipeline completed.",
            metadata={
                "total_questions": len(bank.all_questions()),
                "research_log": [],
                "tools_used": ["query_vector_store"],
            },
        )

    monkeypatch.setattr(jobs_router, "run_interview_research", fake_run_interview_research)

    stream_response = await jobs_router.stream_interview_research(job.id, db=db)
    event_payloads: list[dict[str, object]] = []
    async for chunk in stream_response.body_iterator:
        event_payloads.extend(_parse_sse_data(chunk))

    stages = [event.get("stage") for event in event_payloads if event.get("type") == "status"]
    done_event = next((event for event in event_payloads if event.get("type") == "done"), None)
    done_payload = done_event.get("payload", {}) if done_event else None
    assert done_payload is not None
    assert isinstance(done_payload.get("metadata", {}).get("research_log"), list)
    assert "tools_used" in done_payload.get("metadata", {})
    assert "search_interview_questions" in stages
    assert "search_role_skills" in stages
    assert "search_company_engineering_culture" in stages
    assert "fetch_page" in stages
    assert "query_vector_store" in stages
    assert done_event.get("stage") == "finalizing"

    session_id = str(done_payload["session_id"])
    session = jobs_router.get_interview_research_session(job.id, session_id, db=db)
    assert session.status == "completed"
    assert session.session_id == session_id
    assert len(session.question_bank.behavioral) == 1
    assert len(session.question_bank.technical) == 1
    assert len(session.question_bank.system_design) == 1
    assert "local://fallback" in session.source_urls
    assert done_payload.get("question_count") == len(bank.all_questions())

    record = db.query(InterviewResearchSession).filter(InterviewResearchSession.session_id == session_id).first()
    assert record is not None
