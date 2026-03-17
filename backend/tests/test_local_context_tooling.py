import pytest
import sys
from types import SimpleNamespace
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.cv import CV
from app.models.score import Job, LocalContextToolCallAudit
from app.services.local_context_tooling import (
    ToolAuthorizationError,
    ToolExecutionContext,
    ToolSchemaError,
    FetchRecentHistoryInput,
    execute_from_prompt,
    execute_tool,
    _fetch_recent_history,
)


def _new_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _baseline_context(**overrides):
    defaults = {
        "request_uuid": "req-0000",
        "model_id": "test-model",
        "actor_role": "admin",
        "request_source": "api",
        "environment": "dev",
        "session_id": None,
        "user_id": "user-1",
    }
    defaults.update(overrides)
    return ToolExecutionContext(**defaults)


def test_execute_tool_rejects_unauthorized_role():
    db = _new_session()
    context = _baseline_context(actor_role="agent", environment="prod", request_source="api")

    with pytest.raises(ToolAuthorizationError):
        execute_tool(
            db=db,
            context=context,
            tool_name="get_company_profile",
            arguments={"domain": "example.com"},
        )


def test_execute_tool_validates_input_schema():
    db = _new_session()
    context = _baseline_context(actor_role="admin", environment="dev")

    with pytest.raises(ToolSchemaError):
        execute_tool(
            db=db,
            context=context,
            tool_name="get_jd_similarity",
            arguments={
                "job_description_id": "1",
                "candidate_profile_id": "1",
                "top_k": 99,
            },
        )


def test_execute_tool_deterministic_output_and_audit():
    db = _new_session()
    context = _baseline_context(actor_role="admin", environment="dev")

    db.add(CV(id=1, filename="candidate.pdf", parsed_text="python backend microservices"))
    db.add(Job(id=1, title="Senior Backend", description="Need python backend skills"))
    db.commit()

    first = execute_tool(
        db=db,
        context=context,
        tool_name="get_jd_similarity",
        arguments={
            "job_description_id": "1",
            "candidate_profile_id": "1",
            "top_k": 4,
            "include_explanation": False,
        },
    )
    second = execute_tool(
        db=db,
        context=context,
        tool_name="get_jd_similarity",
        arguments={
            "job_description_id": "1",
            "candidate_profile_id": "1",
            "top_k": 4,
            "include_explanation": False,
        },
    )

    assert first.input_hash == second.input_hash
    assert first.result_hash == second.result_hash
    assert first.decision_rationale == "Executed get_jd_similarity via allowlisted registry"

    audit_count = db.query(LocalContextToolCallAudit).count()
    assert audit_count == 2

    last_audit = db.query(LocalContextToolCallAudit).order_by(
        LocalContextToolCallAudit.id.desc()
    ).first()
    assert last_audit is not None
    assert last_audit.tool_name == "get_jd_similarity"
    assert last_audit.status == "allowed"
    assert isinstance(first.output.get("matches"), list)
    assert len(first.output["matches"]) >= 1


def test_execute_tool_records_company_profile_with_owner():
    db = _new_session()
    context = _baseline_context(actor_role="agent", environment="dev")
    db.add(
        Job(
            id=1,
            company="Acme Corp",
            title="Engineer",
            description="Example role",
            source="test",
            company_website="https://acme.example",
            company_employees_count=123,
            industries="Software",
            company_address={"city": "Boston"},
        )
    )
    db.commit()

    result = execute_tool(
        db=db,
        context=context,
        tool_name="get_company_profile",
        arguments={"domain": "https://acme.example", "include_owners": True},
    )

    assert result.output["domain"] == "acme.example"
    assert result.output["profile"]["name"] == "Acme Corp"
    assert result.output["owners"] is not None
    assert result.output["owners"][0]["role"] in {
        "Hiring Lead",
        "Recruiter",
        "Engineering Manager",
    }


def test_execute_from_prompt_parses_tool_payload():
    db = _new_session()
    context = _baseline_context(actor_role="admin", environment="staging")

    prompt = (
        "Use this decisioning hint and run: "
        '{\"tool\":\"suggest_next_constraint\",\"arguments\":{\"case_id\":\"case-1\",'
        '"candidate_state\":{\"score\":0.7},"prior_steps":["parsed resume","called similarity"],'
        '"risk_level":"medium"}} and then continue.'
    )

    result = execute_from_prompt(db=db, context=context, prompt=prompt)

    assert result.tool_name == "suggest_next_constraint"
    assert result.output["suggestion"]["constraint_key"]
    assert isinstance(result.output["alternatives"], list)


def test_fetch_recent_history_handles_null_fit_score_and_context_actor():
    context = _baseline_context(actor_role="admin", environment="dev", user_id="caller-42")
    fake_row = SimpleNamespace(
        id=1,
        fit_score=None,
        job_title="Product Engineer",
        company="Acme",
        llm_provider="openai",
        llm_model="gpt-4",
        created_at=None,
        job_description="Need distributed systems experience.",
    )

    class _FakeQuery:
        def __init__(self, rows):
            self._rows = rows

        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def limit(self, *args, **kwargs):
            return self

        def all(self):
            return self._rows

    class _FakeDb:
        def query(self, *args, **kwargs):
            return _FakeQuery([fake_row])

    arguments = FetchRecentHistoryInput(
        user_id="payload-user-99",
        window_hours=24,
        max_items=5,
        include_internal_notes=False,
    )

    result = _fetch_recent_history(
        db=_FakeDb(),
        context=context,
        arguments=arguments,
    )

    assert isinstance(result.events, list)
    assert len(result.events) == 1
    assert result.events[0].actor == "caller-42"
    assert result.events[0].metadata["fit_score"] == 0.0
