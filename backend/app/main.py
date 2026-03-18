from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import time
from collections import OrderedDict

from sqlalchemy import inspect, text

sa_text = text
from sqlalchemy.exc import OperationalError

from .database import Base, SessionLocal, engine
from .models import cv, interview_chat, practice, score, user_profile  # noqa: F401 - register models
from .routers import cv as cv_router
from .routers import score as score_router
from .routers import settings as settings_router
from .routers import profile as profile_router
from .routers import jobs as jobs_router
from .routers import practice as practice_router
from .routers import interview_chat as interview_chat_router
from .routers import local_context as local_context_router
from .services.practice_embedder import run_embedding_worker

logger = logging.getLogger(__name__)


def _ensure_jobs_columns() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "jobs" not in table_names:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("jobs")}
    expected_columns = OrderedDict(
        [
            ("posted_at_raw", "VARCHAR(100)"),
            ("external_job_id", "VARCHAR(80)"),
            ("canonical_url", "TEXT"),
            ("employment_type", "VARCHAR(100)"),
            ("job_function", "VARCHAR(255)"),
            ("industries", "VARCHAR(255)"),
            ("applicants_count", "VARCHAR(32)"),
            ("benefits", "JSON"),
            ("salary", "VARCHAR(255)"),
            ("company_logo", "TEXT"),
            ("company_linkedin_url", "TEXT"),
            ("company_website", "TEXT"),
            ("company_address", "JSON"),
            ("company_employees_count", "INTEGER"),
            ("job_poster_name", "VARCHAR(255)"),
            ("job_poster_title", "VARCHAR(255)"),
            ("job_poster_profile_url", "TEXT"),
            ("reason", "TEXT"),
        ]
    )

    with engine.begin() as connection:
        for column_name, column_type in expected_columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                text(f'ALTER TABLE "jobs" ADD COLUMN "{column_name}" {column_type};')
            )
        indexes = {index["name"] for index in inspector.get_indexes("jobs")}
        if "ix_jobs_external_job_id" not in indexes:
            connection.execute(text('CREATE INDEX IF NOT EXISTS "ix_jobs_external_job_id" ON "jobs" ("external_job_id");'))
        if "ix_jobs_canonical_url" not in indexes:
            connection.execute(text('CREATE INDEX IF NOT EXISTS "ix_jobs_canonical_url" ON "jobs" ("canonical_url");'))


def _ensure_score_history_columns() -> None:
    """Add missing score history columns for schema upgrades."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "score_history" not in table_names:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("score_history")}
    expected_columns = OrderedDict([
        ("reason", "TEXT"),
        ("agent_plan", "JSON"),
        ("matched_keyword_evidence", "JSON"),
        ("missing_keyword_evidence", "JSON"),
        ("rewrite_suggestion_evidence", "JSON"),
    ])

    with engine.begin() as connection:
        for column_name, column_type in expected_columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                text(f'ALTER TABLE "score_history" ADD COLUMN "{column_name}" {column_type};')
            )


def _backfill_agent_runs_from_score_history() -> None:
    from .models.score import (
        AGENT_STATE_ACTION_PLAN,
        AGENT_STATE_COMPLETED,
        AGENT_STATE_REWRITE_PLAN,
        AGENT_STATE_SCORING,
        ScoreHistory,
        AgentRun,
        AgentRunArtifact,
        AgentRunTransition,
    )

    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(engine)
    tables = set(inspector.get_table_names())
    if not {"score_history", "agent_runs", "agent_run_artifacts", "agent_run_transitions"}.issubset(tables):
        return

    with SessionLocal() as db:
        existing_score_history_ids = {
            row[0]
            for row in db.query(AgentRun.score_history_id).filter(
                AgentRun.score_history_id.is_not(None),
            ).all()
            if row[0] is not None
        }
        query = db.query(ScoreHistory)
        if existing_score_history_ids:
            query = query.filter(~ScoreHistory.id.in_(existing_score_history_ids))
        score_history_rows = query.order_by(ScoreHistory.id.asc()).all()
        for row in score_history_rows:
            run = AgentRun(
                id=f"bh-{row.id}",
                score_history_id=row.id,
                current_state=AGENT_STATE_COMPLETED,
                status=AGENT_STATE_COMPLETED,
                actor="system",
                source="backfill",
                request_payload={
                    "cv_id": None,
                    "job_title": row.job_title or "",
                    "company": row.company or "",
                    "job_description": row.job_description or "",
                },
                attempt_count=1,
            )
            db.add(run)
            db.flush()

            if row.fit_score is not None:
                scoring_payload = {
                    "fit_score": row.fit_score,
                    "matched_keywords": row.matched_keywords or [],
                    "missing_keywords": row.missing_keywords or [],
                    "gap_analysis": row.gap_analysis or "",
                    "reason": row.reason or "",
                    "rewrite_suggestions": row.rewrite_suggestions or [],
                }
                run_transition = AgentRunTransition(
                    run_id=run.id,
                    score_history_id=row.id,
                    previous_state=AGENT_STATE_COMPLETED,
                    next_state=AGENT_STATE_SCORING,
                    trigger="backfill",
                    attempt=1,
                    idempotency_key=f"backfill-score-history-{row.id}",
                    actor="system",
                    source="backfill",
                )
                db.add(run_transition)
                db.flush()
                db.add(
                    AgentRunArtifact(
                        run_id=run.id,
                        score_history_id=row.id,
                        step=AGENT_STATE_SCORING,
                        actor="system",
                        source="backfill",
                        payload=scoring_payload,
                        evidence={"source": "backfill"},
                        attempt=1,
                        transition_id=run_transition.id,
                    ),
                )

            if row.agent_plan:
                action_plan_payload = row.agent_plan if isinstance(row.agent_plan, dict) else {}
                db.add(
                    AgentRunTransition(
                        run_id=run.id,
                        score_history_id=row.id,
                        previous_state=AGENT_STATE_SCORING,
                        next_state=AGENT_STATE_ACTION_PLAN,
                        trigger="backfill",
                        attempt=1,
                        idempotency_key=f"backfill-score-history-plan-{row.id}",
                        actor="system",
                        source="backfill",
                    ),
                )
                db.flush()
                db.add(
                    AgentRunArtifact(
                        run_id=run.id,
                        score_history_id=row.id,
                        step=AGENT_STATE_ACTION_PLAN,
                        actor="system",
                        source="backfill",
                        payload=action_plan_payload,
                        evidence={"source": "backfill"},
                        attempt=1,
                    ),
                )

            if row.rewrite_suggestions:
                db.add(
                    AgentRunTransition(
                        run_id=run.id,
                        score_history_id=row.id,
                        previous_state=AGENT_STATE_ACTION_PLAN,
                        next_state=AGENT_STATE_REWRITE_PLAN,
                        trigger="backfill",
                        attempt=1,
                        idempotency_key=f"backfill-score-history-rewrite-{row.id}",
                        actor="system",
                        source="backfill",
                    ),
                )
                db.flush()
                db.add(
                    AgentRunArtifact(
                        run_id=run.id,
                        score_history_id=row.id,
                        step=AGENT_STATE_REWRITE_PLAN,
                        actor="system",
                        source="backfill",
                        payload={"rewrite_suggestions": row.rewrite_suggestions or []},
                        evidence={"source": "backfill"},
                        attempt=1,
                    ),
                )

            db.add(
                AgentRunTransition(
                    run_id=run.id,
                    score_history_id=row.id,
                    previous_state=AGENT_STATE_SCORING if row.agent_plan else AGENT_STATE_COMPLETED,
                    next_state=AGENT_STATE_COMPLETED,
                    trigger="backfill-complete",
                    attempt=1,
                    idempotency_key=f"backfill-complete-{row.id}",
                    actor="system",
                    source="backfill",
                )
            )
            run.completed_at = row.created_at
            db.add(run)
        if score_history_rows:
            db.commit()


def _ensure_local_context_tool_call_columns() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "local_context_tool_calls" not in table_names:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("local_context_tool_calls")}
    expected_columns = OrderedDict(
        [
            ("request_uuid", "VARCHAR(36)"),
            ("tool_name", "VARCHAR(80)"),
            ("model_id", "VARCHAR(120)"),
            ("actor_role", "VARCHAR(64)"),
            ("environment", "VARCHAR(32)"),
            ("request_source", "VARCHAR(64)"),
            ("session_id", "VARCHAR(255)"),
            ("user_id", "VARCHAR(255)"),
            ("input_hash", "VARCHAR(64)"),
            ("result_hash", "VARCHAR(64)"),
            ("latency_ms", "DOUBLE PRECISION"),
            ("status", "VARCHAR(24)"),
            ("decision_rationale", "JSON"),
            ("error_message", "TEXT"),
            ("input_payload", "JSON"),
            ("output_payload", "JSON"),
        ]
    )

    with engine.begin() as connection:
        for column_name, column_type in expected_columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                text(f'ALTER TABLE "local_context_tool_calls" ADD COLUMN "{column_name}" {column_type};')
            )


def _ensure_pgvector_extension(connection: object) -> None:
    from sqlalchemy import text as sa_text
    connection.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector;"))


def _migrate_practice_schema_if_needed() -> None:
    """Drop old practice tables if they have the legacy per-company schema."""
    from sqlalchemy import inspect as sa_inspect, text as sa_text
    inspector = sa_inspect(engine)
    tables = set(inspector.get_table_names())
    if "practice_questions" not in tables:
        return
    existing_cols = {col["name"] for col in inspector.get_columns("practice_questions")}
    if "company_slug" not in existing_cols:
        return  # already on new schema
    logger.info("Migrating practice schema from per-company rows to junction table design.")
    with engine.begin() as conn:
        for tbl in ["practice_generations", "practice_session_questions", "practice_sessions", "question_companies", "practice_questions"]:
            conn.execute(sa_text(f"DROP TABLE IF EXISTS {tbl} CASCADE;"))
    logger.info("Old practice tables dropped — will be recreated with new schema.")


def _ensure_practice_questions_columns() -> None:
    """Add new columns to practice_questions that were added after initial table creation."""
    from sqlalchemy import text as sa_text
    from .config import get_settings
    dim = get_settings().practice_embedding_dim

    inspector = inspect(engine)
    if "practice_questions" not in set(inspector.get_table_names()):
        return
    existing = {col["name"]: col for col in inspector.get_columns("practice_questions")}
    with engine.begin() as conn:
        if "embedding" not in existing:
            conn.execute(sa_text(f"ALTER TABLE practice_questions ADD COLUMN embedding vector({dim});"))
        else:
            # If the column exists but with the wrong dimension, recreate it.
            current_type = str(existing["embedding"].get("type", ""))
            if str(dim) not in current_type:
                conn.execute(sa_text("ALTER TABLE practice_questions DROP COLUMN embedding;"))
                conn.execute(sa_text(f"ALTER TABLE practice_questions ADD COLUMN embedding vector({dim});"))
        additional_columns = {
            "scope_type": "VARCHAR(16)",
            "scope_job_id": "INTEGER",
            "source_table": "VARCHAR(64)",
            "source_id": "INTEGER",
            "source_window": "VARCHAR(64)",
        }
        for name, column_type in additional_columns.items():
            if name in existing:
                continue
            conn.execute(sa_text(f'ALTER TABLE practice_questions ADD COLUMN "{name}" {column_type};'))
        existing_indexes = {index["name"] for index in inspector.get_indexes("practice_questions")}
        index_map = {
            "ix_practice_questions_scope_type": "scope_type",
            "ix_practice_questions_scope_job_id": "scope_job_id",
            "ix_practice_questions_source_table": "source_table",
            "ix_practice_questions_source_id": "source_id",
            "ix_practice_questions_source_window": "source_window",
        }
        for index_name, column_name in index_map.items():
            if index_name not in existing_indexes:
                conn.execute(sa_text(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON practice_questions ("{column_name}");'))


def _ensure_interview_documents_columns() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "interview_knowledge_documents" not in table_names:
        return

    existing = {col["name"] for col in inspector.get_columns("interview_knowledge_documents")}
    with engine.begin() as conn:
        expected_columns = {
            "id": "INTEGER",
            "owner_type": "VARCHAR(16)",
            "job_id": "INTEGER",
            "source_filename": "VARCHAR(255)",
            "content_type": "VARCHAR(80)",
            "parsed_text": "TEXT",
            "parser_version": "VARCHAR(64)",
            "source_ref": "VARCHAR(120)",
            "status": "VARCHAR(24)",
            "error_message": "TEXT",
            "total_chunks": "INTEGER DEFAULT 0",
            "embedded_chunks": "INTEGER DEFAULT 0",
            "parsed_word_count": "INTEGER DEFAULT 0",
            "chunk_coverage_ratio": "FLOAT DEFAULT 0",
            "created_by_user_id": "VARCHAR(120)",
        }
        for column_name, column_type in expected_columns.items():
            if column_name in existing:
                continue
            conn.execute(sa_text(f'ALTER TABLE interview_knowledge_documents ADD COLUMN "{column_name}" {column_type};'))
        indexes = {index["name"] for index in inspector.get_indexes("interview_knowledge_documents")}
        index_sql = [
            ('"ix_interview_knowledge_documents_owner_type"', 'owner_type'),
            ('"ix_interview_knowledge_documents_job_id"', 'job_id'),
            ('"ix_interview_knowledge_documents_status"', 'status'),
        ]
        for index_name, column_name in index_sql:
            if index_name.strip('"') in indexes:
                continue
            conn.execute(sa_text(f'CREATE INDEX IF NOT EXISTS {index_name} ON interview_knowledge_documents ("{column_name}");'))
        conn.execute(
            sa_text(
                "UPDATE interview_knowledge_documents "
                "SET total_chunks = COALESCE(total_chunks, 0), "
                "    embedded_chunks = COALESCE(embedded_chunks, 0), "
                "    parsed_word_count = COALESCE(parsed_word_count, 0), "
                "    chunk_coverage_ratio = COALESCE(chunk_coverage_ratio, 0);"
            )
        )


def _ensure_interview_research_sessions_table() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "interview_research_sessions" not in table_names:
        return

    existing = {column["name"] for column in inspector.get_columns("interview_research_sessions")}
    expected_columns = {
        "id": "INTEGER",
        "session_id": "VARCHAR(64)",
        "job_id": "INTEGER",
        "role": "VARCHAR(255)",
        "company": "VARCHAR(255)",
        "status": "VARCHAR(32)",
        "stage": "VARCHAR(40)",
        "question_bank": "JSON",
        "source_urls": "JSON",
        "fallback_used": "BOOLEAN",
        "failure_reason": "VARCHAR(500)",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        "completed_at": "TIMESTAMP",
        "started_at": "TIMESTAMP",
        "processing_ms": "INTEGER",
    }
    with engine.begin() as connection:
        for column_name, column_type in expected_columns.items():
            if column_name in existing:
                continue
            connection.execute(sa_text(f'ALTER TABLE "interview_research_sessions" ADD COLUMN "{column_name}" {column_type};'))


def _ensure_interview_chat_sessions_table() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "interview_chat_sessions" not in table_names:
        return
    existing = {column["name"] for column in inspector.get_columns("interview_chat_sessions")}
    expected_columns = {
        "id": "INTEGER",
        "session_id": "VARCHAR(64)",
        "job_id": "INTEGER",
        "label": "VARCHAR(120)",
        "status": "VARCHAR(24)",
        "phase": "VARCHAR(32)",
        "current_question_index": "INTEGER",
        "is_waiting_for_candidate_question": "BOOLEAN",
        "question_plan": "JSON",
        "session_metadata": "JSON",
        "handoff_run_id": "VARCHAR(64)",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        "completed_at": "TIMESTAMP",
    }
    with engine.begin() as connection:
        for column_name, column_type in expected_columns.items():
            if column_name in existing:
                continue
            connection.execute(sa_text(f'ALTER TABLE "interview_chat_sessions" ADD COLUMN "{column_name}" {column_type};'))

        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text(
                    "UPDATE interview_chat_sessions "
                    "SET question_plan = COALESCE(question_plan, '[]'::json), "
                    "    session_metadata = COALESCE(session_metadata, '{}'::json), "
                    "    current_question_index = COALESCE(current_question_index, 0), "
                    "    is_waiting_for_candidate_question = COALESCE(is_waiting_for_candidate_question, FALSE);"
                )
            )
        else:
            connection.execute(
                sa_text(
                    "UPDATE interview_chat_sessions "
                    "SET question_plan = COALESCE(question_plan, '[]'), "
                    "    session_metadata = COALESCE(session_metadata, '{}'), "
                    "    current_question_index = COALESCE(current_question_index, 0), "
                    "    is_waiting_for_candidate_question = COALESCE(is_waiting_for_candidate_question, 0);"
                )
            )


def _ensure_interview_chat_turns_uniqueness() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "interview_chat_turns" not in table_names:
        return

    with engine.begin() as connection:
        # Keep the earliest row per (session_id, turn_index) before applying uniqueness.
        connection.execute(
            sa_text(
                "DELETE FROM interview_chat_turns "
                "WHERE id IN ("
                "  SELECT t1.id "
                "  FROM interview_chat_turns t1 "
                "  JOIN interview_chat_turns t2 "
                "    ON t1.session_id = t2.session_id "
                "   AND t1.turn_index = t2.turn_index "
                "   AND t1.id > t2.id"
                ");"
            )
        )
        connection.execute(
            sa_text(
                'CREATE UNIQUE INDEX IF NOT EXISTS "uix_interviewchatturn_session_turn" '
                'ON "interview_chat_turns" ("session_id", "turn_index");'
            )
        )


def _ensure_pgvector_index() -> None:
    from sqlalchemy import text as sa_text
    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_practice_questions_embedding "
            "ON practice_questions USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64);"
        ))


def _wait_for_database_and_init_schema(max_attempts: int = 10, base_delay_seconds: float = 1.0) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.begin() as conn:
                _ensure_pgvector_extension(conn)
            _migrate_practice_schema_if_needed()
            Base.metadata.create_all(bind=engine)
            _ensure_jobs_columns()
            _ensure_score_history_columns()
            _ensure_local_context_tool_call_columns()
            _backfill_agent_runs_from_score_history()
            _ensure_practice_questions_columns()
            _ensure_interview_documents_columns()
            _ensure_interview_research_sessions_table()
            _ensure_interview_chat_sessions_table()
            _ensure_interview_chat_turns_uniqueness()
            _ensure_pgvector_index()
            logger.info("Database connected and schema initialized.")
            return
        except OperationalError as exc:
            if attempt >= max_attempts:
                logger.exception("Database initialization failed after %s attempts.", max_attempts)
                raise
            delay = base_delay_seconds * attempt
            logger.warning(
                "Database not ready yet (attempt %s/%s). Retrying in %ss: %s",
                attempt,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)


app = FastAPI(
    title="Vett API",
    description="Local-first CV scorer powered by LiteLLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cv_router.router, prefix="/api")
app.include_router(score_router.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(profile_router.router, prefix="/api")
app.include_router(jobs_router.router, prefix="/api")
app.include_router(practice_router.router, prefix="/api")
app.include_router(interview_chat_router.router, prefix="/api")
app.include_router(local_context_router.router, prefix="/api")


@app.on_event("startup")
async def on_startup() -> None:
    _wait_for_database_and_init_schema()
    asyncio.create_task(run_embedding_worker())


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "vett-api"}
