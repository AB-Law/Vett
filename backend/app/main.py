from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import time
from collections import OrderedDict

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from .database import Base, engine
from .models import cv, score, practice  # noqa: F401 – register models
from .routers import cv as cv_router
from .routers import score as score_router
from .routers import settings as settings_router
from .routers import jobs as jobs_router
from .routers import practice as practice_router
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
            _ensure_practice_questions_columns()
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
app.include_router(jobs_router.router, prefix="/api")
app.include_router(practice_router.router, prefix="/api")


@app.on_event("startup")
async def on_startup() -> None:
    _wait_for_database_and_init_schema()
    asyncio.create_task(run_embedding_worker())


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "vett-api"}
