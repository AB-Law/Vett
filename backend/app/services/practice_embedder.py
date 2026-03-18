"""Background worker that pre-embeds PracticeQuestion rows.

Questions belonging to companies with active jobs are prioritised so that
the most relevant questions are ready first. Embeddings are stored directly
on the PracticeQuestion row and are shared across all users — no one ever
re-embeds the same question twice.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models.practice import PracticeQuestion, QuestionCompany
from ..models.interview import InterviewKnowledgeDocument
from ..models.score import Job
from ..config import get_settings
from . import llm as llm_service
from . import interview_docs

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20          # questions embedded per iteration
_SLEEP_BETWEEN_BATCHES = 2.0   # seconds — keeps CPU/API load low
_SLEEP_WHEN_IDLE = 60.0        # seconds — when all questions are embedded
_DOC_BATCH_SIZE = 6


def _get_priority_company_slugs(db: Session) -> list[str]:
    """Return company slugs that have at least one job on the jobs page."""
    rows = (
        db.query(func.lower(Job.company))
        .filter(Job.company.isnot(None))
        .distinct()
        .all()
    )
    return [row[0].strip() for row in rows if row[0]]


def _fetch_batch(db: Session, priority_slugs: list[str]) -> list[PracticeQuestion]:
    """Fetch up to _BATCH_SIZE questions that need embedding, priority companies first."""
    needs_embed = and_(
        PracticeQuestion.is_active.is_(True),
        PracticeQuestion.embedding.is_(None),
        PracticeQuestion.source_table.is_(None),
    )
    batch: list[PracticeQuestion] = []

    if priority_slugs:
        priority_ids_select = (
            db.query(QuestionCompany.question_id)
            .filter(QuestionCompany.company_slug.in_(priority_slugs))
        )
        batch = (
            db.query(PracticeQuestion)
            .filter(needs_embed, PracticeQuestion.id.in_(priority_ids_select))
            .order_by(PracticeQuestion.id)
            .limit(_BATCH_SIZE)
            .all()
        )

    if len(batch) < _BATCH_SIZE:
        remaining = _BATCH_SIZE - len(batch)
        already_ids = {q.id for q in batch}
        extra = (
            db.query(PracticeQuestion)
            .filter(needs_embed, ~PracticeQuestion.id.in_(already_ids) if already_ids else True)
            .order_by(PracticeQuestion.id)
            .limit(remaining)
            .all()
        )
        batch.extend(extra)

    return batch


def _fetch_document_batch(db: Session, limit: int = _DOC_BATCH_SIZE) -> list[InterviewKnowledgeDocument]:
    return (
        db.query(InterviewKnowledgeDocument)
        .filter(InterviewKnowledgeDocument.status == "pending")
        .order_by(InterviewKnowledgeDocument.id)
        .limit(limit)
        .all()
    )


def _build_document_row(
    document: InterviewKnowledgeDocument,
    chunk_index: int,
    source_window: str,
) -> PracticeQuestion:
    scope_type = "global" if (document.owner_type or "global") == "global" else "job"
    return PracticeQuestion(
        title=f"{document.source_filename} - chunk {chunk_index + 1}",
        url=document.source_ref,
        difficulty=None,
        acceptance=None,
        source_commit=None,
        scope_type=scope_type,
        scope_job_id=document.job_id if scope_type == "job" else None,
        source_table=interview_docs.DOC_TABLE_NAME,
        source_id=document.id,
        source_window=source_window,
    )


def _prepare_document_rows_for_embedding(
    db: Session,
    document: InterviewKnowledgeDocument,
) -> list[tuple[PracticeQuestion, str]]:
    chunks = interview_docs.chunk_interview_text(document.parsed_text)
    if not chunks:
        return []

    scoped_rows: list[tuple[PracticeQuestion, str]] = []
    for chunk_index, chunk_text in enumerate(chunks):
        source_window = interview_docs.make_source_window(chunk_index)
        existing = (
            db.query(PracticeQuestion)
            .filter(
                PracticeQuestion.source_table == interview_docs.DOC_TABLE_NAME,
                PracticeQuestion.source_id == document.id,
                PracticeQuestion.source_window == source_window,
            )
            .first()
        )
        if existing is None:
            existing = _build_document_row(
                document=document,
                chunk_index=chunk_index,
                source_window=source_window,
            )
            db.add(existing)
            db.flush()
            scoped_rows.append((existing, chunk_text))
            continue
        if existing.embedding is not None:
            continue
        scoped_rows.append((existing, chunk_text))

    return scoped_rows


async def _embed_document_rows(
    db: Session,
    practice_rows: list[tuple[PracticeQuestion, str]],
    expected_model: str,
) -> tuple[int, list[str]]:
    if not practice_rows:
        return 0, []

    tasks = [llm_service.generate_embedding(chunk_text) for _, chunk_text in practice_rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures: list[str] = []
    embedded_count = 0
    for (question, _), result in zip(practice_rows, results):
        if isinstance(result, Exception):
            failures.append(str(result))
            continue
        question.embedding = result
        question.embedding_model = expected_model
        db.add(question)
        embedded_count += 1
    if embedded_count:
        db.commit()
    return embedded_count, failures


def _mark_document_failed(document: InterviewKnowledgeDocument, errors: list[str]) -> None:
    document.status = "failed"
    if errors:
        signature = interview_docs.dedupe_error_signature("; ".join(errors))
        if signature:
            document.error_message = f"embedding_failed:{signature}"
        else:
            document.error_message = "embedding_failed"
    else:
        document.error_message = "embedding_failed"


def _mark_document_embedded(document: InterviewKnowledgeDocument, embedded_count: int, total_chunks: int) -> None:
    if embedded_count and embedded_count >= total_chunks:
        document.status = "embedded"
        document.error_message = None
    elif embedded_count:
        document.status = "processing"
        document.error_message = None
    else:
        _mark_document_failed(document, ["no_embedding_vectors_created"])


async def _run_interview_document_worker(db: Session, expected_model: str) -> int:
    documents = _fetch_document_batch(db)
    if not documents:
        return 0

    total_processed = 0
    for document in documents:
        document.status = "processing"
        db.add(document)
        db.commit()

        chunks = interview_docs.chunk_interview_text(document.parsed_text)
        if not chunks:
            _mark_document_failed(document, ["no text parsed"])
            db.add(document)
            db.commit()
            continue

        try:
            rows_to_embed = _prepare_document_rows_for_embedding(db, document)
        except Exception as exc:
            logger.exception("Failed to prepare chunks for interview doc %s", document.id)
            _mark_document_failed(document, [str(exc)])
            db.add(document)
            db.commit()
            continue

        success_count, failures = await _embed_document_rows(db, rows_to_embed, expected_model)
        total_processed += success_count
        if not rows_to_embed:
            document.status = "embedded"
            document.error_message = None
            db.add(document)
            db.commit()
            continue

        if failures:
            logger.debug("Interview embedding failures for doc %s: %s", document.id, "; ".join(failures))
            if success_count > 0:
                document.error_message = f"partial_embed:{interview_docs.dedupe_error_signature('; '.join(failures))}"
                document.status = "processing"
            else:
                _mark_document_failed(document, failures)
        else:
            document.status = "embedded"
            document.error_message = None
        db.add(document)
        db.commit()

    return total_processed


async def _embed_batch(db: Session, questions: list[PracticeQuestion], expected_model: str) -> int:
    """Embed a list of questions, persisting results to DB. Returns count embedded."""
    texts = []
    for q in questions:
        parts = [q.title or ""]
        if q.difficulty:
            parts.append(f"difficulty: {q.difficulty.lower()}")
        if q.acceptance:
            parts.append(f"acceptance: {q.acceptance}")
        texts.append(" ".join(p for p in parts if p).strip())

    tasks = [llm_service.generate_embedding(t) if t else None for t in texts]
    results = await asyncio.gather(*[t for t in tasks if t is not None], return_exceptions=True)

    result_iter = iter(results)
    count = 0
    for question, text_val in zip(questions, texts):
        if not text_val:
            continue
        result = next(result_iter)
        if isinstance(result, Exception):
            logger.debug("Embedding failed for question %s: %s", question.id, result)
            continue
        question.embedding = result
        question.embedding_model = expected_model
        db.add(question)
        count += 1

    if count:
        db.commit()

    return count


async def run_embedding_worker() -> None:
    """Long-running coroutine. Launch via asyncio.create_task on startup."""
    logger.info("Practice embedding worker started.")
    settings = get_settings()
    expected_model = settings.practice_embedding_model

    while True:
        try:
            db: Session = SessionLocal()
            try:
                priority_slugs = _get_priority_company_slugs(db)
                document_count = await _run_interview_document_worker(db, expected_model)
                if document_count:
                    logger.info("Embedded %d interview document chunks.", document_count)
                batch = _fetch_batch(db, priority_slugs)

                if not batch:
                    logger.debug("All questions embedded — sleeping %ss.", _SLEEP_WHEN_IDLE)
                    await asyncio.sleep(_SLEEP_WHEN_IDLE)
                    continue

                count = await _embed_batch(db, batch, expected_model)
                logger.info(
                    "Embedded %d/%d questions (priority slugs: %s).",
                    count,
                    len(batch),
                    priority_slugs[:5],
                )
            finally:
                db.close()

        except Exception:
            logger.exception("Embedding worker error — will retry after sleep.")

        await asyncio.sleep(_SLEEP_BETWEEN_BATCHES)
