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
from ..models.score import Job
from ..config import get_settings
from . import llm as llm_service

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20          # questions embedded per iteration
_SLEEP_BETWEEN_BATCHES = 2.0   # seconds — keeps CPU/API load low
_SLEEP_WHEN_IDLE = 60.0        # seconds — when all questions are embedded


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
