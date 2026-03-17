"""Vector helpers for adaptive practice candidate selection using pgvector."""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.practice import PracticeQuestion
from . import llm as llm_service


def _difficulty_rank(value: str | None) -> int:
    if not value:
        return 2
    v = value.lower()
    if "easy" in v:
        return 1
    if "hard" in v:
        return 3
    return 2


def _build_constraint_query(
    technique: str | None,
    pattern: str | None,
    complexity: str | None,
    language: str | None,
    difficulty_delta: int | None,
    base_difficulty: str | None,
    time_pressure_minutes: int | None,
) -> str | None:
    """Build a natural-language query string from constraints for embedding."""
    parts: list[str] = []

    if technique:
        parts.append(technique)
    if pattern:
        parts.append(pattern)
    if complexity:
        parts.append(f"{complexity} complexity")
    if language:
        parts.append(language)

    # Translate difficulty_delta into a target difficulty word
    base_rank = _difficulty_rank(base_difficulty)
    target_rank = max(1, min(3, base_rank + (difficulty_delta or 0)))
    difficulty_words = {1: "easy", 2: "medium", 3: "hard"}
    parts.append(f"{difficulty_words[target_rank]} difficulty")

    if time_pressure_minutes and time_pressure_minutes <= 15:
        parts.append("quick short problem")
    elif time_pressure_minutes and time_pressure_minutes >= 30:
        parts.append("complex in-depth problem")

    return " ".join(parts) if parts else None


def _truncate_job_description(description: str, max_chars: int = 1500) -> str:
    """Keep only the first max_chars of a job description to stay within token limits."""
    return description[:max_chars].strip()


async def _get_query_vector(
    db: Session,
    base_question: PracticeQuestion,
    job_description: str | None,
    difficulty_delta: int | None,
    language: str | None,
    technique: str | None,
    complexity: str | None,
    time_pressure_minutes: int | None,
    pattern: str | None,
) -> list[float] | None:
    """
    Build the best query vector for finding the next question.

    Priority:
      1. Constraint query string (technique/pattern/complexity → embed a short description)
      2. Job description (what this company actually tests for)
      3. Base question embedding (fallback — similar to what was just solved)
    """
    any_constraint = any([technique, pattern, complexity, language, time_pressure_minutes])

    # 1. Constraint-based query
    if any_constraint:
        constraint_text = _build_constraint_query(
            technique=technique,
            pattern=pattern,
            complexity=complexity,
            language=language,
            difficulty_delta=difficulty_delta,
            base_difficulty=base_question.difficulty,
            time_pressure_minutes=time_pressure_minutes,
        )
        if constraint_text:
            try:
                return await llm_service.generate_embedding(constraint_text)
            except Exception:
                pass  # fall through to next option

    # 2. Job description
    if job_description:
        truncated = _truncate_job_description(job_description)
        try:
            return await llm_service.generate_embedding(truncated)
        except Exception:
            pass  # fall through to next option

    # 3. Base question embedding (already stored, no API call needed)
    if base_question.embedding is not None:
        return list(base_question.embedding)

    # 4. Last resort: embed the base question title on the fly
    title = (base_question.title or "").strip()
    if title:
        try:
            vec = await llm_service.generate_embedding(title)
            base_question.embedding = vec
            db.add(base_question)
            db.flush()
            return vec
        except Exception:
            pass

    return None


async def pick_best_candidate(
    db: Session,
    base_question: PracticeQuestion,
    candidate_questions: list[PracticeQuestion],
    difficulty_delta: int | None,
    job_description: str | None = None,
    language: str | None = None,
    technique: str | None = None,
    complexity: str | None = None,
    time_pressure_minutes: int | None = None,
    pattern: str | None = None,
) -> tuple[PracticeQuestion | None, dict[str, float]]:
    active_candidates = [q for q in candidate_questions if q.id != base_question.id and q.is_active]
    if not active_candidates:
        return None, {}

    candidate_ids = [q.id for q in active_candidates]

    query_vec = await _get_query_vector(
        db=db,
        base_question=base_question,
        job_description=job_description,
        difficulty_delta=difficulty_delta,
        language=language,
        technique=technique,
        complexity=complexity,
        time_pressure_minutes=time_pressure_minutes,
        pattern=pattern,
    )

    if query_vec is not None:
        vec_literal = "[" + ",".join(str(v) for v in query_vec) + "]"
        rows = (
            db.execute(
                text(
                    "SELECT id, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity "
                    "FROM practice_questions "
                    "WHERE id = ANY(:ids) AND embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:vec AS vector) "
                    "LIMIT 1"
                ),
                {"vec": vec_literal, "ids": candidate_ids},
            )
            .fetchall()
        )
        if rows:
            best_id, similarity = rows[0]
            best = next((q for q in active_candidates if q.id == best_id), None)
            if best:
                return best, {"vector_similarity": float(similarity)}

    # Fallback: pick by difficulty alignment if no embeddings available yet
    target_rank = _difficulty_rank(base_question.difficulty) + (difficulty_delta or 0)
    target_rank = max(1, min(3, target_rank))
    best = min(active_candidates, key=lambda q: abs(_difficulty_rank(q.difficulty) - target_rank))
    return best, {"vector_similarity": 0.0}
