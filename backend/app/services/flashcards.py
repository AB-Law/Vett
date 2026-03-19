from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - tests may run without LLM dependency
    litellm = None  # type: ignore[assignment]
from sqlalchemy import func
from sqlalchemy.orm import Session
from ..models.interview import InterviewKnowledgeDocument
from ..models.score import Job
from ..models.study import StudyCard, StudyCardSet, StudyCardSetDocument
from . import interview_docs, llm as llm_service


ReviewRating = Literal["easy", "hard"]


MAX_CONTEXT_SNIPPETS = 8
MAX_CONTEXT_SNIPPETS_CAP = 20
DEFAULT_DECK_NAME = "Untitled deck"
MAX_TOTAL_CARDS = 5000
MAX_CARDS_PER_CHILD_DECK = 50
GENERATION_BATCH_SIZE = 4
FLASHCARD_MODEL_INSTRUCTIONS = """You are an interview trainer.

Return exactly one JSON object with this shape:
{{
  "cards": [
    {{
      "front": "question text",
      "back": "concise answer"
    }}
  ]
}}

Create flashcards that stay grounded in the provided context snippets.
Each front should be a clear interview-study question and each back should be a short, useful answer.
Respond with English text only and no markdown."""


def _normalize_card_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _context_scope_limit(num_cards: int) -> int:
    dynamic_limit = min(max(1, int(num_cards)) * 2, MAX_CONTEXT_SNIPPETS_CAP)
    return max(MAX_CONTEXT_SNIPPETS, dynamic_limit)


def _normalize_document_ids(document_ids: list[int] | None) -> list[int]:
    seen: set[int] = set()
    normalized: list[int] = []
    for raw_id in document_ids or []:
        doc_id = int(raw_id)
        if doc_id <= 0 or doc_id in seen:
            continue
        seen.add(doc_id)
        normalized.append(doc_id)
    return normalized


def _normalize_deck_name(name: str | None) -> str:
    candidate = _normalize_card_text(name)
    if not candidate:
        return DEFAULT_DECK_NAME
    return candidate[:255]


def _coerce_cards_payload(value: object) -> list[dict[str, str]]:
    if isinstance(value, dict):
        raw_cards = value.get("cards")
    else:
        raw_cards = value
    if not isinstance(raw_cards, list):
        return []

    output: list[dict[str, str]] = []
    for item in raw_cards:
        if not isinstance(item, dict):
            continue
        front = _normalize_card_text(item.get("front"))
        back = _normalize_card_text(item.get("back"))
        if not front or not back:
            continue
        output.append({"front": front, "back": back})
    return output


def _extract_json_payload(raw: str) -> str | None:
    matched = llm_service._extract_first_json_object(raw)  # type: ignore[attr-defined]
    if matched:
        return matched
    fallback_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not fallback_match:
        return None
    return fallback_match.group(0)


def _parse_flashcard_response(raw: str) -> list[dict[str, str]]:
    payload_text = _extract_json_payload(raw) or ""
    if not payload_text.strip():
        return []
    try:
        parsed = json.loads(payload_text)
    except Exception:
        try:
            parsed = json.loads(llm_service._sanitize_json_token_stream(payload_text))  # type: ignore[attr-defined]
        except Exception:
            parsed = {}
    if not isinstance(parsed, (dict, list)):
        return []
    return _coerce_cards_payload(parsed)


def _build_query_text(job_title: str | None, job_company: str | None, topic: str | None, job_description: str | None) -> str:
    chunks = [f"interview flashcards for job {job_title or 'this role'}"]
    if job_company:
        chunks.append(f"company: {job_company}")
    if topic:
        chunks.append(f"topic: {topic}")
    if job_description:
        chunks.append(f"job context: {(job_description or '')[:1200]}")
    return "\n".join(chunks).strip()


def _build_context_block(contexts: list[dict[str, str]]) -> str:
    rendered = []
    for index, context in enumerate(contexts, start=1):
        filename = _normalize_card_text(context.get("filename")) or f"source-{index}"
        snippet = _normalize_card_text(context.get("snippet")) or "No snippet available."
        rendered.append(f"{index}. {filename}: {snippet}")
    return "\n".join(rendered)


def _limit_cards(cards: list[dict[str, str]], num_cards: int) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for card in cards:
        front = card.get("front", "")
        back = card.get("back", "")
        key = (front.strip().lower(), back.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(card)
        if len(normalized) >= num_cards:
            break
    return normalized


def _empty_generation_diagnostics() -> dict[str, object]:
    return {
        "requested_cards": 0,
        "llm_cards_parsed": 0,
        "deduped_out": 0,
        "fallback_cards_used": 0,
        "fallback_used": False,
    }


async def _generate_cards_from_context_with_diagnostics(
    topic: str | None,
    contexts: list[dict[str, str]],
    num_cards: int,
    *,
    job_title: str | None,
    job_company: str | None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    diagnostics = _empty_generation_diagnostics()
    diagnostics["requested_cards"] = max(0, int(num_cards))

    prompt = "\n\n".join(
        [
            FLASHCARD_MODEL_INSTRUCTIONS,
            f"topic: {topic or 'general interview study'}",
            f"cards_needed: {num_cards}",
            f"job: {(job_title or 'Unknown role')} ({job_company or 'unknown company'})",
            "context_snippets:",
            _build_context_block(contexts),
        ]
    )

    if litellm is None:
        import litellm as runtime_litellm

        completion_client = runtime_litellm
    else:
        completion_client = litellm

    model, kwargs = llm_service._get_litellm_model()  # type: ignore[attr-defined]
    response = await completion_client.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=8192,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    logger.warning("FLASHCARD RAW RESPONSE (first 500 chars): %r", content[:500])
    llm_cards = _parse_flashcard_response(content)
    logger.warning("FLASHCARD PARSED CARDS COUNT: %d", len(llm_cards))
    llm_cards_limited = _limit_cards(llm_cards, num_cards)
    diagnostics["llm_cards_parsed"] = len(llm_cards_limited)
    diagnostics["fallback_used"] = False
    return llm_cards_limited, diagnostics


async def _generate_cards_from_context(
    topic: str | None,
    contexts: list[dict[str, str]],
    num_cards: int,
    *,
    job_title: str | None,
    job_company: str | None,
) -> list[dict[str, str]]:
    cards, _ = await _generate_cards_from_context_with_diagnostics(
        topic,
        contexts,
        num_cards,
        job_title=job_title,
        job_company=job_company,
    )
    return cards


def _build_review_interval(ease_factor: float, interval_days: int, rating: ReviewRating) -> tuple[float, int]:
    safe_factor = max(1.3, float(ease_factor))
    safe_interval = max(0, int(interval_days))
    if rating == "easy":
        if safe_interval <= 1:
            next_interval = 6
        else:
            next_interval = int(round(safe_interval * safe_factor))
        return safe_factor + 0.1, max(1, next_interval)
    next_interval = 1 if safe_interval <= 1 else max(1, int(round(safe_interval * 0.5)))
    next_factor = max(1.3, safe_factor - 0.2)
    return next_factor, max(1, next_interval)


def serialize_study_card(card: StudyCard) -> dict[str, object]:
    return {
        "id": card.id,
        "front": card.front,
        "back": card.back,
        "last_reviewed_at": _serialize_datetime(card.last_reviewed_at),
        "ease_factor": float(card.ease_factor),
        "interval_days": int(card.interval_days),
    }


def _card_set_document_ids(db: Session, card_set_ids: list[int]) -> dict[int, list[int]]:
    if not card_set_ids:
        return {}
    rows = (
        db.query(StudyCardSetDocument.card_set_id, StudyCardSetDocument.document_id)
        .filter(StudyCardSetDocument.card_set_id.in_(card_set_ids))
        .order_by(StudyCardSetDocument.id.asc())
        .all()
    )
    output: dict[int, list[int]] = {int(card_set_id): [] for card_set_id in card_set_ids}
    for card_set_id, document_id in rows:
        if card_set_id is None or document_id is None:
            continue
        output.setdefault(int(card_set_id), []).append(int(document_id))
    return output


def _cards_by_set_ids(db: Session, card_set_ids: list[int]) -> dict[int, list[StudyCard]]:
    if not card_set_ids:
        return {}
    rows = (
        db.query(StudyCard)
        .filter(StudyCard.card_set_id.in_(card_set_ids))
        .order_by(StudyCard.card_set_id.asc(), StudyCard.id.asc())
        .all()
    )
    output: dict[int, list[StudyCard]] = {int(card_set_id): [] for card_set_id in card_set_ids}
    for card in rows:
        if card.card_set_id is None:
            continue
        output.setdefault(int(card.card_set_id), []).append(card)
    return output


def serialize_study_card_set(card_set: StudyCardSet, card_count: int, document_ids: list[int] | None = None) -> dict[str, object]:
    ids = [int(doc_id) for doc_id in document_ids or []]
    return {
        "id": int(card_set.id),
        "job_id": int(card_set.job_id) if card_set.job_id is not None else None,
        "parent_card_set_id": int(card_set.parent_card_set_id) if card_set.parent_card_set_id is not None else None,
        "name": _normalize_deck_name(card_set.name),
        "topic": card_set.topic,
        "created_at": _serialize_datetime(card_set.created_at),
        "card_count": int(card_count),
        "document_ids": ids,
        "document_count": len(ids),
    }


def _persist_card_set_documents(db: Session, card_set_id: int, document_ids: list[int]) -> None:
    if not document_ids:
        return
    db.add_all(
        [
            StudyCardSetDocument(card_set_id=card_set_id, document_id=document_id)
            for document_id in document_ids
        ]
    )


def _build_linear_contexts_from_documents(
    documents: list[InterviewKnowledgeDocument],
    num_cards: int,
) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    target_limit = _context_scope_limit(num_cards)
    for document in documents:
        chunks = interview_docs.chunk_interview_text(document.parsed_text)
        for chunk in chunks[:2]:
            contexts.append(
                {
                    "filename": document.source_filename,
                    "snippet": chunk,
                }
            )
            if len(contexts) >= target_limit:
                return contexts
    return contexts


def _batched_contexts(
    contexts: list[dict[str, str]],
    *,
    batch_index: int,
    num_cards: int,
) -> list[dict[str, str]]:
    if not contexts:
        return []
    limit = min(len(contexts), _context_scope_limit(num_cards))
    if len(contexts) <= limit:
        # Rotate ordering to avoid identical prompt ordering across child decks.
        offset = batch_index % len(contexts)
        return contexts[offset:] + contexts[:offset]
    start = (batch_index * max(1, limit // 2)) % len(contexts)
    window = [contexts[(start + i) % len(contexts)] for i in range(limit)]
    return window


def _dedupe_cards_global(cards: list[dict[str, str]], seen: set[tuple[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    for card in cards:
        front = _normalize_card_text(card.get("front"))
        back = _normalize_card_text(card.get("back"))
        if not front or not back:
            continue
        key = (front.lower(), back.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append({"front": front, "back": back})
    return unique


def _persist_card_set_with_cards(
    db: Session,
    *,
    job_id: int | None,
    topic: str | None,
    name: str,
    document_ids: list[int],
    cards_data: list[dict[str, str]],
    num_cards: int,
    parent_card_set_id: int | None = None,
) -> tuple[StudyCardSet, list[StudyCard]]:
    card_set = StudyCardSet(
        job_id=job_id,
        topic=topic or None,
        name=name,
        parent_card_set_id=parent_card_set_id,
    )
    db.add(card_set)
    db.flush()
    _persist_card_set_documents(db, int(card_set.id), document_ids)

    cards_to_create: list[StudyCard] = []
    for card in cards_data[:num_cards]:
        cards_to_create.append(
            StudyCard(
                card_set_id=card_set.id,
                front=card["front"],
                back=card["back"],
                ease_factor=2.5,
                interval_days=1,
            )
        )
    db.add_all(cards_to_create)
    db.flush()
    created_cards = list(cards_to_create)
    return card_set, created_cards


async def create_study_card_set(
    db: Session,
    job_id: int | None = None,
    topic: str | None = None,
    num_cards: int = 10,
    document_ids: list[int] | None = None,
    name: str | None = None,
    generate_per_section: bool = False,
) -> dict[str, object]:
    num_cards = max(1, min(MAX_TOTAL_CARDS, int(num_cards)))
    normalized_document_ids = _normalize_document_ids(document_ids)
    normalized_name = _normalize_deck_name(name)
    contexts: list[dict[str, str]] = []
    selected_docs: list[InterviewKnowledgeDocument] = []
    job_title: str | None = None
    job_company: str | None = None

    if normalized_document_ids:
        selected_docs = (
            db.query(InterviewKnowledgeDocument)
            .filter(InterviewKnowledgeDocument.id.in_(normalized_document_ids))
            .order_by(InterviewKnowledgeDocument.id.desc())
            .all()
        )
        if not selected_docs:
            raise ValueError("No matching interview documents found for selected document_ids.")
        if topic:
            try:
                query_vector = await llm_service.generate_embedding(topic)
            except Exception as exc:
                raise ValueError(f"Could not create embedding for retrieval: {exc}")
            if not query_vector:
                raise ValueError("Could not create retrieval vector for flashcard generation.")
            try:
                contexts = interview_docs.fetch_context_from_interview_documents(
                    db,
                    query_vector=query_vector,
                    job_id=None,
                    document_ids=normalized_document_ids,
                    scope_limit=_context_scope_limit(num_cards),
                    scope_overlap=2,
                )
            except Exception as exc:
                raise ValueError(f"Could not retrieve interview context for flashcards: {exc}")
        if not contexts:
            contexts = _build_linear_contexts_from_documents(selected_docs, num_cards)
    else:
        if job_id is None:
            raise ValueError("Provide either job_id or document_ids for flashcard generation.")
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError("Job not found")
        job_title = job.title
        job_company = job.company

        try:
            query_text = _build_query_text(job.title, job.company, topic, job.description)
            query_vector = await llm_service.generate_embedding(query_text)
        except Exception as exc:
            raise ValueError(f"Could not create embedding for retrieval: {exc}")

        if query_vector is None:
            raise ValueError("Could not create retrieval vector for flashcard generation.")
        try:
            contexts = interview_docs.fetch_context_from_interview_documents(
                db,
                query_vector=query_vector,
                job_id=job_id,
                document_ids=None,
                scope_limit=_context_scope_limit(num_cards),
                scope_overlap=2,
            )
        except Exception as exc:
            raise ValueError(f"Could not retrieve interview context for flashcards: {exc}")

    if not contexts:
        raise ValueError("No interview document content available for flashcard generation.")

    if not job_title:
        job_title = "Standalone Study"
    if not job_company:
        job_company = "Vett"

    created_sets: list[tuple[StudyCardSet, str]] = []
    parent_card_set_id: int | None = None
    generation_diagnostics = _empty_generation_diagnostics()
    should_create_parent = num_cards > MAX_CARDS_PER_CHILD_DECK
    if should_create_parent:
        parent_card_set = StudyCardSet(
            job_id=job_id,
            topic=topic or None,
            name=normalized_name,
            parent_card_set_id=None,
        )
        db.add(parent_card_set)
        db.flush()
        _persist_card_set_documents(db, int(parent_card_set.id), normalized_document_ids)
        parent_card_set_id = int(parent_card_set.id)

    cards_per_child = min(num_cards, MAX_CARDS_PER_CHILD_DECK)
    child_deck_count = max(1, (num_cards + MAX_CARDS_PER_CHILD_DECK - 1) // MAX_CARDS_PER_CHILD_DECK)
    global_seen_cards: set[tuple[str, str]] = set()
    if generate_per_section and normalized_document_ids:
        if not selected_docs:
            raise ValueError("No matching interview documents found for selected document_ids.")
        section_specs: list[tuple[str, list[dict[str, str]]]] = []
        for doc_index, document in enumerate(selected_docs):
            sections = interview_docs.split_document_into_sections(document.parsed_text)
            for section_index, section in enumerate(sections):
                section_title = _normalize_card_text(section.get("title")) or "Section"
                section_text = _normalize_card_text(section.get("text"))
                if not section_text:
                    continue
                section_contexts_full = [
                    {
                        "filename": f"{document.source_filename} · {section_title}",
                        "snippet": snippet,
                    }
                    for snippet in interview_docs.chunk_interview_text(section_text)
                ]
                section_contexts = _batched_contexts(
                    section_contexts_full,
                    batch_index=doc_index + section_index,
                    num_cards=cards_per_child,
                )
                if not section_contexts:
                    continue
                section_deck_name = _normalize_deck_name(f"{normalized_name} - {section_title}")
                section_specs.append((section_deck_name, section_contexts))
        for offset in range(0, len(section_specs), GENERATION_BATCH_SIZE):
            batch_specs = section_specs[offset : offset + GENERATION_BATCH_SIZE]
            batch_tasks = [
                _generate_cards_from_context_with_diagnostics(
                    topic,
                    section_contexts,
                    cards_per_child,
                    job_title=job_title,
                    job_company=job_company,
                )
                for _, section_contexts in batch_specs
            ]
            generated_sections = await asyncio.gather(*batch_tasks) if batch_tasks else []
            batch_created_sets: list[StudyCardSet] = []
            for (section_deck_name, _), generated_section in zip(batch_specs, generated_sections):
                section_cards_data, section_generation_diagnostics = generated_section
                generation_diagnostics["requested_cards"] = int(generation_diagnostics["requested_cards"]) + int(
                    section_generation_diagnostics["requested_cards"]
                )
                generation_diagnostics["llm_cards_parsed"] = int(generation_diagnostics["llm_cards_parsed"]) + int(
                    section_generation_diagnostics["llm_cards_parsed"]
                )
                generation_diagnostics["fallback_cards_used"] = int(generation_diagnostics["fallback_cards_used"]) + int(
                    section_generation_diagnostics["fallback_cards_used"]
                )
                unique_before_global_dedupe = len(section_cards_data)
                unique_cards = _dedupe_cards_global(section_cards_data, global_seen_cards)
                generation_diagnostics["deduped_out"] = int(generation_diagnostics["deduped_out"]) + max(
                    0,
                    unique_before_global_dedupe - len(unique_cards),
                )
                if not unique_cards:
                    continue
                card_set, _ = _persist_card_set_with_cards(
                    db,
                    job_id=job_id,
                    topic=topic,
                    name=section_deck_name,
                    document_ids=normalized_document_ids,
                    cards_data=unique_cards,
                    num_cards=cards_per_child,
                    parent_card_set_id=parent_card_set_id,
                )
                created_sets.append((card_set, section_deck_name))
                batch_created_sets.append(card_set)
            if batch_created_sets:
                db.commit()
                for card_set in batch_created_sets:
                    db.refresh(card_set)
        if not created_sets:
            raise ValueError("No sections produced valid flashcards.")
    else:
        child_specs: list[tuple[int, int, list[dict[str, str]], str]] = []
        for child_index in range(child_deck_count):
            remaining_cards = num_cards - (child_index * MAX_CARDS_PER_CHILD_DECK)
            target_cards_for_child = min(MAX_CARDS_PER_CHILD_DECK, remaining_cards)
            child_name = normalized_name
            if child_deck_count > 1:
                child_name = _normalize_deck_name(f"{normalized_name} - Deck {child_index + 1}")
            child_specs.append(
                (
                    child_index,
                    target_cards_for_child,
                    _batched_contexts(contexts, batch_index=child_index, num_cards=target_cards_for_child),
                    child_name,
                )
            )
        for offset in range(0, len(child_specs), GENERATION_BATCH_SIZE):
            batch_specs = child_specs[offset : offset + GENERATION_BATCH_SIZE]
            batch_tasks = [
                _generate_cards_from_context_with_diagnostics(
                    topic,
                    child_contexts,
                    target_cards_for_child,
                    job_title=job_title,
                    job_company=job_company,
                )
                for _, target_cards_for_child, child_contexts, _ in batch_specs
            ]
            generated_children = await asyncio.gather(*batch_tasks) if batch_tasks else []
            batch_created_sets: list[StudyCardSet] = []
            for (child_index, target_cards_for_child, _, child_name), generated_child in zip(batch_specs, generated_children):
                generated_cards_for_child, child_generation_diagnostics = generated_child
                generation_diagnostics["requested_cards"] = int(generation_diagnostics["requested_cards"]) + int(
                    child_generation_diagnostics["requested_cards"]
                )
                generation_diagnostics["llm_cards_parsed"] = int(generation_diagnostics["llm_cards_parsed"]) + int(
                    child_generation_diagnostics["llm_cards_parsed"]
                )
                generation_diagnostics["fallback_cards_used"] = int(generation_diagnostics["fallback_cards_used"]) + int(
                    child_generation_diagnostics["fallback_cards_used"]
                )
                pre_dedupe_count = len(generated_cards_for_child)
                cards_data = _dedupe_cards_global(generated_cards_for_child, global_seen_cards)
                generation_diagnostics["deduped_out"] = int(generation_diagnostics["deduped_out"]) + max(
                    0,
                    pre_dedupe_count - len(cards_data),
                )
                if not cards_data:
                    continue
                card_set, _ = _persist_card_set_with_cards(
                    db,
                    job_id=job_id,
                    topic=topic,
                    name=child_name,
                    document_ids=normalized_document_ids,
                    cards_data=cards_data,
                    num_cards=target_cards_for_child,
                    parent_card_set_id=parent_card_set_id,
                )
                created_sets.append((card_set, child_name))
                batch_created_sets.append(card_set)
            if batch_created_sets:
                db.commit()
                for card_set in batch_created_sets:
                    db.refresh(card_set)
        if not created_sets:
            raise ValueError("No valid flashcards were generated.")

    first_card_set, _ = created_sets[0]
    created_set_ids = [int(card_set.id) for card_set, _ in created_sets]
    cards_by_set = _cards_by_set_ids(db, created_set_ids)
    document_ids_by_set = _card_set_document_ids(db, created_set_ids)
    card_sets_payload: list[dict[str, object]] = []
    for card_set, deck_name in created_sets:
        set_cards = cards_by_set.get(int(card_set.id), [])
        document_ids = document_ids_by_set.get(int(card_set.id), [])
        card_sets_payload.append(
            {
                "card_set_id": int(card_set.id),
                "cards": [serialize_study_card(card) for card in set_cards],
                "card_set": serialize_study_card_set(card_set, len(set_cards), document_ids=document_ids),
                "name": deck_name,
            }
        )
    return {
        "card_set_id": int(first_card_set.id),
        "cards": [serialize_study_card(card) for card in cards_by_set.get(int(first_card_set.id), [])],
        "card_set": serialize_study_card_set(
            first_card_set,
            len(cards_by_set.get(int(first_card_set.id), [])),
            document_ids=document_ids_by_set.get(int(first_card_set.id), []),
        ),
        "card_sets": card_sets_payload if len(card_sets_payload) > 1 else [],
        "parent_card_set_id": parent_card_set_id,
        "generation_diagnostics": {
            **generation_diagnostics,
            "fallback_used": int(generation_diagnostics["fallback_cards_used"]) > 0,
        },
    }


def list_study_card_sets(db: Session, limit: int = 20) -> list[dict[str, object]]:
    safe_limit = max(1, min(100, int(limit)))
    card_sets = (
        db.query(StudyCardSet)
        .order_by(StudyCardSet.id.desc())
        .limit(safe_limit)
        .all()
    )
    if not card_sets:
        return []

    set_ids = [int(card_set.id) for card_set in card_sets]
    count_rows = (
        db.query(StudyCard.card_set_id, func.count(StudyCard.id))
        .filter(StudyCard.card_set_id.in_(set_ids))
        .group_by(StudyCard.card_set_id)
        .all()
    )
    count_by_set_id = {int(card_set_id): int(total) for card_set_id, total in count_rows}
    document_ids_by_set_id = _card_set_document_ids(db, set_ids)
    return [
        serialize_study_card_set(
            card_set,
            count_by_set_id.get(int(card_set.id), 0),
            document_ids=document_ids_by_set_id.get(int(card_set.id), []),
        )
        for card_set in card_sets
    ]


def get_study_card_set_summary(db: Session, card_set_id: int) -> dict[str, object]:
    card_set = db.query(StudyCardSet).filter(StudyCardSet.id == card_set_id).first()
    if not card_set:
        raise ValueError("Study card set not found")
    card_count = (
        db.query(func.count(StudyCard.id))
        .filter(StudyCard.card_set_id == card_set_id)
        .scalar()
        or 0
    )
    document_ids = _card_set_document_ids(db, [int(card_set.id)]).get(int(card_set.id), [])
    return serialize_study_card_set(card_set, int(card_count), document_ids=document_ids)


def get_study_card_set_cards(db: Session, card_set_id: int) -> dict[str, object]:
    card_set = db.query(StudyCardSet).filter(StudyCardSet.id == card_set_id).first()
    if not card_set:
        raise ValueError("Study card set not found")
    cards = (
        db.query(StudyCard)
        .filter(StudyCard.card_set_id == card_set_id)
        .order_by(StudyCard.id.asc())
        .all()
    )
    document_ids_by_set_id = _card_set_document_ids(db, [int(card_set.id)])
    document_ids = document_ids_by_set_id.get(int(card_set.id), [])
    return {
        "card_set_id": int(card_set.id),
        "job_id": int(card_set.job_id) if card_set.job_id is not None else None,
        "parent_card_set_id": int(card_set.parent_card_set_id) if card_set.parent_card_set_id is not None else None,
        "name": _normalize_deck_name(card_set.name),
        "topic": card_set.topic,
        "created_at": _serialize_datetime(card_set.created_at),
        "cards": [serialize_study_card(card) for card in cards],
        "document_ids": document_ids,
        "document_count": len(document_ids),
    }


def update_study_card_set_name(db: Session, card_set_id: int, name: str) -> StudyCardSet:
    card_set = db.query(StudyCardSet).filter(StudyCardSet.id == card_set_id).first()
    if not card_set:
        raise ValueError("Study card set not found")
    card_set.name = _normalize_deck_name(name)
    db.add(card_set)
    db.commit()
    db.refresh(card_set)
    return card_set


def delete_study_card_set(db: Session, card_set_id: int) -> None:
    card_set = db.query(StudyCardSet).filter(StudyCardSet.id == card_set_id).first()
    if not card_set:
        raise ValueError("Study card set not found")
    child_set_ids = [int(row[0]) for row in db.query(StudyCardSet.id).filter(StudyCardSet.parent_card_set_id == card_set_id).all()]
    target_ids = [card_set_id, *child_set_ids]
    db.query(StudyCard).filter(StudyCard.card_set_id.in_(target_ids)).delete(synchronize_session=False)
    db.query(StudyCardSetDocument).filter(StudyCardSetDocument.card_set_id.in_(target_ids)).delete(synchronize_session=False)
    db.query(StudyCardSet).filter(StudyCardSet.id.in_(target_ids)).delete(synchronize_session=False)
    db.commit()


def review_study_card(db: Session, card_id: int, rating: ReviewRating) -> StudyCard:
    card = db.query(StudyCard).filter(StudyCard.id == card_id).first()
    if not card:
        raise ValueError("Study card not found")

    ease_factor = card.ease_factor if card.ease_factor is not None else 2.5
    interval_days = card.interval_days if card.interval_days is not None else 1
    next_ease_factor, next_interval = _build_review_interval(ease_factor, interval_days, rating)

    card.ease_factor = round(next_ease_factor, 4)
    card.interval_days = next_interval
    card.last_reviewed_at = datetime.now(timezone.utc)
    db.add(card)
    db.commit()
    db.refresh(card)
    return card
