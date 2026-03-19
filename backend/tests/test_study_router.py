import asyncio
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.interview import InterviewKnowledgeDocument
from app.models.score import Job
from app.models.study import MindMap, StudyCard, StudyCardSet, StudyCardSetDocument
from app.routers import study as study_router


def _new_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine,
        tables=[
            Job.__table__,
            InterviewKnowledgeDocument.__table__,
            StudyCardSet.__table__,
            StudyCard.__table__,
            StudyCardSetDocument.__table__,
            MindMap.__table__,
        ],
    )
    return sessionmaker(bind=engine)()


def test_review_flashcard_route_returns_updated_card(monkeypatch):
    db = _new_session()

    card = StudyCard(
        id=88,
        card_set_id=1,
        front="Q",
        back="A",
        ease_factor=2.5,
        interval_days=1,
    )

    def _fake_review(db_session, card_id: int, rating: str):
        return card

    monkeypatch.setattr(study_router.flashcards, "review_study_card", _fake_review)

    result = study_router.review_flashcard(88, payload=study_router.ReviewRequest(rating="easy"), db=db)

    assert result.id == 88
    assert result.front == "Q"


def test_create_flashcards_route_maps_errors(monkeypatch):
    db = _new_session()

    async def _fake_create_set(*args, **kwargs):
        raise ValueError("Job not found")

    monkeypatch.setattr(study_router.flashcards, "create_study_card_set", _fake_create_set)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            study_router.create_flashcards(
                payload=study_router.FlashcardRequest(job_id=404, topic="python", num_cards=2),
                db=db,
            ),
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == "Job not found"


def test_review_flashcard_route_maps_not_found(monkeypatch):
    db = _new_session()

    def _fake_review(db_session, card_id: int, rating: str):
        raise ValueError("Study card not found")

    monkeypatch.setattr(study_router.flashcards, "review_study_card", _fake_review)

    with pytest.raises(HTTPException) as exc:
        study_router.review_flashcard(404, payload=study_router.ReviewRequest(rating="easy"), db=db)

    assert exc.value.status_code == 404
    assert exc.value.detail == "Study card not found"


@pytest.mark.asyncio
async def test_create_flashcards_route_happy_path(monkeypatch):
    db = _new_session()

    async def _fake_create_set(
        db_session,
        job_id: int | None,
        name: str | None,
        topic: str | None,
        num_cards: int,
        document_ids: list[int] | None = None,
        generate_per_section: bool = False,
    ):
        return {
            "card_set_id": 12,
            "cards": [
                {"id": 1, "card_set_id": 12, "front": "Q1", "back": "A1", "last_reviewed_at": None, "ease_factor": 2.5, "interval_days": 1},
                {"id": 2, "card_set_id": 12, "front": "Q2", "back": "A2", "last_reviewed_at": None, "ease_factor": 2.5, "interval_days": 1},
            ],
            "card_set": {
                "id": 12,
                "job_id": job_id,
                "parent_card_set_id": None,
                "name": name or "Untitled deck",
                "topic": topic,
                "created_at": None,
                "card_count": 2,
                "document_ids": document_ids or [],
                "document_count": len(document_ids or []),
            },
            "card_sets": [],
            "parent_card_set_id": None,
        }

    monkeypatch.setattr(study_router.flashcards, "create_study_card_set", _fake_create_set)

    response = await study_router.create_flashcards(
        payload=study_router.FlashcardRequest(job_id=5, topic="python", num_cards=2, name="Python Deck"),
        db=db,
    )
    assert response.card_set_id == 12
    assert len(response.cards) == 2
    assert response.cards[0].front == "Q1"
    assert response.card_set.name == "Python Deck"


def test_list_card_sets_route(monkeypatch):
    db = _new_session()

    def _fake_list_sets(db_session, limit: int = 20):
        return [
            {
                "id": 12,
                "job_id": None,
                "parent_card_set_id": None,
                "name": "System design",
                "topic": "system design",
                "created_at": "2026-03-19T12:00:00+00:00",
                "card_count": 10,
                "document_ids": [1, 2],
                "document_count": 2,
            }
        ]

    monkeypatch.setattr(study_router.flashcards, "list_study_card_sets", _fake_list_sets)

    response = study_router.list_card_sets(limit=20, db=db)
    assert len(response) == 1
    assert response[0].id == 12
    assert response[0].card_count == 10


def test_get_card_set_route(monkeypatch):
    db = _new_session()

    def _fake_get_set(db_session, card_set_id: int):
        return {
            "card_set_id": 22,
            "job_id": None,
            "parent_card_set_id": None,
            "name": "Behavioral deck",
            "topic": "behavioral",
            "created_at": "2026-03-19T12:00:00+00:00",
            "document_ids": [3],
            "document_count": 1,
            "cards": [
                {"id": 1, "front": "Q", "back": "A", "last_reviewed_at": None, "ease_factor": 2.5, "interval_days": 1}
            ],
        }

    monkeypatch.setattr(study_router.flashcards, "get_study_card_set_cards", _fake_get_set)

    response = study_router.get_card_set(card_set_id=22, db=db)
    assert response.card_set_id == 22
    assert len(response.cards) == 1
    assert response.name == "Behavioral deck"


def test_get_card_set_route_not_found(monkeypatch):
    db = _new_session()

    def _fake_get_set(db_session, card_set_id: int):
        raise ValueError("Study card set not found")

    monkeypatch.setattr(study_router.flashcards, "get_study_card_set_cards", _fake_get_set)

    with pytest.raises(HTTPException) as exc:
        study_router.get_card_set(card_set_id=999, db=db)
    assert exc.value.status_code == 404


def test_rename_card_set_route(monkeypatch):
    db = _new_session()

    def _fake_update(db_session, card_set_id: int, name: str):
        return StudyCardSet(id=card_set_id, name=name)

    def _fake_summary(db_session, card_set_id: int):
        return {
            "id": card_set_id,
            "job_id": None,
            "parent_card_set_id": None,
            "name": "Renamed deck",
            "topic": "topic",
            "created_at": None,
            "card_count": 3,
            "document_ids": [1],
            "document_count": 1,
        }

    monkeypatch.setattr(study_router.flashcards, "update_study_card_set_name", _fake_update)
    monkeypatch.setattr(study_router.flashcards, "get_study_card_set_summary", _fake_summary)

    response = study_router.rename_card_set(10, payload=study_router.StudyCardSetRenameRequest(name="Renamed deck"), db=db)
    assert response.id == 10
    assert response.name == "Renamed deck"


def test_delete_card_set_route(monkeypatch):
    db = _new_session()

    def _fake_delete(db_session, card_set_id: int):
        return None

    monkeypatch.setattr(study_router.flashcards, "delete_study_card_set", _fake_delete)
    response = study_router.delete_card_set(5, db=db)
    assert response["status"] == "deleted"


def test_get_mindmap_route_returns_payload(monkeypatch):
    db = _new_session()

    def _fake_get_cached(db_session, *, job_id: int, doc_id: int | None):
        return {
            "id": 1,
            "job_id": job_id,
            "doc_id": doc_id,
            "content_hash": "abc123",
            "graph": {
                "nodes": [{"id": "distributed_systems", "label": "Distributed Systems", "group": "system_design"}],
                "edges": [],
            },
            "node_sources": {"distributed_systems": "Distributed systems interview notes"},
            "created_at": "2026-03-19T12:00:00+00:00",
        }

    monkeypatch.setattr(study_router.mindmap, "get_cached_mind_map", _fake_get_cached)

    response = study_router.get_mindmap(job_id=9, doc_id=3, db=db)

    assert response.job_id == 9
    assert response.doc_id == 3
    assert response.cached is True
    assert response.graph.nodes[0].id == "distributed_systems"


@pytest.mark.asyncio
async def test_create_or_get_mindmap_route_happy_path(monkeypatch):
    db = _new_session()

    async def _fake_generate(db_session, *, job_id: int, doc_id: int | None):
        return (
            {
                "id": 11,
                "job_id": job_id,
                "doc_id": doc_id,
                "content_hash": "hash-1",
                "graph": {
                    "nodes": [{"id": "queues", "label": "Queues", "group": "system_design"}],
                    "edges": [{"source": "queues", "target": "queues", "label": "related_to"}],
                },
                "node_sources": {"queues": "Queue notes"},
                "created_at": "2026-03-19T12:00:00+00:00",
            },
            False,
        )

    monkeypatch.setattr(study_router.mindmap, "generate_or_get_mind_map", _fake_generate)

    response = await study_router.create_or_get_mindmap(
        payload=study_router.MindMapRequest(job_id=5, doc_id=None),
        db=db,
    )

    assert response.job_id == 5
    assert response.cached is False
    assert response.graph.nodes[0].id == "queues"
