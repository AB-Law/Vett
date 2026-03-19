from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.interview import InterviewKnowledgeDocument
from app.models.score import Job
from app.models.study import StudyCard, StudyCardSet, StudyCardSetDocument
from app.services import flashcards


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
        ],
    )
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_create_study_card_set_saves_cards(monkeypatch):
    db = _new_session()
    db.add(Job(id=21, title="Backend Engineer", company="Acme", description="Build interview tools."))
    db.commit()

    async def _mock_embedding(_: str):
        return [0.1, 0.2, 0.3]

    def _mock_context(db_session, query_vector, job_id: int, document_ids, scope_limit: int, scope_overlap: int):
        return [{"filename": "notes.txt", "snippet": "Binary search tree basics and BFS"}]

    async def _mock_cards_from_context_with_diagnostics(
        topic: str | None,
        contexts: list[dict[str, str]],
        num_cards: int,
        *,
        job_title: str | None,
        job_company: str | None,
    ):
        cards = [
            {"front": "What is DFS?", "back": "Depth-first search."},
            {"front": "When to use BFS?", "back": "When exploring by levels."},
        ]
        diagnostics = {
            "requested_cards": num_cards,
            "llm_cards_parsed": min(len(cards), num_cards),
            "deduped_out": 0,
            "fallback_cards_used": 0,
            "fallback_used": False,
        }
        return cards[:num_cards], diagnostics

    monkeypatch.setattr(flashcards.llm_service, "generate_embedding", _mock_embedding)
    monkeypatch.setattr(flashcards.interview_docs, "fetch_context_from_interview_documents", _mock_context)
    monkeypatch.setattr(flashcards, "_generate_cards_from_context_with_diagnostics", _mock_cards_from_context_with_diagnostics)

    result = await flashcards.create_study_card_set(
        db,
        job_id=21,
        name="Backend prep",
        topic="graphs",
        num_cards=2,
    )

    assert result["card_set_id"] is not None
    assert len(result["cards"]) == 2
    assert result["cards"][0]["front"] == "What is DFS?"
    assert result["card_set"]["name"] == "Backend prep"
    assert db.query(StudyCard).count() == 2
    assert db.query(StudyCardSet).count() == 1


def test_review_study_card_adjusts_spaced_repetition():
    db = _new_session()
    db.add(Job(id=22, title="Senior Engineer", company="Bright"))
    db.flush()
    card_set = StudyCardSet(job_id=22, topic="algorithms")
    db.add(card_set)
    db.flush()
    card = StudyCard(
        card_set_id=card_set.id,
        front="What is a hash map?",
        back="A key-value data structure.",
        ease_factor=2.5,
        interval_days=1,
    )
    db.add(card)
    db.commit()

    easy_card = flashcards.review_study_card(db, card.id, "easy")
    assert easy_card.interval_days == 6
    assert easy_card.ease_factor == 2.6
    assert easy_card.last_reviewed_at is not None

    hard_card = flashcards.review_study_card(db, card.id, "hard")
    assert hard_card.interval_days == 3
    assert hard_card.ease_factor < 2.6


def test_review_study_card_rejects_missing_card():
    db = _new_session()

    with pytest.raises(ValueError, match="not found"):
        flashcards.review_study_card(db, card_id=999, rating="easy")


def test_list_and_get_study_card_sets():
    db = _new_session()
    db.add(Job(id=33, title="Platform Engineer", company="Acme"))
    db.flush()
    set_a = StudyCardSet(job_id=33, topic="systems", name="Systems deck")
    db.add(set_a)
    db.flush()
    set_b = StudyCardSet(job_id=33, topic="behavioral", name="Behavioral deck")
    db.add(set_b)
    db.flush()
    document = InterviewKnowledgeDocument(
        owner_type="global",
        source_filename="notes.txt",
        content_type="text/plain",
        parsed_text="Heading\nDistributed systems basics",
        status="ready",
    )
    db.add(document)
    db.flush()
    db.add(StudyCardSetDocument(card_set_id=set_b.id, document_id=document.id))
    db.add_all(
        [
            StudyCard(card_set_id=set_a.id, front="Q1", back="A1", ease_factor=2.5, interval_days=1),
            StudyCard(card_set_id=set_a.id, front="Q2", back="A2", ease_factor=2.5, interval_days=1),
            StudyCard(card_set_id=set_b.id, front="Q3", back="A3", ease_factor=2.5, interval_days=1),
        ]
    )
    db.commit()

    summaries = flashcards.list_study_card_sets(db, limit=10)
    assert len(summaries) == 2
    assert summaries[0]["id"] == set_b.id
    assert summaries[0]["card_count"] == 1
    assert summaries[0]["document_count"] == 1
    assert summaries[1]["card_count"] == 2

    loaded_set = flashcards.get_study_card_set_cards(db, card_set_id=set_a.id)
    assert loaded_set["card_set_id"] == set_a.id
    assert len(loaded_set["cards"]) == 2


@pytest.mark.asyncio
async def test_create_study_card_set_rejects_missing_job():
    db = _new_session()

    with pytest.raises(ValueError, match="Job not found"):
        await flashcards.create_study_card_set(db, job_id=999)


@pytest.mark.asyncio
async def test_create_study_card_set_persists_document_ids(monkeypatch):
    db = _new_session()
    doc = InterviewKnowledgeDocument(
        owner_type="global",
        source_filename="algorithms.md",
        content_type="text/markdown",
        parsed_text="Trees and graphs notes",
        status="ready",
    )
    db.add(doc)
    db.commit()

    async def _mock_embedding(_: str):
        return [0.1, 0.2, 0.3]

    def _mock_context(db_session, query_vector, job_id, document_ids, scope_limit, scope_overlap):
        assert document_ids == [doc.id]
        return [{"filename": "algorithms.md", "snippet": "Graph traversal"}]

    async def _mock_cards_from_context_with_diagnostics(
        topic: str | None,
        contexts: list[dict[str, str]],
        num_cards: int,
        *,
        job_title: str | None,
        job_company: str | None,
    ):
        cards = [{"front": "What is BFS?", "back": "Breadth-first search."}]
        diagnostics = {
            "requested_cards": num_cards,
            "llm_cards_parsed": min(len(cards), num_cards),
            "deduped_out": 0,
            "fallback_cards_used": 0,
            "fallback_used": False,
        }
        return cards[:num_cards], diagnostics

    monkeypatch.setattr(flashcards.llm_service, "generate_embedding", _mock_embedding)
    monkeypatch.setattr(flashcards.interview_docs, "fetch_context_from_interview_documents", _mock_context)
    monkeypatch.setattr(flashcards, "_generate_cards_from_context_with_diagnostics", _mock_cards_from_context_with_diagnostics)

    result = await flashcards.create_study_card_set(
        db,
        document_ids=[doc.id],
        topic="graphs",
        name="Graph prep",
        num_cards=1,
    )

    assert result["card_set"]["name"] == "Graph prep"
    assert result["card_set"]["document_ids"] == [doc.id]
    assert db.query(StudyCardSetDocument).count() == 1


def test_update_and_delete_study_card_set():
    db = _new_session()
    deck = StudyCardSet(job_id=None, topic="behavioral", name="Old name")
    db.add(deck)
    db.flush()
    db.add(StudyCard(card_set_id=deck.id, front="Q1", back="A1", ease_factor=2.5, interval_days=1))
    db.commit()

    updated = flashcards.update_study_card_set_name(db, card_set_id=deck.id, name="New name")
    assert updated.name == "New name"

    flashcards.delete_study_card_set(db, card_set_id=deck.id)
    assert db.query(StudyCardSet).count() == 0
    assert db.query(StudyCard).count() == 0


@pytest.mark.asyncio
async def test_generate_per_section_creates_multiple_sets(monkeypatch):
    db = _new_session()
    doc = InterviewKnowledgeDocument(
        owner_type="global",
        source_filename="system-design.pdf",
        content_type="application/pdf",
        parsed_text="Section 1\nCaching notes\nSection 2\nQueues notes",
        status="ready",
    )
    db.add(doc)
    db.commit()

    def _mock_split_sections(_: str):
        return [
            {"title": "Caching", "text": "Use redis cache."},
            {"title": "Queues", "text": "Use message queues."},
        ]

    async def _mock_cards_from_context_with_diagnostics(
        topic: str | None,
        contexts: list[dict[str, str]],
        num_cards: int,
        *,
        job_title: str | None,
        job_company: str | None,
    ):
        cards = [{"front": "Question", "back": contexts[0]["snippet"]}]
        diagnostics = {
            "requested_cards": num_cards,
            "llm_cards_parsed": min(len(cards), num_cards),
            "deduped_out": 0,
            "fallback_cards_used": 0,
            "fallback_used": False,
        }
        return cards[:num_cards], diagnostics

    monkeypatch.setattr(flashcards.interview_docs, "split_document_into_sections", _mock_split_sections)
    monkeypatch.setattr(flashcards, "_generate_cards_from_context_with_diagnostics", _mock_cards_from_context_with_diagnostics)

    result = await flashcards.create_study_card_set(
        db,
        document_ids=[doc.id],
        name="System Design Q3",
        num_cards=1,
        generate_per_section=True,
    )

    assert len(result["card_sets"]) == 2
    assert db.query(StudyCardSet).count() == 2


@pytest.mark.asyncio
async def test_large_targets_are_split_into_child_decks(monkeypatch):
    db = _new_session()
    doc = InterviewKnowledgeDocument(
        owner_type="global",
        source_filename="distributed-systems.pdf",
        content_type="application/pdf",
        parsed_text="Large content for retrieval and generation",
        status="ready",
    )
    db.add(doc)
    db.commit()

    async def _mock_embedding(_: str):
        return [0.1, 0.2, 0.3]

    def _mock_context(db_session, query_vector, job_id, document_ids, scope_limit, scope_overlap):
        return [{"filename": "distributed-systems.pdf", "snippet": "Consensus, partitioning, replication"}]

    call_index = {"value": 0}

    async def _mock_cards_from_context_with_diagnostics(
        topic: str | None,
        contexts: list[dict[str, str]],
        num_cards: int,
        *,
        job_title: str | None,
        job_company: str | None,
    ):
        call_index["value"] += 1
        offset = (call_index["value"] - 1) * 1000
        cards = [{"front": f"Q{offset + i}", "back": f"A{offset + i}"} for i in range(1, num_cards + 1)]
        diagnostics = {
            "requested_cards": num_cards,
            "llm_cards_parsed": len(cards),
            "deduped_out": 0,
            "fallback_cards_used": 0,
            "fallback_used": False,
        }
        return cards, diagnostics

    monkeypatch.setattr(flashcards.llm_service, "generate_embedding", _mock_embedding)
    monkeypatch.setattr(flashcards.interview_docs, "fetch_context_from_interview_documents", _mock_context)
    monkeypatch.setattr(flashcards, "_generate_cards_from_context_with_diagnostics", _mock_cards_from_context_with_diagnostics)

    result = await flashcards.create_study_card_set(
        db,
        document_ids=[doc.id],
        topic="distributed systems",
        name="Systems Mega Deck",
        num_cards=120,
    )

    assert result["parent_card_set_id"] is not None
    assert len(result["card_sets"]) == 3
    assert all(item["card_set"]["card_count"] <= 50 for item in result["card_sets"])
    assert db.query(StudyCardSet).count() == 4  # 1 parent + 3 child decks
