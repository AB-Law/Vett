from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.interview import InterviewKnowledgeDocument
from app.models.score import Job
from app.models.study import MindMap
from app.services import mindmap


def _new_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine,
        tables=[
            Job.__table__,
            InterviewKnowledgeDocument.__table__,
            MindMap.__table__,
        ],
    )
    return sessionmaker(bind=engine)()


def _seed_job_and_doc(db):
    job = Job(id=77, title="Backend Engineer", company="Acme", description="Build distributed services.")
    document = InterviewKnowledgeDocument(
        id=301,
        owner_type="job",
        job_id=77,
        source_filename="notes.md",
        content_type="text/markdown",
        parsed_text=(
            "Distributed systems need consistency and availability.\n"
            "Queues smooth traffic spikes and support retries.\n"
            "Caching can reduce database load."
        ),
        status="embedded",
    )
    db.add(job)
    db.add(document)
    db.commit()
    return job, document


@pytest.mark.asyncio
async def test_generate_or_get_mind_map_caches_by_content_hash(monkeypatch):
    db = _new_session()
    _seed_job_and_doc(db)
    call_count = {"value": 0}

    async def _fake_generate(_contexts):
        call_count["value"] += 1
        return {
            "nodes": [{"id": "distributed_systems", "label": "Distributed Systems", "group": "system_design"}],
            "edges": [],
        }

    monkeypatch.setattr(mindmap, "_generate_graph_from_contexts", _fake_generate)

    first, cached_first = await mindmap.generate_or_get_mind_map(db, job_id=77, doc_id=301)
    second, cached_second = await mindmap.generate_or_get_mind_map(db, job_id=77, doc_id=301)

    assert cached_first is False
    assert cached_second is True
    assert first["content_hash"] == second["content_hash"]
    assert call_count["value"] == 1
    assert db.query(MindMap).count() == 1


@pytest.mark.asyncio
async def test_get_cached_mind_map_requires_current_content_hash(monkeypatch):
    db = _new_session()
    _, document = _seed_job_and_doc(db)

    async def _fake_generate(_contexts):
        return {
            "nodes": [{"id": "queues", "label": "Queues", "group": "architecture"}],
            "edges": [],
        }

    monkeypatch.setattr(mindmap, "_generate_graph_from_contexts", _fake_generate)

    generated, _ = await mindmap.generate_or_get_mind_map(db, job_id=77, doc_id=301)
    assert generated["graph"]["nodes"][0]["id"] == "queues"

    # Simulate document content update after cached graph creation.
    document.parsed_text = document.parsed_text + "\nNew section on idempotency."
    db.add(document)
    db.commit()

    with pytest.raises(ValueError, match="not found"):
        mindmap.get_cached_mind_map(db, job_id=77, doc_id=301)
