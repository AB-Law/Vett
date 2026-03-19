import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.interview import InterviewKnowledgeDocument
from app.routers import settings as settings_router
from app.services.interview_docs import chunk_integrity_stats


def _new_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[InterviewKnowledgeDocument.__table__])
    return sessionmaker(bind=engine)()


class _FakeQuery:
    def __init__(self, rows: int):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def count(self) -> int:
        return self._rows


class _FakeDb:
    def __init__(self, rows: list[int]):
        self._rows = rows
        self._index = 0

    def query(self, model):
        if model is settings_router.PracticeQuestion:
            value = self._rows[self._index]
            self._index += 1
            return _FakeQuery(value)
        raise AssertionError("Unexpected query model: %r" % model)


def test_list_global_interview_document_progress_returns_percent():
    db = _new_session()
    db.add_all(
        [
            InterviewKnowledgeDocument(
                owner_type="global",
                job_id=None,
                source_filename="doc-a.txt",
                content_type="text/plain",
                parsed_text="chunked text sample for testing",
                total_chunks=4,
                embedded_chunks=1,
                status="processing",
                error_message=None,
                parsed_word_count=6,
            ),
            InterviewKnowledgeDocument(
                owner_type="job",
                job_id=12,
                source_filename="doc-b.txt",
                content_type="text/plain",
                parsed_text="another doc",
                total_chunks=2,
                embedded_chunks=2,
                status="embedded",
                error_message=None,
                parsed_word_count=2,
            ),
        ]
    )
    db.commit()

    docs = settings_router.list_global_interview_document_progress(db=db, owner_type=None, job_id=None)

    by_filename = {doc.source_filename: doc.progress_percent for doc in docs}
    assert by_filename["doc-a.txt"] == 25.0
    assert by_filename["doc-b.txt"] == 100.0


def test_list_global_interview_document_progress_filters_owner_and_job():
    db = _new_session()
    db.add_all(
        [
            InterviewKnowledgeDocument(
                owner_type="global",
                job_id=None,
                source_filename="global.txt",
                content_type="text/plain",
                parsed_text="global document",
                total_chunks=1,
                embedded_chunks=1,
                status="embedded",
                error_message=None,
                parsed_word_count=2,
            ),
            InterviewKnowledgeDocument(
                owner_type="job",
                job_id=9,
                source_filename="job.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                parsed_text="job document",
                total_chunks=1,
                embedded_chunks=0,
                status="pending",
                error_message=None,
                parsed_word_count=2,
            ),
            InterviewKnowledgeDocument(
                owner_type="job",
                job_id=10,
                source_filename="other-job.txt",
                content_type="text/plain",
                parsed_text="other job",
                total_chunks=1,
                embedded_chunks=0,
                status="pending",
                error_message=None,
                parsed_word_count=2,
            ),
        ]
    )
    db.commit()

    global_docs = settings_router.list_global_interview_document_progress(db=db, owner_type="global", job_id=None)
    job_9_docs = settings_router.list_global_interview_document_progress(db=db, owner_type="job", job_id=9)

    assert len(global_docs) == 1
    assert global_docs[0].source_filename == "global.txt"
    assert len(job_9_docs) == 1
    assert job_9_docs[0].source_filename == "job.docx"


def test_embedding_progress_endpoint_uses_is_active_and_embedding_presence():
    result = settings_router.embedding_progress(
        db=_FakeDb([3, 1]),
    )

    assert result == {"total": 3, "embedded": 1, "percent": 33.3}


def test_chunk_integrity_stats_for_overlap():
    result = chunk_integrity_stats(
        "alpha beta gamma delta epsilon zeta eta theta",
        chunk_size=4,
        overlap=1,
    )

    assert result["parsed_word_count"] == 8
    assert result["chunk_count"] == 3
    assert result["covered_word_count"] == 8
    assert result["coverage_ratio"] == 100.0
    assert result["chunk_word_slots"] == 10
    assert result["duplicate_slots"] == 2
    assert result["has_gaps"] is False


def test_chunk_integrity_stats_handles_empty_text():
    result = chunk_integrity_stats("", chunk_size=4, overlap=1)

    assert result["parsed_word_count"] == 0
    assert result["chunk_count"] == 0
    assert result["covered_word_count"] == 0
    assert result["coverage_ratio"] == 0.0
