"""Shared helpers for interview knowledge base chunking and context retrieval."""

from __future__ import annotations

import hashlib
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.interview import InterviewKnowledgeDocument
from ..models.practice import PracticeQuestion


DOC_TABLE_NAME = "interview_knowledge_documents"
DEFAULT_DOCUMENT_SOURCE_WINDOW = "interview-docs"

DOC_CHUNK_WORDS = 240
DOC_CHUNK_OVERLAP = 40


def _word_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    stride = max(1, chunk_size - max(0, overlap))
    chunks: list[str] = []
    for start in range(0, len(words), stride):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def chunk_interview_text(text: str, chunk_size: int = DOC_CHUNK_WORDS, overlap: int = DOC_CHUNK_OVERLAP) -> list[str]:
    return _word_chunks((text or "").strip(), chunk_size=max(1, chunk_size), overlap=max(0, overlap))


def make_source_window(chunk_index: int) -> str:
    return f"chunk:{chunk_index}"


def chunk_index_from_source_window(source_window: str | None) -> int | None:
    if not source_window:
        return None
    if not source_window.startswith("chunk:"):
        return None
    try:
        return int(source_window.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def chunk_for_document(document: InterviewKnowledgeDocument, source_window: str | None, *, chunk_size: int = DOC_CHUNK_WORDS) -> str | None:
    chunk_index = chunk_index_from_source_window(source_window)
    if chunk_index is None:
        return None
    chunks = chunk_interview_text(document.parsed_text, chunk_size=chunk_size)
    if chunk_index < 0 or chunk_index >= len(chunks):
        return None
    return chunks[chunk_index]


def build_doc_chunk_title(document: InterviewKnowledgeDocument, chunk_index: int) -> str:
    checksum = (document.source_ref or "").strip() or "nodigest"
    return f"interview-doc-{document.id}-chunk-{chunk_index}-{checksum[:8]}"


def build_parser_signature() -> str:
    return "cv_parser:v1"


def dedupe_error_signature(error_text: str | None) -> str | None:
    if not error_text:
        return None
    text_content = str(error_text).strip()
    if not text_content:
        return None
    return hashlib.sha1(text_content.encode("utf-8", errors="replace")).hexdigest()[:10]


def fetch_context_from_interview_documents(
    db: Session,
    query_vector: list[float],
    *,
    job_id: int | None,
    scope_limit: int = 6,
    scope_overlap: int = 2,
) -> list[dict[str, str]]:
    if not query_vector:
        return []

    vector_literal = "[" + ",".join(str(v) for v in query_vector) + "]"
    job_filter = "pq.scope_type = 'global'"
    parameters = {
        "job_id": int(job_id or 0),
        "scope_limit": int(scope_limit),
        "vector_literal": vector_literal,
    }
    if job_id:
        job_filter = "(pq.scope_type = 'global' OR (pq.scope_type = 'job' AND pq.scope_job_id = :job_id))"

    rows = db.execute(
        text(
            """
            SELECT pq.id, pq.source_id, pq.source_window
            FROM practice_questions pq
            WHERE
              pq.source_table = :source_table
              AND pq.embedding IS NOT NULL
              AND ("""
            + job_filter
            + ")"
            + """
            ORDER BY pq.embedding <=> CAST(:vector_literal AS vector)
            LIMIT :scope_limit
            """
        ),
        {
            **parameters,
            "source_table": DOC_TABLE_NAME,
            "vector_literal": vector_literal,
        },
    ).fetchall()

    if not rows:
        return []

    doc_ids = [row[1] for row in rows if row[1] is not None]
    documents_by_id = {
        document.id: document
        for document in db.query(InterviewKnowledgeDocument)
        .filter(InterviewKnowledgeDocument.id.in_(doc_ids))
        .all()
    }

    contexts: list[dict[str, str]] = []
    for _, source_id, source_window in rows:
        document = documents_by_id.get(int(source_id)) if source_id is not None else None
        if document is None:
            continue
        snippet = chunk_for_document(document, source_window)
        if not snippet:
            continue
        contexts.append(
            {
                "filename": document.source_filename,
                "snippet": snippet,
                "owner_type": document.owner_type,
                "chunk_id": f"{source_id}:{source_window}",
            }
        )
        if len(contexts) >= scope_limit + scope_overlap:
            break

    return contexts[:scope_limit]


def serialize_docs_for_response(documents: Iterable[InterviewKnowledgeDocument]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for document in documents:
        output.append(
            {
                "id": int(document.id),
                "owner_type": document.owner_type,
                "job_id": document.job_id,
                "source_filename": document.source_filename,
                "content_type": document.content_type,
                "status": document.status,
                "error_message": document.error_message,
                "parser_version": document.parser_version,
                "source_ref": document.source_ref,
                "created_at": str(document.created_at) if document.created_at else "",
                "created_by_user_id": document.created_by_user_id,
            }
        )
    return output
