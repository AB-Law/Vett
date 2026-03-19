"""Shared helpers for interview knowledge base chunking and context retrieval."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from sqlalchemy import bindparam, text
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


def _word_chunks_with_meta(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    words = text.split()
    if not words:
        return []
    stride = max(1, chunk_size - max(0, overlap))
    chunks: list[tuple[int, int, str]] = []
    for start in range(0, len(words), stride):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append((start, min(end, len(words)), chunk))
        if end >= len(words):
            break
    return chunks


def chunk_interview_text(text: str, chunk_size: int = DOC_CHUNK_WORDS, overlap: int = DOC_CHUNK_OVERLAP) -> list[str]:
    return _word_chunks((text or "").strip(), chunk_size=max(1, chunk_size), overlap=max(0, overlap))


def chunk_interview_text_meta(
    text: str,
    chunk_size: int = DOC_CHUNK_WORDS,
    overlap: int = DOC_CHUNK_OVERLAP,
) -> list[tuple[int, int, str]]:
    return _word_chunks_with_meta(
        (text or "").strip(),
        chunk_size=max(1, chunk_size),
        overlap=max(0, overlap),
    )


def chunk_integrity_stats(
    text: str,
    chunk_size: int = DOC_CHUNK_WORDS,
    overlap: int = DOC_CHUNK_OVERLAP,
) -> dict[str, object]:
    words = (text or "").strip().split()
    if not words:
        return {
            "parsed_word_count": 0,
            "chunk_count": 0,
            "covered_word_count": 0,
            "coverage_ratio": 0.0,
            "chunk_word_slots": 0,
            "has_gaps": False,
            "duplicate_slots": 0,
        }

    chunks = chunk_interview_text_meta((text or "").strip(), chunk_size=chunk_size, overlap=overlap)
    covered_indices: set[int] = set()
    for start, end, _ in chunks:
        for index in range(start, end):
            covered_indices.add(index)

    coverage_ratio = round(len(covered_indices) / len(words) * 100, 1)
    chunk_word_slots = sum(max(0, end - start) for start, end, _ in chunks)
    duplicate_slots = max(0, chunk_word_slots - len(covered_indices))
    return {
        "parsed_word_count": len(words),
        "chunk_count": len(chunks),
        "covered_word_count": len(covered_indices),
        "coverage_ratio": coverage_ratio,
        "chunk_word_slots": chunk_word_slots,
        "has_gaps": len(covered_indices) < len(words),
        "duplicate_slots": duplicate_slots,
    }


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
    document_ids: list[int] | None = None,
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
        "document_ids": [int(doc_id) for doc_id in (document_ids or []) if int(doc_id) > 0],
    }
    if job_id:
        job_filter = "(pq.scope_type = 'global' OR (pq.scope_type = 'job' AND pq.scope_job_id = :job_id))"

    document_filter_sql = ""
    if parameters["document_ids"]:
        document_filter_sql = " AND pq.source_id IN :document_ids"

    statement = text(
            """
            SELECT pq.id, pq.source_id, pq.source_window
            FROM practice_questions pq
            WHERE
              pq.source_table = :source_table
              AND pq.embedding IS NOT NULL
              AND ("""
            + job_filter
            + ")"
            + document_filter_sql
            + """
            ORDER BY pq.embedding <=> CAST(:vector_literal AS vector)
            LIMIT :scope_limit
            """
    )
    if parameters["document_ids"]:
        statement = statement.bindparams(bindparam("document_ids", expanding=True))

    rows = db.execute(
        statement,
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


def _is_heading_line(line: str) -> bool:
    text_value = (line or "").strip()
    if not text_value or len(text_value) > 140:
        return False
    if text_value.lower().startswith(("http://", "https://")):
        return False
    if text_value.endswith((".", "?", "!")):
        return False
    if re.match(r"^\d+(\.\d+)*[\)\.: -].+", text_value):
        return True
    if re.match(r"^(chapter|section|part)\s+\d+[\s:\.-]", text_value, flags=re.IGNORECASE):
        return True
    alpha_count = sum(1 for char in text_value if char.isalpha())
    upper_count = sum(1 for char in text_value if char.isupper())
    if alpha_count >= 4 and upper_count / max(1, alpha_count) > 0.8:
        return True
    words = text_value.split()
    if 1 <= len(words) <= 8 and text_value == text_value.title():
        return True
    return False


def split_document_into_sections(text: str) -> list[dict[str, str]]:
    raw_text = (text or "").strip()
    if not raw_text:
        return []

    lines = [line.rstrip() for line in raw_text.splitlines()]
    sections: list[dict[str, str]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    def _flush_section() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"title": current_title, "text": body})

    for line in lines:
        stripped = line.strip()
        if _is_heading_line(stripped):
            _flush_section()
            current_title = stripped[:120]
            current_lines = []
            continue
        current_lines.append(line)

    _flush_section()
    if not sections:
        return [{"title": "Overview", "text": raw_text}]
    return sections


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
