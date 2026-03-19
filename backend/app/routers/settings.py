"""Settings router – reads/writes the .env file directly for persistence."""
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import Literal, Optional
import os
import re
import logging
import time
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.practice import (
    PracticeQuestion,
    PracticeSessionQuestion,
    PracticeGeneration,
    QuestionCompany,
)
from ..models.interview import InterviewKnowledgeDocument
from ..services.llm import test_connection
from ..services.cv_parser import parse_cv
from ..services.interview_docs import (
    build_parser_signature,
    chunk_interview_text,
    chunk_integrity_stats,
    chunk_interview_text_meta,
    DOC_CHUNK_OVERLAP,
    DOC_CHUNK_WORDS,
    DOC_TABLE_NAME,
)

DOCUMENT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}

router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)

ENV_FILE = Path(os.getenv("ENV_FILE", "/app/.env"))


class LLMSettings(BaseModel):
    active_provider: str
    # Claude
    anthropic_api_key: Optional[str] = None
    claude_model: Optional[str] = None
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    # Azure
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_api_version: Optional[str] = None
    # Ollama
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None
    # LM Studio
    lm_studio_base_url: Optional[str] = None
    lm_studio_model: Optional[str] = None
    # General
    save_history: Optional[bool] = None
    default_export_format: Optional[str] = None
    tts_provider: Optional[Literal["native", "kokoro"]] = None
    voice_preferred_name: Optional[str] = None
    voice_rate: Optional[float] = None
    voice_pitch: Optional[float] = None


class SettingsResponse(BaseModel):
    active_provider: str
    claude_model: str
    openai_model: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str
    ollama_base_url: str
    ollama_model: str
    lm_studio_base_url: str
    lm_studio_model: str
    save_history: bool
    default_export_format: str
    tts_provider: Literal["native", "kokoro"]
    voice_preferred_name: str
    voice_rate: float
    voice_pitch: float
    # Masked keys – presence indicator only
    has_anthropic_key: bool
    has_openai_key: bool
    has_azure_key: bool


class InterviewKnowledgeDocumentResponse(BaseModel):
    id: int
    owner_type: str
    job_id: int | None
    source_filename: str
    content_type: str
    status: str
    error_message: str | None
    parser_version: str | None
    source_ref: str | None
    total_chunks: int
    embedded_chunks: int
    parsed_word_count: int
    created_at: str
    created_by_user_id: str | None


class InterviewKnowledgeDocumentProgressResponse(BaseModel):
    id: int
    owner_type: str
    job_id: int | None
    source_filename: str
    status: str
    total_chunks: int
    embedded_chunks: int
    progress_percent: float
    error_message: str | None
    parsed_word_count: int
    created_at: str
    created_by_user_id: str | None


class InterviewDocumentChunkPreview(BaseModel):
    index: int
    start_word: int
    end_word: int
    word_count: int
    text: str


class InterviewDocumentChunkDebugResponse(BaseModel):
    id: int
    owner_type: str
    job_id: int | None
    source_filename: str
    status: str
    chunk_size: int
    overlap: int
    parsed_word_count: int
    chunk_count: int
    covered_word_count: int
    coverage_ratio: float
    chunk_word_slots: int
    duplicate_slots: int
    has_gaps: bool
    chunks: list[InterviewDocumentChunkPreview]


class InterviewKnowledgeDocumentCleanupResponse(BaseModel):
    requested_filters: dict[str, object]
    dry_run: bool
    matched_documents: int
    deleted_documents: int
    deleted_chunks: int
    deleted_chunk_companies: int
    deleted_session_questions: int
    deleted_generation_rows: int


@router.get("/", response_model=SettingsResponse)
def get_settings_endpoint():
    s = get_settings()
    return SettingsResponse(
        active_provider=s.active_llm_provider,
        claude_model=s.claude_model,
        openai_model=s.openai_model,
        azure_openai_endpoint=s.azure_openai_endpoint,
        azure_openai_deployment=s.azure_openai_deployment,
        azure_openai_api_version=s.azure_openai_api_version,
        ollama_base_url=s.ollama_base_url,
        ollama_model=s.ollama_model,
        lm_studio_base_url=s.lm_studio_base_url,
        lm_studio_model=s.lm_studio_model,
        save_history=s.save_history,
        default_export_format=s.default_export_format,
        tts_provider=s.tts_provider if s.tts_provider in {"native", "kokoro"} else "kokoro",
        voice_preferred_name=s.voice_preferred_name,
        voice_rate=s.voice_rate,
        voice_pitch=s.voice_pitch,
        has_anthropic_key=bool(s.anthropic_api_key),
        has_openai_key=bool(s.openai_api_key),
        has_azure_key=bool(s.azure_openai_api_key),
    )


@router.post("/")
def update_settings(payload: LLMSettings):
    """Write changed values back to the .env file."""
    updates: dict[str, str] = {}

    if payload.active_provider:
        updates["ACTIVE_LLM_PROVIDER"] = payload.active_provider
    if payload.anthropic_api_key is not None:
        updates["ANTHROPIC_API_KEY"] = payload.anthropic_api_key
    if payload.claude_model:
        updates["CLAUDE_MODEL"] = payload.claude_model
    if payload.openai_api_key is not None:
        updates["OPENAI_API_KEY"] = payload.openai_api_key
    if payload.openai_model:
        updates["OPENAI_MODEL"] = payload.openai_model
    if payload.azure_openai_api_key is not None:
        updates["AZURE_OPENAI_API_KEY"] = payload.azure_openai_api_key
    if payload.azure_openai_endpoint:
        updates["AZURE_OPENAI_ENDPOINT"] = payload.azure_openai_endpoint
    if payload.azure_openai_deployment:
        updates["AZURE_OPENAI_DEPLOYMENT"] = payload.azure_openai_deployment
    if payload.azure_openai_api_version:
        updates["AZURE_OPENAI_API_VERSION"] = payload.azure_openai_api_version
    if payload.ollama_base_url:
        updates["OLLAMA_BASE_URL"] = payload.ollama_base_url
    if payload.ollama_model:
        updates["OLLAMA_MODEL"] = payload.ollama_model
    if payload.lm_studio_base_url:
        updates["LM_STUDIO_BASE_URL"] = payload.lm_studio_base_url
    if payload.lm_studio_model:
        updates["LM_STUDIO_MODEL"] = payload.lm_studio_model
    if payload.save_history is not None:
        updates["SAVE_HISTORY"] = "true" if payload.save_history else "false"
    if payload.default_export_format:
        updates["DEFAULT_EXPORT_FORMAT"] = payload.default_export_format
    if payload.tts_provider is not None:
        updates["TTS_PROVIDER"] = payload.tts_provider
    if payload.voice_preferred_name is not None:
        updates["VOICE_PREFERRED_NAME"] = payload.voice_preferred_name
    if payload.voice_rate is not None:
        updates["VOICE_RATE"] = str(payload.voice_rate)
    if payload.voice_pitch is not None:
        updates["VOICE_PITCH"] = str(payload.voice_pitch)

    _patch_env_file(updates)

    # Bust the settings cache so next read picks up changes
    get_settings.cache_clear()

    return {"ok": True, "updated": list(updates.keys())}


def _parse_document_upload(file: UploadFile) -> tuple[str, str]:
    filename = (file.filename or "interview-doc.txt").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in DOCUMENT_ALLOWED_EXTENSIONS:
        logger.warning("Rejected interview document upload: filename=%s extension=%s", filename, suffix)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(DOCUMENT_ALLOWED_EXTENSIONS))}",
        )
    logger.info("Accepted interview document upload filename=%s extension=%s", filename, suffix)
    return filename, suffix


def _serialize_interview_document(document: InterviewKnowledgeDocument) -> InterviewKnowledgeDocumentResponse:
    return InterviewKnowledgeDocumentResponse(
        id=document.id,
        owner_type=document.owner_type,
        job_id=document.job_id,
        source_filename=document.source_filename,
        content_type=document.content_type or "",
        status=document.status,
        error_message=document.error_message,
        parser_version=document.parser_version,
        source_ref=document.source_ref,
        total_chunks=int(document.total_chunks or 0),
        embedded_chunks=int(document.embedded_chunks or 0),
        parsed_word_count=int(document.parsed_word_count or 0),
        created_at=str(document.created_at),
        created_by_user_id=document.created_by_user_id,
    )


def _serialize_interview_document_progress(document: InterviewKnowledgeDocument) -> InterviewKnowledgeDocumentProgressResponse:
    total_chunks = int(document.total_chunks or 0)
    embedded_chunks = int(document.embedded_chunks or 0)
    if total_chunks <= 0:
        progress_percent = 0.0
    else:
        progress_percent = round(embedded_chunks / total_chunks * 100, 1)
    return InterviewKnowledgeDocumentProgressResponse(
        id=document.id,
        owner_type=document.owner_type,
        job_id=document.job_id,
        source_filename=document.source_filename,
        status=document.status,
        total_chunks=total_chunks,
        embedded_chunks=embedded_chunks,
        progress_percent=progress_percent,
        error_message=document.error_message,
        parsed_word_count=int(document.parsed_word_count or 0),
        created_at=str(document.created_at),
        created_by_user_id=document.created_by_user_id,
    )


def _collect_cleanup_document_ids(
    db: Session,
    *,
    owner_type: str | None,
    job_id: int | None,
    status: str | None,
    older_than_days: int | None,
) -> list[int]:
    if (
        older_than_days is None
        and owner_type is None
        and job_id is None
        and status is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Specify at least one filter to avoid deleting all documents.",
        )

    query = db.query(InterviewKnowledgeDocument)
    if owner_type is not None:
        query = query.filter(InterviewKnowledgeDocument.owner_type == owner_type)
    if job_id is not None:
        query = query.filter(InterviewKnowledgeDocument.job_id == job_id)
    if status is not None:
        query = query.filter(InterviewKnowledgeDocument.status == status)
    if older_than_days is not None:
        query = query.filter(
            InterviewKnowledgeDocument.created_at < datetime.utcnow() - timedelta(days=older_than_days),
        )
    return [doc.id for doc in query.all()]

def _cleanup_interview_documents(
    db: Session,
    document_ids: list[int],
) -> tuple[int, int, int, int, int]:
    if not document_ids:
        return 0, 0, 0, 0, 0
    practice_question_ids = [
        row[0]
        for row in db.query(PracticeQuestion.id)
        .filter(
            PracticeQuestion.source_table == DOC_TABLE_NAME,
            PracticeQuestion.source_id.in_(document_ids),
        )
        .all()
    ]

    deleted_chunk_companies = 0
    deleted_session_questions = 0
    deleted_generation_rows = 0
    deleted_chunks = 0
    if practice_question_ids:
        deleted_chunk_companies = (
            db.query(QuestionCompany)
            .filter(QuestionCompany.question_id.in_(practice_question_ids))
            .delete(synchronize_session=False)
        )
        deleted_session_questions = (
            db.query(PracticeSessionQuestion)
            .filter(PracticeSessionQuestion.question_id.in_(practice_question_ids))
            .delete(synchronize_session=False)
        )
        deleted_generation_rows = (
            db.query(PracticeGeneration)
            .filter(PracticeGeneration.source_question_id.in_(practice_question_ids))
            .delete(synchronize_session=False)
        )
        deleted_chunks = (
            db.query(PracticeQuestion)
            .filter(PracticeQuestion.id.in_(practice_question_ids))
            .delete(synchronize_session=False)
        )

    deleted_documents = (
        db.query(InterviewKnowledgeDocument)
        .filter(InterviewKnowledgeDocument.id.in_(document_ids))
        .delete(synchronize_session=False)
    )

    return (
        deleted_documents,
        deleted_chunks,
        deleted_chunk_companies,
        deleted_session_questions,
        deleted_generation_rows,
    )


def _build_document_chunk_debug(document: InterviewKnowledgeDocument, max_chunks: int) -> InterviewDocumentChunkDebugResponse:
    integrity = chunk_integrity_stats(document.parsed_text, DOC_CHUNK_WORDS, DOC_CHUNK_OVERLAP)
    chunks = chunk_interview_text_meta(document.parsed_text, DOC_CHUNK_WORDS, DOC_CHUNK_OVERLAP)
    chunk_samples = [
        InterviewDocumentChunkPreview(
            index=index,
            start_word=start,
            end_word=end,
            word_count=max(0, end - start),
            text=chunk,
        )
        for index, (start, end, chunk) in enumerate(chunks[:max_chunks])
    ]
    return InterviewDocumentChunkDebugResponse(
        id=document.id,
        owner_type=document.owner_type,
        job_id=document.job_id,
        source_filename=document.source_filename,
        status=document.status,
        chunk_size=DOC_CHUNK_WORDS,
        overlap=DOC_CHUNK_OVERLAP,
        parsed_word_count=int(integrity.get("parsed_word_count", 0)),
        chunk_count=int(integrity.get("chunk_count", 0)),
        covered_word_count=int(integrity.get("covered_word_count", 0)),
        coverage_ratio=float(integrity.get("coverage_ratio", 0.0)),
        chunk_word_slots=int(integrity.get("chunk_word_slots", 0)),
        duplicate_slots=int(integrity.get("duplicate_slots", 0)),
        has_gaps=bool(integrity.get("has_gaps", False)),
        chunks=chunk_samples,
    )


@router.get("/interview-documents", response_model=list[InterviewKnowledgeDocumentResponse])
def list_global_interview_documents(
    db: Annotated[Session, Depends(get_db)],
):
    docs = (
        db.query(InterviewKnowledgeDocument)
        .filter(InterviewKnowledgeDocument.owner_type == "global")
        .order_by(InterviewKnowledgeDocument.id.desc())
        .all()
    )
    return [_serialize_interview_document(document) for document in docs]


@router.delete("/interview-documents", response_model=InterviewKnowledgeDocumentCleanupResponse)
def cleanup_interview_documents(
    db: Annotated[Session, Depends(get_db)],
    older_than_days: int | None = Query(default=None, ge=0),
    owner_type: str | None = Query(default=None),
    job_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
):
    if (
        older_than_days is None
        and owner_type is None
        and job_id is None
        and status is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one filter to avoid deleting all documents.",
        )

    document_ids = _collect_cleanup_document_ids(
        db,
        owner_type=owner_type,
        job_id=job_id,
        status=status,
        older_than_days=older_than_days,
    )
    matched_documents = len(document_ids)
    if matched_documents == 0:
        return InterviewKnowledgeDocumentCleanupResponse(
            requested_filters={
                "older_than_days": older_than_days,
                "owner_type": owner_type,
                "job_id": job_id,
                "status": status,
            },
            dry_run=False,
            matched_documents=0,
            deleted_documents=0,
            deleted_chunks=0,
            deleted_chunk_companies=0,
            deleted_session_questions=0,
            deleted_generation_rows=0,
        )

    deleted_documents, deleted_chunks, deleted_chunk_companies, deleted_session_questions, deleted_generation_rows = (
        _cleanup_interview_documents(db, document_ids)
    )
    db.commit()
    return InterviewKnowledgeDocumentCleanupResponse(
        requested_filters={
            "older_than_days": older_than_days,
            "owner_type": owner_type,
            "job_id": job_id,
            "status": status,
        },
        dry_run=False,
        matched_documents=matched_documents,
        deleted_documents=deleted_documents,
        deleted_chunks=deleted_chunks,
        deleted_chunk_companies=deleted_chunk_companies,
        deleted_session_questions=deleted_session_questions,
        deleted_generation_rows=deleted_generation_rows,
    )


@router.get("/interview-documents/{document_id}/chunks", response_model=InterviewDocumentChunkDebugResponse)
def get_interview_document_chunk_debug(
    document_id: int,
    db: Annotated[Session, Depends(get_db)],
    max_chunks: int = Query(default=6, ge=1, le=50),
):
    document = db.query(InterviewKnowledgeDocument).filter(InterviewKnowledgeDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    return _build_document_chunk_debug(document, max_chunks=max_chunks)


@router.post("/interview-documents", response_model=InterviewKnowledgeDocumentResponse)
async def upload_global_interview_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    start_ts = time.perf_counter()
    filename, suffix = _parse_document_upload(file)
    data = await file.read()
    logger.info(
        "Starting global interview document parse filename=%s size=%s content_type=%s",
        filename,
        len(data),
        file.content_type,
    )

    status = "pending"
    error_message: str | None = None
    try:
        parse_start = time.perf_counter()
        parsed_text = parse_cv(data, filename)
        logger.info(
            "Global interview document parsed filename=%s parsed_chars=%s duration_ms=%.1f",
            filename,
            len(parsed_text),
            (time.perf_counter() - parse_start) * 1000,
        )
    except ValueError as exc:
        parsed_text = ""
        status = "failed"
        error_message = str(exc)
        logger.warning(
            "Global interview document failed validation filename=%s error=%s",
            filename,
            error_message,
        )
    except Exception as exc:
        parsed_text = ""
        status = "failed"
        error_message = f"Upload failed: {exc}"
        logger.exception("Global interview document parsing failed filename=%s", filename)

    if not parsed_text.strip() and status == "pending":
        status = "failed"
        error_message = "No text could be extracted from this document."
        logger.warning("Global interview document parse produced no text filename=%s", filename)

    document = InterviewKnowledgeDocument(
        owner_type="global",
        job_id=None,
        source_filename=filename,
        content_type=(suffix[1:] if suffix else "txt"),
        parsed_text=parsed_text,
        total_chunks=len(chunk_interview_text(parsed_text)),
        embedded_chunks=0,
        parsed_word_count=len((parsed_text or "").strip().split()),
        chunk_coverage_ratio=chunk_integrity_stats(parsed_text).get("coverage_ratio", 0.0),
        parser_version=build_parser_signature(),
        source_ref=build_parser_signature(),
        status=status,
        error_message=error_message,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(
        "Global interview document persisted id=%s filename=%s status=%s duration_ms=%.1f",
        document.id,
        filename,
        status,
        (time.perf_counter() - start_ts) * 1000,
    )
    return _serialize_interview_document(document)


@router.get("/interview-documents/progress", response_model=list[InterviewKnowledgeDocumentProgressResponse])
def list_global_interview_document_progress(
    db: Annotated[Session, Depends(get_db)],
    owner_type: str | None = Query(None),
    job_id: int | None = Query(None),
):
    q = db.query(InterviewKnowledgeDocument)
    if owner_type:
        q = q.filter(InterviewKnowledgeDocument.owner_type == owner_type)
    if job_id is not None:
        q = q.filter(InterviewKnowledgeDocument.job_id == job_id)
    docs = q.order_by(InterviewKnowledgeDocument.id.desc()).all()
    return [_serialize_interview_document_progress(document) for document in docs]


@router.post("/test-connection")
async def test_llm_connection():
    """Always returns 200 so the frontend can display the actual error."""
    result = await test_connection()
    return result


@router.get("/embedding-progress")
def embedding_progress(db: Annotated[Session, Depends(get_db)]):
    """Return how many practice questions have been embedded vs total."""
    total = db.query(PracticeQuestion).filter(PracticeQuestion.is_active.is_(True)).count()
    embedded = (
        db.query(PracticeQuestion)
        .filter(PracticeQuestion.is_active.is_(True), PracticeQuestion.embedding.isnot(None))
        .count()
    )
    pct = round(embedded / total * 100, 1) if total > 0 else 0.0
    return {"total": total, "embedded": embedded, "percent": pct}


def _patch_env_file(updates: dict[str, str]):
    """Update or add key=value pairs in the .env file."""
    if not ENV_FILE.exists():
        lines = []
    else:
        lines = ENV_FILE.read_text().splitlines()

    for key, value in updates.items():
        replaced = False
        for i, line in enumerate(lines):
            if re.match(rf"^\s*{re.escape(key)}\s*=", line):
                lines[i] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n")
