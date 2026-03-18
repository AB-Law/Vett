"""Settings router – reads/writes the .env file directly for persistence."""
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import os
import re

from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.practice import PracticeQuestion
from ..models.interview import InterviewKnowledgeDocument
from ..services.llm import test_connection
from ..services.cv_parser import parse_cv
from ..services.interview_docs import build_parser_signature

DOCUMENT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}

router = APIRouter(prefix="/settings", tags=["settings"])

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
    created_at: str
    created_by_user_id: str | None


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

    _patch_env_file(updates)

    # Bust the settings cache so next read picks up changes
    get_settings.cache_clear()

    return {"ok": True, "updated": list(updates.keys())}


def _parse_document_upload(file: UploadFile) -> tuple[str, str]:
    filename = (file.filename or "interview-doc.txt").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in DOCUMENT_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(DOCUMENT_ALLOWED_EXTENSIONS))}",
        )
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
        created_at=str(document.created_at),
        created_by_user_id=document.created_by_user_id,
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


@router.post("/interview-documents", response_model=InterviewKnowledgeDocumentResponse)
async def upload_global_interview_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    filename, suffix = _parse_document_upload(file)
    data = await file.read()

    status = "pending"
    error_message: str | None = None
    try:
        parsed_text = parse_cv(data, filename)
    except ValueError as exc:
        parsed_text = ""
        status = "failed"
        error_message = str(exc)
    except Exception as exc:
        parsed_text = ""
        status = "failed"
        error_message = f"Upload failed: {exc}"

    if not parsed_text.strip() and status == "pending":
        status = "failed"
        error_message = "No text could be extracted from this document."

    document = InterviewKnowledgeDocument(
        owner_type="global",
        job_id=None,
        source_filename=filename,
        content_type=(suffix[1:] if suffix else "txt"),
        parsed_text=parsed_text,
        parser_version=build_parser_signature(),
        source_ref=build_parser_signature(),
        status=status,
        error_message=error_message,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return _serialize_interview_document(document)


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
