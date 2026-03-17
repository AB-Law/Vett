"""Settings router – reads/writes the .env file directly for persistence."""
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import os
import re

from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.practice import PracticeQuestion
from ..services.llm import test_connection

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
