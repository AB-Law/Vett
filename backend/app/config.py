from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "sqlite:////data/vett.db"
    upload_dir: str = "/data/uploads"
    env_file: str = ".env"

    # LLM
    active_llm_provider: str = "ollama"

    anthropic_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20241022"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-01"

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2"

    lm_studio_base_url: str = "http://host.docker.internal:1234/v1"
    lm_studio_model: str = "local-model"

    save_history: bool = True
    default_export_format: str = "json"

    # Research infra (self-hosted SearXNG)
    searxng_enabled: bool = True
    searxng_base_url: str = "http://searxng:8080"
    searxng_timeout_seconds: int = 8
    interview_research_timeout_seconds: int = 20

    # Phase 2
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    # Practice question sync
    practice_repo_url: str = "https://github.com/AB-Law/leetcode-companywise-interview-questions.git"
    practice_repo_branch: str = "master"
    practice_repo_path: str = "/data/practice_repo"
    practice_github_token: str = ""
    practice_default_limit: int = 8
    practice_restrict_dup_window_minutes: int = 60
    practice_embedding_model: str = "text-embedding-3-small"
    practice_embedding_dim: int = 1536  # set to 1024 for nomic-embed-text / mxbai-embed-large

    # Local-context tooling
    local_context_environment: str = "dev"

    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
