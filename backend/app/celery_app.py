"""Phase 2 – Celery application (scaffolded).

To enable: uncomment celery_worker in docker-compose.yml
"""
from celery import Celery
from .config import get_settings

settings = get_settings()

celery_app = Celery(
    "vett",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.scraper"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
