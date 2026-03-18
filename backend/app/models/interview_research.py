from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.sql import func

from ..database import Base


class InterviewResearchSession(Base):
    __tablename__ = "interview_research_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    role = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="initialized")
    stage = Column(String(40), nullable=False, default="initialized")
    question_bank = Column(JSON, nullable=True)
    source_urls = Column(JSON, nullable=True)
    fallback_used = Column(Boolean, default=False, nullable=False)
    failure_reason = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    processing_ms = Column(Integer, nullable=True)
