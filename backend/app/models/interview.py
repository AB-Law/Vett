"""Interview knowledge base upload artifacts."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.sql import func as sql_func
from ..database import Base


class InterviewKnowledgeDocument(Base):
    __tablename__ = "interview_knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)
    owner_type = Column(String(16), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), index=True)
    source_filename = Column(String(255), nullable=False)
    content_type = Column(String(80))
    parsed_text = Column(Text, nullable=False)
    parser_version = Column(String(64))
    source_ref = Column(String(120))
    status = Column(String(24), default="pending", nullable=False, index=True)
    error_message = Column(Text)
    created_by_user_id = Column(String(120))
    created_at = Column(DateTime(timezone=True), server_default=sql_func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=sql_func.now())
