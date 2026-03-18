"""ORM models for adaptive interview practice workflows."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from ..database import Base
from ..config import get_settings


class PracticeQuestion(Base):
    __tablename__ = "practice_questions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False, unique=True)
    url = Column(Text, nullable=True)
    difficulty = Column(String(64), index=True)
    acceptance = Column(String(64))
    embedding = Column(Vector(get_settings().practice_embedding_dim), nullable=True)
    embedding_model = Column(String(128))
    is_active = Column(Boolean, default=True, nullable=False)
    source_commit = Column(String(64))
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    scope_type = Column(String(16), index=True)
    scope_job_id = Column(Integer, ForeignKey("jobs.id"), index=True)
    source_table = Column(String(64), index=True)
    source_id = Column(Integer, ForeignKey("interview_knowledge_documents.id"), index=True)
    source_window = Column(String(64), index=True)


class QuestionCompany(Base):
    __tablename__ = "question_companies"
    __table_args__ = (
        UniqueConstraint("question_id", "company_slug", name="uq_question_company"),
    )

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("practice_questions.id"), index=True, nullable=False)
    company_slug = Column(String(255), index=True, nullable=False)
    frequency = Column(Float, nullable=True)
    source_window = Column(String(64), nullable=False, default="all")
    is_active = Column(Boolean, default=True, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PracticeSession(Base):
    __tablename__ = "practice_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    job_id = Column(Integer, index=True)
    company_slug = Column(String(255), index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_constraint = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PracticeSessionQuestion(Base):
    __tablename__ = "practice_session_questions"
    __table_args__ = (
        UniqueConstraint(
            "practice_session_id",
            "question_id",
            name="uq_practice_session_question",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    practice_session_id = Column(Integer, ForeignKey("practice_sessions.id"), index=True, nullable=False)
    question_id = Column(Integer, ForeignKey("practice_questions.id"), index=True, nullable=False)
    status = Column(String(24), default="seen", nullable=False)
    asked_at = Column(DateTime(timezone=True), server_default=func.now())
    solved_at = Column(DateTime(timezone=True))


class PracticeGeneration(Base):
    __tablename__ = "practice_generations"

    id = Column(Integer, primary_key=True, index=True)
    practice_session_id = Column(Integer, ForeignKey("practice_sessions.id"), index=True, nullable=False)
    source_question_id = Column(Integer, ForeignKey("practice_questions.id"), index=True, nullable=False)
    generated_text = Column(Text, nullable=False)
    constraint_type = Column(String(64))
    applied_constraints = Column(JSON)
    reason = Column(Text)
    llm_provider = Column(String(100))
    llm_model = Column(String(100))
    base_question_link = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
