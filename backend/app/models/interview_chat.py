from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from ..database import Base


class InterviewChatSession(Base):
    __tablename__ = "interview_chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), index=True, nullable=False)
    label = Column(String(120), nullable=False)
    status = Column(String(24), nullable=False, default="active", index=True)
    phase = Column(String(32), nullable=False, default="opening")
    current_question_index = Column(Integer, nullable=False, default=0)
    is_waiting_for_candidate_question = Column(Boolean, nullable=False, default=False)
    question_plan = Column(JSON, nullable=False, default=list)
    session_metadata = Column(JSON, nullable=False, default=dict)
    handoff_run_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class InterviewChatTurn(Base):
    __tablename__ = "interview_chat_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_index", name="uix_interviewchatturn_session_turn"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_chat_sessions.id"), index=True, nullable=False)
    turn_index = Column(Integer, nullable=False)
    speaker = Column(String(16), nullable=False)
    turn_type = Column(String(24), nullable=False)
    content = Column(Text, nullable=False)
    tool_calls = Column(JSON, nullable=False, default=list)
    context_sources = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
