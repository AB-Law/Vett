"""Models for study card sets and spaced-repetition flashcards."""

from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class StudyCardSet(Base):
    __tablename__ = "study_card_sets"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), index=True, nullable=True)
    parent_card_set_id = Column(Integer, ForeignKey("study_card_sets.id"), index=True, nullable=True)
    name = Column(String(255), nullable=False, server_default="Untitled deck")
    topic = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    cards = relationship("StudyCard", back_populates="card_set", cascade="all, delete-orphan")
    selected_documents = relationship("StudyCardSetDocument", back_populates="card_set", cascade="all, delete-orphan")
    parent_card_set = relationship("StudyCardSet", remote_side=[id], back_populates="child_card_sets")
    child_card_sets = relationship("StudyCardSet", back_populates="parent_card_set", cascade="all, delete-orphan")


class StudyCard(Base):
    __tablename__ = "study_cards"

    id = Column(Integer, primary_key=True, index=True)
    card_set_id = Column(Integer, ForeignKey("study_card_sets.id", ondelete="CASCADE"), index=True, nullable=False)
    front = Column(Text, nullable=False)
    back = Column(Text, nullable=False)
    last_reviewed_at = Column(DateTime(timezone=True))
    ease_factor = Column(Float, nullable=False, default=2.5)
    interval_days = Column(Integer, nullable=False, default=1)
    card_set = relationship("StudyCardSet", back_populates="cards")


class StudyCardSetDocument(Base):
    __tablename__ = "study_card_set_documents"
    __table_args__ = (
        UniqueConstraint("card_set_id", "document_id", name="uq_study_card_set_document"),
    )

    id = Column(Integer, primary_key=True, index=True)
    card_set_id = Column(Integer, ForeignKey("study_card_sets.id", ondelete="CASCADE"), index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("interview_knowledge_documents.id"), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    card_set = relationship("StudyCardSet", back_populates="selected_documents")


class MindMapJob(Base):
    __tablename__ = "mind_map_jobs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(64), nullable=False, unique=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), index=True, nullable=True)
    doc_id = Column(Integer, ForeignKey("interview_knowledge_documents.id"), index=True, nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending | processing | done | failed
    error = Column(Text, nullable=True)
    graph_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    node_infos = relationship("MindMapNodeInfo", back_populates="mind_map_job", cascade="all, delete-orphan")


class MindMapNodeInfo(Base):
    __tablename__ = "mind_map_node_infos"
    __table_args__ = (
        UniqueConstraint("mind_map_job_id", "node_id", name="uq_mind_map_node_info"),
    )

    id = Column(Integer, primary_key=True, index=True)
    mind_map_job_id = Column(Integer, ForeignKey("mind_map_jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    node_id = Column(String(128), nullable=False)
    encyclopedic = Column(Text, nullable=False)
    interview_prep = Column(Text, nullable=False)
    sources_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    mind_map_job = relationship("MindMapJob", back_populates="node_infos")


class MindMap(Base):
    __tablename__ = "mind_maps"
    __table_args__ = (
        UniqueConstraint("job_id", "doc_id", "content_hash", name="uq_mind_maps_job_doc_hash"),
    )

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), index=True, nullable=False)
    doc_id = Column(Integer, ForeignKey("interview_knowledge_documents.id"), index=True, nullable=True)
    content_hash = Column(String(64), nullable=False, index=True)
    graph_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
