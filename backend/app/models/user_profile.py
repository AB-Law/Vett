"""User profile model for candidate self-description."""

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from ..database import Base


class UserProfile(Base):
    __tablename__ = "user_profile"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False, default="", server_default="")
    headline_or_target_role = Column(String(255), nullable=False, default="", server_default="")
    current_company = Column(String(255), nullable=False, default="", server_default="")
    years_experience = Column(Integer)
    top_skills = Column(JSON, nullable=False, default=list, server_default="[]")
    location = Column(String(255), nullable=False, default="", server_default="")
    linkedin_url = Column(String(255), nullable=False, default="", server_default="")
    summary = Column(Text, nullable=False, default="", server_default="")
    source = Column(String(32), nullable=False, default="manual", server_default="manual")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
