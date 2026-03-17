from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, JSON, String, Text, DateTime
from sqlalchemy.sql import func
from ..database import Base


class ScoreHistory(Base):
    __tablename__ = "score_history"

    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String(255))
    company = Column(String(255))
    job_description = Column(Text, nullable=False)
    fit_score = Column(Float, nullable=False)
    matched_keywords = Column(JSON)   # list of strings
    missing_keywords = Column(JSON)   # list of strings
    gap_analysis = Column(Text)
    rewrite_suggestions = Column(JSON)  # list of strings
    llm_provider = Column(String(100))
    llm_model = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    """Phase 2 – scraped jobs"""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    company = Column(String(255))
    location = Column(String(255))
    url = Column(Text)
    description = Column(Text)
    source = Column(String(50))   # linkedin | indeed | naukri
    work_type = Column(String(50))  # remote | hybrid | onsite
    seniority = Column(String(50))
    posted_at = Column(DateTime(timezone=True))
    posted_at_raw = Column(String(100))
    external_job_id = Column(String(80), index=True)
    canonical_url = Column(Text)
    employment_type = Column(String(100))
    job_function = Column(String(255))
    industries = Column(String(255))
    applicants_count = Column(String(32))
    benefits = Column(JSON)
    salary = Column(String(255))
    company_logo = Column(Text)
    company_linkedin_url = Column(Text)
    company_website = Column(Text)
    company_address = Column(JSON)
    company_employees_count = Column(Integer)
    job_poster_name = Column(String(255))
    job_poster_title = Column(String(255))
    job_poster_profile_url = Column(Text)
    fit_score = Column(Float)
    matched_keywords = Column(JSON)
    missing_keywords = Column(JSON)
    gap_analysis = Column(Text)
    scored_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ScrapeRequest(Base):
    """Audit record for scrape invocations."""

    __tablename__ = "scrape_requests"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False)  # linkedin | indeed | naukri
    role = Column(String(255))
    job = Column(String(255))
    location = Column(String(255))
    years_of_experience = Column(Integer)
    num_records = Column(Integer)
    requested_by = Column(String(80))
    return_raw = Column(Boolean, default=False)
    result_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ScrapedJob(Base):
    """Raw mapped rows for a scrape request (for replay/forensics)."""

    __tablename__ = "scraped_jobs"

    id = Column(Integer, primary_key=True, index=True)
    scrape_request_id = Column(Integer, ForeignKey("scrape_requests.id"), index=True, nullable=False)
    source = Column(String(50))
    title = Column(String(255))
    company = Column(String(255))
    location = Column(String(255))
    url = Column(Text)
    posted_at = Column(DateTime(timezone=True))
    seniority = Column(String(50))
    description = Column(Text)
    raw_payload = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RescoreRun(Base):
    __tablename__ = "rescore_runs"

    id = Column(String(36), primary_key=True, index=True)
    status = Column(String(20), nullable=False, default="queued")
    source = Column(String(50))
    only_unscored = Column(Boolean, default=False)
    total_jobs = Column(Integer, default=0)
    processed_jobs = Column(Integer, default=0)
    scored_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    failed_job_ids = Column(JSON)
    message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
