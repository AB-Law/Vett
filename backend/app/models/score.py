from datetime import datetime

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
    matched_keyword_evidence = Column(JSON)  # list[dict]
    missing_keyword_evidence = Column(JSON)  # list[dict]
    rewrite_suggestion_evidence = Column(JSON)  # list[dict]
    gap_analysis = Column(Text)
    reason = Column(Text)
    rewrite_suggestions = Column(JSON)  # list of strings
    agent_plan = Column(JSON)  # plan payload from second scoring pass
    llm_provider = Column(String(100))
    llm_model = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


AGENT_STATE_QUEUED = "queued"
AGENT_STATE_ROLE_ANALYSIS = "role_analysis"
AGENT_STATE_EVIDENCE_SCAN = "evidence_scan"
AGENT_STATE_SCORING = "scoring"
AGENT_STATE_GAP_AUDIT = "gap_audit"
AGENT_STATE_ACTION_PLAN = "action_plan"
AGENT_STATE_REWRITE_PLAN = "rewrite_plan"
AGENT_STATE_COMPLETED = "completed"
AGENT_STATE_FAILED = "failed"
AGENT_STATE_RETRY_SCHEDULED = "retry_scheduled"
AGENT_STATE_CANCELLED = "cancelled"

AGENT_TERMINAL_STATES = {
    AGENT_STATE_COMPLETED,
    AGENT_STATE_FAILED,
    AGENT_STATE_RETRY_SCHEDULED,
    AGENT_STATE_CANCELLED,
}


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(String(36), primary_key=True, index=True)
    score_history_id = Column(Integer, ForeignKey("score_history.id"), index=True, nullable=True)
    cv_id = Column(Integer, nullable=True)
    idempotency_key = Column(String(128), nullable=False, unique=True, index=True)
    current_state = Column(String(40), nullable=False, default=AGENT_STATE_QUEUED)
    actor = Column(String(80), default="system")
    source = Column(String(80), default="api")
    request_payload = Column(JSON, nullable=True)
    failed_step = Column(String(40), nullable=True)
    failure_reason = Column(Text, nullable=True)
    attempt_count = Column(Integer, default=0, nullable=False)
    status = Column(String(40), nullable=False, default=AGENT_STATE_QUEUED)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class AgentRunArtifact(Base):
    __tablename__ = "agent_run_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("agent_runs.id"), index=True, nullable=False)
    score_history_id = Column(Integer, ForeignKey("score_history.id"), index=True, nullable=True)
    step = Column(String(40), nullable=False)
    actor = Column(String(80), default="system")
    source = Column(String(80), default="api")
    payload = Column(JSON, nullable=True)
    evidence = Column(JSON, nullable=True)
    transition_id = Column(Integer, ForeignKey("agent_run_transitions.id"), nullable=True)
    attempt = Column(Integer, default=1, nullable=False)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentRunTransition(Base):
    __tablename__ = "agent_run_transitions"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("agent_runs.id"), index=True, nullable=False)
    score_history_id = Column(Integer, ForeignKey("score_history.id"), index=True, nullable=True)
    previous_state = Column(String(40), nullable=True)
    next_state = Column(String(40), nullable=False)
    trigger = Column(String(80), nullable=False)
    attempt = Column(Integer, default=1, nullable=False)
    failure_reason = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    actor = Column(String(80), default="system")
    source = Column(String(80), default="api")
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
    reason = Column(Text)
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


class LocalContextToolCallAudit(Base):
    __tablename__ = "local_context_tool_calls"

    id = Column(Integer, primary_key=True, index=True)
    request_uuid = Column(String(36), nullable=False, index=True)
    tool_name = Column(String(80), nullable=False, index=True)
    model_id = Column(String(120), nullable=False)
    actor_role = Column(String(64), nullable=False)
    environment = Column(String(32), nullable=False)
    request_source = Column(String(64), nullable=False)
    session_id = Column(String(255))
    user_id = Column(String(255))
    input_hash = Column(String(64), nullable=False)
    result_hash = Column(String(64), nullable=False)
    latency_ms = Column(Float, nullable=False)
    status = Column(String(24), nullable=False)
    decision_rationale = Column(JSON, nullable=False)
    error_message = Column(Text)
    input_payload = Column(JSON, nullable=False)
    output_payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
