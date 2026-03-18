"""Jobs routes for scraping and retrieving job records."""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from datetime import datetime
import time
import threading
import uuid
import re
import logging
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
import asyncio
import json
from pathlib import Path
from ..config import get_settings
import uuid
from contextlib import suppress
from ..scrapers.linkedin import scrape_linkedin
from ..services.scoring_orchestrator import execute_scoring_orchestrator
from ..services.scrape_storage import store_scrape_results
from ..services.llm import score_cv_fast
from ..services.interview_research import InterviewResearchRunContext, run_interview_research

from ..database import SessionLocal, get_db
from ..models.cv import CV
from ..models.score import Job, RescoreRun, ScrapeRequest
from ..models.interview_research import InterviewResearchSession
from ..models.score import AGENT_STATE_COMPLETED, AGENT_STATE_FAILED
from ..models.interview import InterviewKnowledgeDocument
from ..services.cv_parser import parse_cv
from ..services.interview_docs import build_parser_signature, chunk_interview_text, chunk_integrity_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobSearchRequest(BaseModel):
    query: Optional[str] = None
    role: Optional[str] = None
    job: Optional[str] = None
    location: Optional[str] = None
    source: str = "all"  # linkedin | indeed | naukri | all
    years_of_experience: int | str | None = None
    num_records: int = Field(default=25, ge=10, le=100)
    return_raw: bool = False


class JobSearchResponse(BaseModel):
    status: str
    request_id: int | None = None
    source: str
    count: int
    stored_count: int
    message: str


class JobRescoreRequest(BaseModel):
    source: Optional[str] = None
    only_unscored: bool = False


class JobRescoreResponse(BaseModel):
    status: str
    run_id: str
    source: Optional[str]
    only_unscored: bool
    total_jobs: int
    processed_jobs: int
    scored_count: int
    failed_count: int
    failed_job_ids: list[int] = Field(default_factory=list)
    message: str


class InterviewResearchQuestion(BaseModel):
    question: str
    tool: str
    query: str
    source_url: str
    source_title: str
    timestamp: str
    snippet: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class InterviewResearchQuestionBank(BaseModel):
    behavioral: list[InterviewResearchQuestion]
    technical: list[InterviewResearchQuestion]
    system_design: list[InterviewResearchQuestion]
    company_specific: list[InterviewResearchQuestion]
    source_urls: list[str]


class InterviewResearchSessionResponse(BaseModel):
    session_id: str
    role: str
    company: str
    status: str
    job_id: int
    fallback_used: bool = False
    message: str = ""
    question_bank: InterviewResearchQuestionBank
    metadata: dict[str, object] = Field(default_factory=dict)
    source_urls: list[str]
    failure_reason: str | None = None
    stage: str | None = None
    processing_ms: int | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class JobAnalysisResponse(BaseModel):
    job_id: int
    run_id: str
    run_status: str
    run_state: str
    fit_score: Optional[float]
    matched_keywords: list[str]
    missing_keywords: list[str]
    gap_analysis: Optional[str]
    reason: Optional[str]
    rewrite_suggestions: list[str]
    matched_keyword_evidence: list[dict]
    missing_keyword_evidence: list[dict]
    rewrite_suggestion_evidence: list[dict]
    agent_plan: Optional[dict]
    failure_reason: Optional[str]
    failed_step: Optional[str]


_RESCORE_WORKER_LOCK = threading.Lock()
_RESCORE_WORKER_STARTED = False
_RESCORE_WORKER_POLL_INTERVAL_SECONDS = 0.75


def _log_excerpt(text: object, *, max_chars: int = 140) -> str:
    if text is None:
        return ""
    value = re.sub(r"\s+", " ", str(text)).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _normalise_rescore_source(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "all"}:
        return None
    return normalized


def _as_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    output: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
    return output


def _coerce_int(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _result_score_fields(result: dict[str, object]) -> tuple[int, list[str], list[str], str, str, list[str]]:
    fit_score = _coerce_int(result.get("fit_score"), 0)
    if fit_score < 0:
        fit_score = 0
    if fit_score > 100:
        fit_score = 100

    matched = result.get("matched_keywords")
    missing = result.get("missing_keywords")
    gap_analysis = result.get("gap_analysis")
    rewrite = result.get("rewrite_suggestions")

    if not isinstance(matched, list):
        matched = []
    if not isinstance(missing, list):
        missing = []
    if not isinstance(rewrite, list):
        rewrite = []

    return (
        fit_score,
        [str(item).strip() for item in matched if str(item).strip()],
        [str(item).strip() for item in missing if str(item).strip()],
        str(gap_analysis) if gap_analysis is not None else "",
        str((result or {}).get("reason", "")) if isinstance((result or {}).get("reason"), str) else "",
        [str(item).strip() for item in rewrite if str(item).strip()],
    )


def _rescore_run_payload(run: RescoreRun) -> JobRescoreResponse:
    failed_job_ids = _as_int_list(run.failed_job_ids)
    return JobRescoreResponse(
        status=run.status,
        run_id=run.id,
        source=run.source,
        only_unscored=bool(run.only_unscored),
        total_jobs=run.total_jobs,
        processed_jobs=run.processed_jobs,
        scored_count=run.scored_count,
        failed_count=run.failed_count,
        failed_job_ids=failed_job_ids,
        message=run.message or "",
    )


def _coerce_research_question_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {
            "question": str(value.get("question", "")),
            "tool": str(value.get("tool", "")),
            "query": str(value.get("query", "")),
            "source_url": str(value.get("source_url", "")),
            "source_title": str(value.get("source_title", "")),
            "timestamp": str(value.get("timestamp", "")),
            "snippet": str(value.get("snippet", "")),
            "confidence_score": float(value.get("confidence_score", 0.5) or 0.5),
        }
    return {
        "question": "",
        "tool": "",
        "query": "",
        "source_url": "",
        "source_title": "",
        "timestamp": "",
        "snippet": "",
        "confidence_score": 0.5,
    }


def _coerce_research_question_bank(value: object) -> InterviewResearchQuestionBank:
    if not isinstance(value, dict):
        empty = {
            "behavioral": [],
            "technical": [],
            "system_design": [],
            "company_specific": [],
            "source_urls": [],
        }
        return InterviewResearchQuestionBank(**empty)
    payload = {
        "behavioral": [_coerce_research_question_dict(item) for item in value.get("behavioral", [])],
        "technical": [_coerce_research_question_dict(item) for item in value.get("technical", [])],
        "system_design": [_coerce_research_question_dict(item) for item in value.get("system_design", [])],
        "company_specific": [_coerce_research_question_dict(item) for item in value.get("company_specific", [])],
        "source_urls": [str(item) for item in (value.get("source_urls", []) if isinstance(value.get("source_urls", []), list) else [])],
    }
    return InterviewResearchQuestionBank(**payload)


def _serialize_interview_research_session(row: InterviewResearchSession) -> InterviewResearchSessionResponse:
    question_bank = _coerce_research_question_bank(row.question_bank or {})
    default_message = "Interview research failed." if row.status == "failed" else "Interview research completed."
    session_message = row.failure_reason or default_message
    return InterviewResearchSessionResponse(
        session_id=row.session_id,
        role=row.role or "",
        company=row.company or "",
        status=row.status or "",
        job_id=row.job_id,
        fallback_used=bool(row.fallback_used),
        message=session_message,
        question_bank=question_bank,
        metadata={
            "total_questions": len(question_bank.behavioral) + len(question_bank.technical) + len(question_bank.system_design) + len(question_bank.company_specific),
        },
        source_urls=list(question_bank.source_urls or row.source_urls or []),
        failure_reason=row.failure_reason,
        stage=row.stage,
        processing_ms=row.processing_ms,
        created_at=str(row.created_at),
        updated_at=str(row.updated_at),
        started_at=str(row.started_at) if row.started_at else None,
        completed_at=str(row.completed_at) if row.completed_at else None,
    )


def _upsert_interview_research_session(
    db: Session,
    *,
    session_id: str,
    job_id: int,
    role: str,
    company: str,
    status: str,
    stage: str,
    question_bank: InterviewResearchQuestionBank,
    fallback_used: bool,
    message: str,
    processing_ms: int,
    failure_reason: str | None = None,
    completed_at: bool = False,
) -> InterviewResearchSession:
    row = (
        db.query(InterviewResearchSession)
        .filter(InterviewResearchSession.session_id == session_id)
        .first()
    )
    if row is None:
        row = InterviewResearchSession(
            session_id=session_id,
            job_id=job_id,
            role=role,
            company=company,
            status=status,
            stage=stage,
            question_bank=question_bank.model_dump(),
            source_urls=question_bank.source_urls,
            fallback_used=bool(fallback_used),
            failure_reason=failure_reason or message,
            processing_ms=processing_ms,
        )
        row.started_at = datetime.utcnow()
        row.updated_at = datetime.utcnow()
        if completed_at:
            row.completed_at = row.updated_at
        db.add(row)
        db.flush()
        return row

    row.status = status
    row.stage = stage
    row.role = role
    row.company = company
    row.question_bank = question_bank.model_dump()
    row.source_urls = question_bank.source_urls
    row.fallback_used = bool(fallback_used)
    row.failure_reason = failure_reason or message
    row.processing_ms = processing_ms
    row.updated_at = datetime.utcnow()
    if completed_at:
        row.completed_at = row.updated_at
    return row


def _run_query_for_rescore(q_session: Session, source: Optional[str], only_unscored: bool):
    q = q_session.query(Job)
    if source:
        q = q.filter(Job.source == source)
    if only_unscored:
        q = q.filter(Job.fit_score == None)
    return q


def _update_rescore_run(run_id: str, **updates) -> None:
    db = SessionLocal()
    try:
        run = db.query(RescoreRun).filter(RescoreRun.id == run_id).first()
        if run is None:
            return
        for key, value in updates.items():
            setattr(run, key, value)
        run.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def _start_rescore_worker() -> None:
    global _RESCORE_WORKER_STARTED
    with _RESCORE_WORKER_LOCK:
        if _RESCORE_WORKER_STARTED:
            return

        def _worker() -> None:
            while True:
                run_id: Optional[str] = None
                db: Optional[Session] = None
                try:
                    logger.debug("Rescore worker polling for queued runs.")
                    db = SessionLocal()
                    queued_run = (
                        db.query(RescoreRun)
                        .filter(RescoreRun.status == "queued")
                        .order_by(RescoreRun.created_at.asc())
                        .first()
                    )
                    if queued_run is None:
                        db.close()
                        time.sleep(_RESCORE_WORKER_POLL_INTERVAL_SECONDS)
                        continue

                    run_id = queued_run.id
                    queued_run.status = "running"
                    queued_run.started_at = datetime.utcnow()
                    queued_run.message = "Rescoring started."
                    queued_run.updated_at = queued_run.started_at
                    logger.info(
                        "Dequeued rescore run %s (source=%s only_unscored=%s)",
                        queued_run.id,
                        queued_run.source,
                        bool(queued_run.only_unscored),
                    )
                    db.commit()
                except Exception:
                    if db is not None:
                        db.close()
                    if run_id is not None:
                        _update_rescore_run(
                            run_id,
                            status="failed",
                            completed_at=datetime.utcnow(),
                            message="Worker failed while starting this run.",
                        )
                    logger.exception("Rescore worker failed to start run=%s", run_id)
                    time.sleep(_RESCORE_WORKER_POLL_INTERVAL_SECONDS)
                    continue
                finally:
                    if db is not None:
                        db.close()

                asyncio.run(_process_rescore_run(run_id))

        thread = threading.Thread(target=_worker, daemon=True, name="rescore-worker")
        thread.start()
        _RESCORE_WORKER_STARTED = True


async def _score_single_job_fast(
    job_id: int,
    cv_text: str,
    run_id: str,
) -> tuple[int, bool, str | None]:
    """Score one job with fast single-call scoring. Returns (job_id, success, error_msg)."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return job_id, False, "Job not found"
        if not job.description:
            return job_id, False, "No description"
        try:
            result = await score_cv_fast(cv_text, job.description)
        except Exception as exc:
            logger.warning(
                "Fast scoring failed for job %s in run %s; falling back to orchestrator. %s",
                job_id,
                run_id,
                exc,
            )
            cv = db.query(CV).order_by(CV.id.desc()).first()
            if cv is None:
                return job_id, False, "No CV uploaded."
            orchestrator_result = await execute_scoring_orchestrator(
                db,
                cv_id=cv.id,
                cv_text=cv.parsed_text or "",
                job_title=job.title,
                company=job.company,
                job_description=job.description,
                actor="rescore-worker",
                source="rescore",
                idempotency_key=f"{run_id}:job:{job_id}",
            )
            if orchestrator_result.status != AGENT_STATE_COMPLETED:
                return (
                    job_id,
                    False,
                    orchestrator_result.failure_reason
                    or orchestrator_result.status
                    or AGENT_STATE_FAILED,
                )
            result = orchestrator_result.result if isinstance(orchestrator_result.result, dict) else {}
        fit_score = result.get("fit_score", 0)
        matched = result.get("matched_keywords", [])
        missing = result.get("missing_keywords", [])
        gap = result.get("gap_analysis", "")
        job.fit_score = fit_score
        job.matched_keywords = matched if isinstance(matched, list) else []
        job.missing_keywords = missing if isinstance(missing, list) else []
        job.gap_analysis = str(gap) if gap else ""
        job.reason = ""
        job.scored_at = datetime.utcnow()
        db.commit()
        return job_id, True, None
    except Exception as exc:
        logger.exception("Fast scoring failed for job %s in run %s", job_id, run_id)
        return job_id, False, str(exc)
    finally:
        db.close()


async def _process_rescore_run(run_id: str) -> None:
    db = SessionLocal()
    try:
        run = db.query(RescoreRun).filter(RescoreRun.id == run_id).first()
        if run is None or run.status != "running":
            logger.warning("Rescore run %s was not running at process start.", run_id)
            return

        logger.info(
            "Starting rescore run processing run_id=%s source=%s only_unscored=%s",
            run_id,
            run.source,
            bool(run.only_unscored),
        )

        cv = db.query(CV).order_by(CV.id.desc()).first()
        if not cv:
            logger.warning("Rescore run %s failed: no CV uploaded.", run_id)
            run.status = "failed"
            run.completed_at = datetime.utcnow()
            run.message = "No CV uploaded. Please upload a CV first."
            run.updated_at = datetime.utcnow()
            db.commit()
            return

        q = _run_query_for_rescore(db, run.source, bool(run.only_unscored))
        jobs = q.order_by(Job.created_at.desc()).all()
        run.total_jobs = len(jobs)
        run.message = f"Scoring {run.total_jobs} jobs..."
        run.updated_at = datetime.utcnow()
        logger.info(
            "Run %s discovered %s jobs to score (source=%s).",
            run_id,
            run.total_jobs,
            run.source or "all",
        )
        db.commit()

        failed_job_ids = _as_int_list(run.failed_job_ids)
        job_ids = [job.id for job in jobs if job.description]
        no_desc_ids = [job.id for job in jobs if not job.description]

        for jid in no_desc_ids:
            logger.warning("Run %s skipped job %s due to missing description.", run_id, jid)
            if jid not in failed_job_ids:
                failed_job_ids.append(jid)
                run.failed_count += 1
            run.processed_jobs += 1
        if no_desc_ids:
            run.failed_job_ids = failed_job_ids
            run.updated_at = datetime.utcnow()
            db.commit()

        chunk_size = 10
        for i in range(0, len(job_ids), chunk_size):
            chunk = job_ids[i:i + chunk_size]
            results = await asyncio.gather(
                *[_score_single_job_fast(jid, cv.parsed_text, run_id) for jid in chunk],
                return_exceptions=False,
            )
            for jid, success, _err in results:
                run.processed_jobs += 1
                if success:
                    run.scored_count += 1
                else:
                    if jid not in failed_job_ids:
                        failed_job_ids.append(jid)
                    run.failed_count += 1
            run.failed_job_ids = failed_job_ids
            run.message = (
                f"Scoring in progress ({run.processed_jobs}/{run.total_jobs}). "
                f"{run.scored_count} scored, {run.failed_count} failed."
            )
            run.updated_at = datetime.utcnow()
            db.commit()

        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.message = (
            f"Re-scoring complete. Scored {run.scored_count}/{run.total_jobs} jobs. "
            f"{run.failed_count} failed."
        )
        run.updated_at = datetime.utcnow()
        logger.info(
            "Rescore run %s completed: scored=%s failed=%s total=%s",
            run_id,
            run.scored_count,
            run.failed_count,
            run.total_jobs,
        )
        db.commit()
    except Exception:
        logger.exception("Unexpected failure while processing rescore run %s", run_id)
        run = db.query(RescoreRun).filter(RescoreRun.id == run_id).first()
        if run is not None:
            run.status = "failed"
            run.completed_at = datetime.utcnow()
            run.message = "Rescore run failed due to an unexpected error."
            run.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


class JobItem(BaseModel):
    id: int
    title: Optional[str]
    company: Optional[str]
    location: Optional[str]
    url: Optional[str]
    source: Optional[str]
    work_type: Optional[str]
    external_job_id: Optional[str]
    canonical_url: Optional[str]
    posted_at_raw: Optional[str]
    employment_type: Optional[str]
    job_function: Optional[str]
    industries: Optional[str]
    applicants_count: Optional[str]
    benefits: Optional[list[str]]
    salary: Optional[str]
    company_logo: Optional[str]
    company_linkedin_url: Optional[str]
    company_website: Optional[str]
    company_address: Optional[dict[str, Any]]
    company_employees_count: Optional[int]
    job_poster_name: Optional[str]
    job_poster_title: Optional[str]
    job_poster_profile_url: Optional[str]
    seniority: Optional[str]
    posted_at: Optional[str]
    fit_score: Optional[float]
    matched_keywords: Optional[list[str]]
    missing_keywords: Optional[list[str]]
    gap_analysis: Optional[str]
    reason: Optional[str]
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class InterviewKnowledgeDocumentResponse(BaseModel):
    id: int
    owner_type: str
    job_id: int | None
    source_filename: str
    content_type: str
    status: str
    error_message: str | None
    parser_version: str | None
    source_ref: str | None
    total_chunks: int
    embedded_chunks: int
    parsed_word_count: int
    created_at: str
    created_by_user_id: str | None


DOCUMENT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}


def _serialize_job_document(document: InterviewKnowledgeDocument) -> InterviewKnowledgeDocumentResponse:
    return InterviewKnowledgeDocumentResponse(
        id=document.id,
        owner_type=document.owner_type,
        job_id=document.job_id,
        source_filename=document.source_filename,
        content_type=document.content_type or "",
        status=document.status,
        error_message=document.error_message,
        parser_version=document.parser_version,
        source_ref=document.source_ref,
        total_chunks=int(document.total_chunks or 0),
        embedded_chunks=int(document.embedded_chunks or 0),
        parsed_word_count=int(document.parsed_word_count or 0),
        created_at=str(document.created_at),
        created_by_user_id=document.created_by_user_id,
    )


def _parse_document_upload(file: UploadFile) -> tuple[str, str]:
    filename = (file.filename or "interview-doc.txt").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in DOCUMENT_ALLOWED_EXTENSIONS:
        logger.warning("Rejected job interview document upload: filename=%s extension=%s", filename, suffix)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(DOCUMENT_ALLOWED_EXTENSIONS))}",
        )
    logger.info("Accepted job interview document upload filename=%s extension=%s", filename, suffix)
    return filename, suffix


@router.post("/search")
def start_job_search(req: JobSearchRequest, db: Session = Depends(get_db)):
    """Run scrape immediately and persist raw + mapped payload."""
    source = req.source.lower()
    if source not in {"linkedin", "all"}:
        raise HTTPException(status_code=400, detail=f"Source '{req.source}' is not supported yet.")

    role = (req.role or req.query or "").strip()
    if not role and not req.job:
        raise HTTPException(status_code=400, detail="Either role/query or job is required.")

    # Start by scraping LinkedIn (all currently means LinkedIn only for now).
    try:
        results = scrape_linkedin(
            role=role,
            location=req.location,
            job=req.job,
            years_of_experience=req.years_of_experience,
            num_records=req.num_records,
            return_raw=req.return_raw,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Scrape failed: {exc}")

    scrape = ScrapeRequest(
        source="linkedin",
        role=role,
        job=req.job,
        location=req.location,
        years_of_experience=_to_int(req.years_of_experience),
        num_records=req.num_records,
        requested_by="ui",
        return_raw=req.return_raw,
        result_count=len(results),
    )
    db.add(scrape)
    db.flush()

    before_count = db.query(Job).filter(Job.source == "linkedin").count()
    stored_count = store_scrape_results(db, scrape.id, "linkedin", results, store_mapped=True)
    persisted_count = db.query(Job).filter(Job.source == "linkedin").count() - before_count

    db.commit()

    if len(results) == 0:
        message = "No jobs were returned from LinkedIn for this search."
    elif persisted_count == 0:
        message = (
            "LinkedIn returned results, but none were saved into the jobs table. "
            "Check scraper field mapping for missing required fields."
        )
    else:
        message = (
            f"Scraped {len(results)} LinkedIn jobs and persisted {persisted_count} new jobs. "
            "If source was 'all', only LinkedIn was executed in this phase."
        )

    response = JobSearchResponse(
        status="ok",
        request_id=scrape.id,
        source="linkedin",
        count=len(results),
        stored_count=persisted_count,
        message=message,
    )
    logger.info(
        "jobs/search response body: %s",
        response.model_dump(),
    )
    return response


@router.post("/rescore", response_model=JobRescoreResponse)
async def rescore_jobs(req: JobRescoreRequest, db: Session = Depends(get_db)):
    """Queue a background job to re-score stored jobs against the latest uploaded CV."""
    cv = db.query(CV).order_by(CV.id.desc()).first()
    if not cv:
        raise HTTPException(status_code=400, detail="No CV uploaded. Please upload a CV first.")

    source = _normalise_rescore_source(req.source)
    q = _run_query_for_rescore(db, source, req.only_unscored)
    total_jobs = q.count()

    run = RescoreRun(
        id=uuid.uuid4().hex,
        status="queued" if total_jobs > 0 else "completed",
        source=source,
        only_unscored=bool(req.only_unscored),
        total_jobs=total_jobs,
        processed_jobs=0,
        scored_count=0,
        failed_count=0,
        failed_job_ids=[],
        message="No jobs matched the request." if total_jobs == 0 else "Queued for background scoring.",
    )
    if total_jobs == 0:
        now = datetime.utcnow()
        run.started_at = now
        run.completed_at = now
        run.updated_at = now
    db.add(run)
    db.commit()

    if total_jobs == 0:
        return _rescore_run_payload(run)

    _start_rescore_worker()
    return _rescore_run_payload(run)


@router.get("/rescore/{run_id}", response_model=JobRescoreResponse)
def get_rescore_status(run_id: str, db: Session = Depends(get_db)):
    run = db.query(RescoreRun).filter(RescoreRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _rescore_run_payload(run)


@router.get("/{job_id}/interview-research/session/{session_id}", response_model=InterviewResearchSessionResponse)
def get_interview_research_session(
    job_id: int,
    session_id: str,
    db: Session = Depends(get_db),
):
    row = (
        db.query(InterviewResearchSession)
        .filter(InterviewResearchSession.session_id == session_id, InterviewResearchSession.job_id == job_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Interview research session not found.")
    return _serialize_interview_research_session(row)


@router.get("/{job_id}/interview-research/stream")
async def stream_interview_research(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    role = (job.title or "Interview role").strip()
    company = (job.company or "this company").strip()
    session_id = uuid.uuid4().hex
    settings = get_settings()
    queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

    async def emit(payload: dict[str, object]) -> None:
        event = dict(payload)
        event["session_id"] = session_id
        await queue.put(event)

    async def produce() -> None:
        start_ms = time.perf_counter()
        try:
            await emit({
                "type": "status",
                "stage": "initialized",
                "message": f"Researching interview questions for {role} at {company}...",
                "stage_payload": {},
                "payload": {},
            })
            context = InterviewResearchRunContext(
                role=role,
                company=company,
                job=job,
                emit=emit,
                timeout_seconds=settings.interview_research_timeout_seconds,
            )
            result = await run_interview_research(db, context)
            result.session_id = session_id
            processing_ms = int((time.perf_counter() - start_ms) * 1000)
            record = _upsert_interview_research_session(
                db,
                session_id=session_id,
                job_id=job.id,
                role=role,
                company=company,
                status="completed",
                stage="finalizing",
                question_bank=result.question_bank,
                fallback_used=result.fallback_used,
                message=result.message,
                processing_ms=processing_ms,
                completed_at=True,
            )
            db.commit()
            await emit({
                "type": "done",
                "stage": "finalizing",
                "message": result.message,
                "payload": {
                    "session_id": session_id,
                    "question_count": len(result.question_bank.all_questions()),
                    "fallback_used": bool(result.fallback_used),
                    "status": record.status,
                },
            })
        except Exception as exc:
            logger.exception("Interview research stream failed for job=%s", job_id)
            processing_ms = int((time.perf_counter() - start_ms) * 1000)
            _upsert_interview_research_session(
                db,
                session_id=session_id,
                job_id=job.id,
                role=role,
                company=company,
                status="failed",
                stage="finalizing",
                question_bank=InterviewResearchQuestionBank(
                    behavioral=[],
                    technical=[],
                    system_design=[],
                    company_specific=[],
                    source_urls=[],
                ),
                fallback_used=False,
                message="Interview research failed.",
                processing_ms=processing_ms,
                failure_reason=str(exc),
                completed_at=True,
            )
            db.rollback()
            db.commit()
            await emit({
                "type": "error",
                "stage": "finalizing",
                "message": str(exc),
                "payload": {"session_id": session_id, "status": "failed"},
            })
        finally:
            await queue.put(None)

    async def event_generator():
        producer = asyncio.create_task(produce())
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                yield f"data: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            producer.cancel()
            raise
        finally:
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/stream")
async def stream_job_results(task_id: str = Query(...)):
    """Phase 2: SSE stream of scored job results from a Celery task."""
    async def event_generator():
        # Placeholder – real implementation polls Celery task result
        for i in range(3):
            await asyncio.sleep(0.5)
            data = json.dumps({
                "type": "status",
                "message": f"Phase 2 not implemented. Event {i+1}/3",
            })
            yield f"data: {data}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/", response_model=list[JobItem])
def list_jobs(
    db: Session = Depends(get_db),
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    work_type: Optional[str] = None,
    seniority: Optional[str] = None,
    source: Optional[str] = None,
    score_status: Optional[str] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    limit: int = 50,
):
    q = db.query(Job)
    if min_score is not None:
        q = q.filter(Job.fit_score >= min_score)
    if max_score is not None:
        q = q.filter(Job.fit_score <= max_score)
    if score_status == "scored":
        q = q.filter(Job.fit_score != None)
    elif score_status == "unscored":
        q = q.filter(Job.fit_score == None)
    if work_type:
        q = q.filter(Job.work_type == work_type)
    if seniority:
        q = q.filter(Job.seniority == seniority)
    if source:
        q = q.filter(Job.source == source)

    sort_key = (sort_by or "created_at").lower()
    direction = (sort_dir or "desc").lower()
    ordering = desc
    if direction == "asc":
        ordering = asc

    sort_columns: dict[str, Any] = {
        "created_at": Job.created_at,
        "title": Job.title,
        "company": Job.company,
        "location": Job.location,
        "source": Job.source,
        "work_type": Job.work_type,
        "seniority": Job.seniority,
        "score": Job.fit_score,
        "fit_score": Job.fit_score,
        "posted_at": Job.posted_at,
    }
    order_expr = sort_columns.get(sort_key, Job.created_at)
    if order_expr is Job.fit_score:
        order_by_clause = ordering(order_expr).nulls_last()
    elif order_expr is Job.posted_at:
        order_by_clause = ordering(order_expr).nulls_last()
    else:
        order_by_clause = ordering(order_expr)

    jobs = q.order_by(order_by_clause).limit(limit).all()
    return [
        JobItem(
            id=j.id,
            title=j.title,
            company=j.company,
            location=j.location,
            url=j.url,
            source=j.source,
            work_type=j.work_type,
            external_job_id=j.external_job_id,
            canonical_url=j.canonical_url,
            posted_at_raw=j.posted_at_raw,
            employment_type=j.employment_type,
            job_function=j.job_function,
            industries=j.industries,
            applicants_count=j.applicants_count,
            benefits=j.benefits,
            salary=j.salary,
            company_logo=j.company_logo,
            company_linkedin_url=j.company_linkedin_url,
            company_website=j.company_website,
            company_address=j.company_address,
            company_employees_count=j.company_employees_count,
            job_poster_name=j.job_poster_name,
            job_poster_title=j.job_poster_title,
            job_poster_profile_url=j.job_poster_profile_url,
            seniority=j.seniority,
            posted_at=str(j.posted_at) if j.posted_at else None,
            fit_score=j.fit_score,
            matched_keywords=j.matched_keywords or [],
            missing_keywords=j.missing_keywords or [],
            gap_analysis=j.gap_analysis,
            reason=j.reason,
            created_at=str(j.created_at),
        )
        for j in jobs
    ]


@router.get("/{job_id}", response_model=JobItem)
def get_job(job_id: int, db: Session = Depends(get_db)):
    from fastapi import HTTPException
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return JobItem(
        id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        url=job.url,
        source=job.source,
        work_type=job.work_type,
        external_job_id=job.external_job_id,
        canonical_url=job.canonical_url,
        posted_at_raw=job.posted_at_raw,
        employment_type=job.employment_type,
        job_function=job.job_function,
        industries=job.industries,
        applicants_count=job.applicants_count,
        benefits=job.benefits,
        salary=job.salary,
        company_logo=job.company_logo,
        company_linkedin_url=job.company_linkedin_url,
        company_website=job.company_website,
        company_address=job.company_address,
        company_employees_count=job.company_employees_count,
        job_poster_name=job.job_poster_name,
        job_poster_title=job.job_poster_title,
        job_poster_profile_url=job.job_poster_profile_url,
        seniority=job.seniority,
        posted_at=str(job.posted_at) if job.posted_at else None,
        fit_score=job.fit_score,
        matched_keywords=job.matched_keywords or [],
        missing_keywords=job.missing_keywords or [],
        gap_analysis=job.gap_analysis,
        reason=job.reason,
        created_at=str(job.created_at),
    )


@router.get("/{job_id}/interview-documents", response_model=list[InterviewKnowledgeDocumentResponse])
def list_job_interview_documents(job_id: int, db: Session = Depends(get_db)):
    if not db.query(Job).filter(Job.id == job_id).first():
        raise HTTPException(404, "Job not found")
    docs = (
        db.query(InterviewKnowledgeDocument)
        .filter(
            InterviewKnowledgeDocument.owner_type == "job",
            InterviewKnowledgeDocument.job_id == job_id,
        )
        .order_by(InterviewKnowledgeDocument.id.desc())
        .all()
    )
    return [_serialize_job_document(document) for document in docs]


@router.post("/{job_id}/interview-documents", response_model=InterviewKnowledgeDocumentResponse)
async def upload_job_interview_document(
    job_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    start_ts = time.perf_counter()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.warning("Job interview document upload failed: job not found job_id=%s", job_id)
        raise HTTPException(404, "Job not found")

    filename, suffix = _parse_document_upload(file)
    data = await file.read()
    logger.info(
        "Starting parse for job interview document job_id=%s filename=%s size=%s content_type=%s",
        job_id,
        filename,
        len(data),
        file.content_type,
    )

    status = "pending"
    error_message: str | None = None
    try:
        parse_start = time.perf_counter()
        parsed_text = parse_cv(data, filename)
        logger.info(
            "Parsed job interview document job_id=%s filename=%s parsed_chars=%s duration_ms=%.1f",
            job_id,
            filename,
            len(parsed_text),
            (time.perf_counter() - parse_start) * 1000,
        )
    except ValueError as exc:
        parsed_text = ""
        status = "failed"
        error_message = str(exc)
        logger.warning(
            "Job interview document failed validation job_id=%s filename=%s error=%s",
            job_id,
            filename,
            error_message,
        )
    except Exception as exc:
        parsed_text = ""
        status = "failed"
        error_message = f"Upload failed: {exc}"
        logger.exception("Job interview document parse failed job_id=%s filename=%s", job_id, filename)

    if not parsed_text.strip() and status == "pending":
        status = "failed"
        error_message = "No text could be extracted from this document."
        logger.warning("Job interview document parse produced no text job_id=%s filename=%s", job_id, filename)

    document = InterviewKnowledgeDocument(
        owner_type="job",
        job_id=job.id,
        source_filename=filename,
        content_type=(suffix[1:] if suffix else "txt"),
        parsed_text=parsed_text,
        total_chunks=len(chunk_interview_text(parsed_text)),
        embedded_chunks=0,
        parsed_word_count=len((parsed_text or "").strip().split()),
        chunk_coverage_ratio=chunk_integrity_stats(parsed_text).get("coverage_ratio", 0.0),
        parser_version=build_parser_signature(),
        source_ref=build_parser_signature(),
        status=status,
        error_message=error_message,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(
        "Persisted job interview document job_id=%s document_id=%s status=%s duration_ms=%.1f",
        job_id,
        document.id,
        status,
        (time.perf_counter() - start_ts) * 1000,
    )
    return _serialize_job_document(document)


@router.post("/{job_id}/analyze", response_model=JobAnalysisResponse)
async def analyze_job(job_id: int, db: Session = Depends(get_db)):
    from fastapi import HTTPException
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.description:
        raise HTTPException(400, "Job has no description to analyze")
    cv = db.query(CV).order_by(CV.id.desc()).first()
    if not cv:
        raise HTTPException(400, "No CV uploaded")

    result = await execute_scoring_orchestrator(
        db,
        cv_id=cv.id,
        cv_text=cv.parsed_text,
        job_title=job.title,
        company=job.company,
        job_description=job.description,
        actor="jobs.analyze",
        source="jobs",
        idempotency_key=None,  # always fresh run
    )

    r = result.result
    fit_score = r.get("fit_score")
    matched = r.get("matched_keywords") or []
    missing = r.get("missing_keywords") or []
    gap = r.get("gap_analysis") or ""
    reason = r.get("reason") or ""
    rewrites = r.get("rewrite_suggestions") or []

    # Persist the enriched fields back to the job
    job.fit_score = fit_score
    job.matched_keywords = matched if isinstance(matched, list) else []
    job.missing_keywords = missing if isinstance(missing, list) else []
    job.gap_analysis = str(gap) if gap else ""
    job.reason = str(reason) if reason else ""
    job.scored_at = datetime.utcnow()
    db.commit()

    return JobAnalysisResponse(
        job_id=job_id,
        run_id=result.run_id,
        run_status=result.status,
        run_state=result.current_state,
        fit_score=fit_score,
        matched_keywords=matched if isinstance(matched, list) else [],
        missing_keywords=missing if isinstance(missing, list) else [],
        gap_analysis=str(gap) if gap else None,
        reason=str(reason) if reason else None,
        rewrite_suggestions=rewrites if isinstance(rewrites, list) else [],
        matched_keyword_evidence=r.get("matched_keyword_evidence") or [],
        missing_keyword_evidence=r.get("missing_keyword_evidence") or [],
        rewrite_suggestion_evidence=r.get("rewrite_suggestion_evidence") or [],
        agent_plan=r.get("agent_plan"),
        failure_reason=result.failure_reason,
        failed_step=result.failed_step,
    )


def _to_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\d+", value.strip())
        if match:
            return int(match.group(0))
    return None
