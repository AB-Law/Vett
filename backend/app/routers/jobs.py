"""Jobs routes for scraping and retrieving job records."""
from fastapi import APIRouter, Depends, HTTPException, Query
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
from ..scrapers.linkedin import scrape_linkedin
from ..services.scoring_orchestrator import execute_scoring_orchestrator
from ..services.scrape_storage import store_scrape_results

from ..database import SessionLocal, get_db
from ..models.cv import CV
from ..models.score import Job, RescoreRun, ScrapeRequest

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


_RESCORE_WORKER_LOCK = threading.Lock()
_RESCORE_WORKER_STARTED = False


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


def _result_score_fields(result: dict[str, object]) -> tuple[int, list[str], list[str], str, list[str]]:
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
                    db = SessionLocal()
                    queued_run = (
                        db.query(RescoreRun)
                        .filter(RescoreRun.status == "queued")
                        .order_by(RescoreRun.created_at.asc())
                        .first()
                    )
                    if queued_run is None:
                        db.close()
                        time.sleep(0.75)
                        continue

                    run_id = queued_run.id
                    queued_run.status = "running"
                    queued_run.started_at = datetime.utcnow()
                    queued_run.message = "Rescoring started."
                    queued_run.updated_at = queued_run.started_at
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
                    time.sleep(0.75)
                    continue
                finally:
                    if db is not None:
                        db.close()

                asyncio.run(_process_rescore_run(run_id))

        thread = threading.Thread(target=_worker, daemon=True, name="rescore-worker")
        thread.start()
        _RESCORE_WORKER_STARTED = True


async def _process_rescore_run(run_id: str) -> None:
    db = SessionLocal()
    try:
        run = db.query(RescoreRun).filter(RescoreRun.id == run_id).first()
        if run is None or run.status != "running":
            return

        cv = db.query(CV).order_by(CV.id.desc()).first()
        if not cv:
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
        db.commit()

        failed_job_ids = _as_int_list(run.failed_job_ids)
        for job in jobs:
            run.processed_jobs += 1

            if not job.description:
                if job.id not in failed_job_ids:
                    failed_job_ids.append(job.id)
                    run.failed_count += 1
                run.failed_job_ids = failed_job_ids
                run.message = (
                    f"Scoring in progress ({run.processed_jobs}/{run.total_jobs}). "
                    f"{run.scored_count} scored, {run.failed_count} failed."
                )
                run.updated_at = datetime.utcnow()
                db.commit()
                continue

            try:
                orchestrator_result = await execute_scoring_orchestrator(
                    db,
                    cv_id=cv.id,
                    cv_text=cv.parsed_text,
                    job_title=job.title,
                    company=job.company,
                    job_description=job.description,
                    actor="jobs.rescore_worker",
                    source="jobs",
                    idempotency_key=f"jobs:{run_id}:job:{job.id}",
                )
            except Exception:
                logger.exception("Score failed for job %s (run %s)", job.id, run_id)
                if job.id not in failed_job_ids:
                    failed_job_ids.append(job.id)
                    run.failed_count += 1
                run.failed_job_ids = failed_job_ids
                run.message = (
                    f"Scoring in progress ({run.processed_jobs}/{run.total_jobs}). "
                    f"{run.scored_count} scored, {run.failed_count} failed."
                )
                run.updated_at = datetime.utcnow()
                db.commit()
                continue

            if orchestrator_result.status == "failed":
                if job.id not in failed_job_ids:
                    failed_job_ids.append(job.id)
                    run.failed_count += 1
                run.failed_job_ids = failed_job_ids
                run.message = (
                    f"Scoring in progress ({run.processed_jobs}/{run.total_jobs}). "
                    f"{run.scored_count} scored, {run.failed_count} failed."
                )
                run.updated_at = datetime.utcnow()
                db.commit()
                continue

            fit_score, matched_keywords, missing_keywords, gap_analysis, _ = _result_score_fields(orchestrator_result.result)
            job.fit_score = fit_score
            job.matched_keywords = matched_keywords
            job.missing_keywords = missing_keywords
            job.gap_analysis = gap_analysis
            job.scored_at = datetime.utcnow()

            run.scored_count += 1
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
    created_at: str

    model_config = ConfigDict(from_attributes=True)


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
        created_at=str(job.created_at),
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
