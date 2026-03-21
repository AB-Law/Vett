from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..services import flashcards
from ..services.flashcards import ReviewRating
from ..services import mindmap


router = APIRouter(prefix="/study", tags=["study"])
FLASHCARD_JOB_TTL_SECONDS = 60 * 60
FlashcardJobStatus = Literal["pending", "running", "completed", "failed"]
_flashcard_jobs: dict[str, dict[str, object]] = {}
_flashcard_jobs_lock = asyncio.Lock()
_flashcard_job_tasks: set[asyncio.Task[None]] = set()


class FlashcardRequest(BaseModel):
    job_id: int | None = None
    document_ids: list[int] = Field(default_factory=list)
    name: str | None = Field(default=None, max_length=255)
    topic: str | None = None
    num_cards: int = Field(default=10, ge=1, le=5000)
    generate_per_section: bool = False


class FlashcardResponse(BaseModel):
    id: int
    front: str
    back: str
    last_reviewed_at: str | None = None
    ease_factor: float
    interval_days: int

    model_config = ConfigDict(from_attributes=True)


class FlashcardSetItemResponse(BaseModel):
    card_set_id: int
    cards: list[FlashcardResponse]
    card_set: "StudyCardSetSummaryResponse"


class FlashcardGenerationDiagnosticsResponse(BaseModel):
    requested_cards: int
    llm_cards_parsed: int
    deduped_out: int
    fallback_cards_used: int
    fallback_used: bool


class FlashcardSetResponse(FlashcardSetItemResponse):
    card_sets: list[FlashcardSetItemResponse] = Field(default_factory=list)
    parent_card_set_id: int | None = None
    generation_diagnostics: FlashcardGenerationDiagnosticsResponse | None = None


class FlashcardAsyncStartResponse(BaseModel):
    job_id: str
    status: FlashcardJobStatus


class FlashcardJobStatusResponse(BaseModel):
    job_id: str
    status: FlashcardJobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    result: FlashcardSetResponse | None = None


class StudyCardSetSummaryResponse(BaseModel):
    id: int
    job_id: int | None = None
    parent_card_set_id: int | None = None
    name: str
    topic: str | None = None
    created_at: str | None = None
    card_count: int
    document_ids: list[int] = Field(default_factory=list)
    document_count: int = 0


class StudyCardSetDetailResponse(BaseModel):
    card_set_id: int
    job_id: int | None = None
    parent_card_set_id: int | None = None
    name: str
    topic: str | None = None
    created_at: str | None = None
    cards: list[FlashcardResponse]
    document_ids: list[int] = Field(default_factory=list)
    document_count: int = 0


class StudyCardSetRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ReviewRequest(BaseModel):
    rating: ReviewRating


class MindMapRequest(BaseModel):
    job_id: int | None = None
    doc_id: int | None = None


class MindMapNodeResponse(BaseModel):
    id: str
    label: str
    group: str


class MindMapEdgeResponse(BaseModel):
    source: str
    target: str
    label: str


class MindMapGraphResponse(BaseModel):
    nodes: list[MindMapNodeResponse]
    edges: list[MindMapEdgeResponse]


class MindMapResponse(BaseModel):
    id: int
    job_id: int
    doc_id: int | None = None
    content_hash: str
    graph: MindMapGraphResponse
    node_sources: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    cached: bool = False
    task_id: str | None = None


class MindMapJobStartResponse(BaseModel):
    task_id: str
    status: str


class MindMapJobStatusResponse(BaseModel):
    task_id: str
    status: str
    error: str | None = None
    graph: MindMapGraphResponse | None = None


class MindMapNodeSource(BaseModel):
    index: int
    filename: str
    snippet: str
    doc_id: int | None = None
    page_number: int | None = None


class MindMapNodeInfoResponse(BaseModel):
    node_id: str
    encyclopedic: str
    interview_prep: str
    sources: list[MindMapNodeSource] = Field(default_factory=list)


class MindMapChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class MindMapChatResponse(BaseModel):
    answer: str
    sources: list[MindMapNodeSource] = Field(default_factory=list)


_mindmap_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _prune_flashcard_jobs_locked(now: datetime) -> None:
    stale_job_ids: list[str] = []
    for job_id, payload in _flashcard_jobs.items():
        updated = _parse_iso(payload.get("updated_at"))
        if updated is None:
            stale_job_ids.append(job_id)
            continue
        if (now - updated).total_seconds() > FLASHCARD_JOB_TTL_SECONDS:
            stale_job_ids.append(job_id)
    for job_id in stale_job_ids:
        _flashcard_jobs.pop(job_id, None)


async def _run_flashcard_generation_job(job_id: str, payload: FlashcardRequest) -> None:
    async with _flashcard_jobs_lock:
        job = _flashcard_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["updated_at"] = _now_iso()

    db = SessionLocal()
    try:
        result = await flashcards.create_study_card_set(
            db,
            job_id=payload.job_id,
            name=payload.name,
            topic=payload.topic,
            num_cards=payload.num_cards,
            document_ids=payload.document_ids,
            generate_per_section=payload.generate_per_section,
        )
        validated = FlashcardSetResponse.model_validate(result).model_dump(mode="json")
    except ValueError as exc:
        async with _flashcard_jobs_lock:
            job = _flashcard_jobs.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["updated_at"] = _now_iso()
    except Exception:
        async with _flashcard_jobs_lock:
            job = _flashcard_jobs.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = "Unexpected failure during flashcard generation."
                job["updated_at"] = _now_iso()
    else:
        async with _flashcard_jobs_lock:
            job = _flashcard_jobs.get(job_id)
            if job is not None:
                job["status"] = "completed"
                job["result"] = validated
                job["error"] = None
                job["updated_at"] = _now_iso()
    finally:
        db.close()


def _track_job_task(task: asyncio.Task[None]) -> None:
    _flashcard_job_tasks.add(task)
    task.add_done_callback(_flashcard_job_tasks.discard)


@router.get("/card-sets", response_model=list[StudyCardSetSummaryResponse])
def list_card_sets(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[StudyCardSetSummaryResponse]:
    payload = flashcards.list_study_card_sets(db, limit=limit)
    return [StudyCardSetSummaryResponse.model_validate(item) for item in payload]


@router.get("/card-sets/{card_set_id}", response_model=StudyCardSetDetailResponse)
def get_card_set(
    card_set_id: int,
    db: Session = Depends(get_db),
) -> StudyCardSetDetailResponse:
    try:
        payload = flashcards.get_study_card_set_cards(db, card_set_id=card_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return StudyCardSetDetailResponse.model_validate(payload)


@router.post("/flashcards", response_model=FlashcardSetResponse)
async def create_flashcards(
    payload: FlashcardRequest,
    db: Session = Depends(get_db),
) -> FlashcardSetResponse:
    try:
        result = await flashcards.create_study_card_set(
            db,
            job_id=payload.job_id,
            name=payload.name,
            topic=payload.topic,
            num_cards=payload.num_cards,
            document_ids=payload.document_ids,
            generate_per_section=payload.generate_per_section,
        )
    except ValueError as exc:
        detail = str(exc)
        if "Job not found" in detail:
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=400, detail=detail)

    return FlashcardSetResponse.model_validate(result)


@router.post("/flashcards/async", response_model=FlashcardAsyncStartResponse)
async def create_flashcards_async(
    payload: FlashcardRequest,
) -> FlashcardAsyncStartResponse:
    job_id = str(uuid4())
    now_iso = _now_iso()
    async with _flashcard_jobs_lock:
        _prune_flashcard_jobs_locked(datetime.now(timezone.utc))
        _flashcard_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": now_iso,
            "updated_at": now_iso,
            "error": None,
            "result": None,
        }
    task = asyncio.create_task(_run_flashcard_generation_job(job_id, payload))
    _track_job_task(task)
    return FlashcardAsyncStartResponse(job_id=job_id, status="pending")


@router.get("/flashcards/jobs/{job_id}", response_model=FlashcardJobStatusResponse)
async def get_flashcard_job_status(job_id: str) -> FlashcardJobStatusResponse:
    async with _flashcard_jobs_lock:
        _prune_flashcard_jobs_locked(datetime.now(timezone.utc))
        payload = _flashcard_jobs.get(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Flashcard generation job not found")
        return FlashcardJobStatusResponse.model_validate(payload)


@router.patch("/card-sets/{card_set_id}", response_model=StudyCardSetSummaryResponse)
def rename_card_set(
    card_set_id: int,
    payload: StudyCardSetRenameRequest,
    db: Session = Depends(get_db),
) -> StudyCardSetSummaryResponse:
    try:
        flashcards.update_study_card_set_name(db, card_set_id=card_set_id, name=payload.name)
        summary = flashcards.get_study_card_set_summary(db, card_set_id=card_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return StudyCardSetSummaryResponse.model_validate(summary)


@router.delete("/card-sets/{card_set_id}")
def delete_card_set(
    card_set_id: int,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    try:
        flashcards.delete_study_card_set(db, card_set_id=card_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "deleted"}


@router.patch("/cards/{card_id}/review", response_model=FlashcardResponse)
def review_flashcard(
    card_id: int,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
) -> FlashcardResponse:
    try:
        card = flashcards.review_study_card(db, card_id=card_id, rating=payload.rating)
    except ValueError as exc:
        detail = str(exc)
        if "not found" in detail.lower():
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=400, detail=detail)
    return FlashcardResponse.model_validate(flashcards.serialize_study_card(card))


@router.post("/mindmap", response_model=MindMapJobStartResponse)
async def start_mindmap_job(
    payload: MindMapRequest,
    db: Session = Depends(get_db),
) -> MindMapJobStartResponse:
    if payload.job_id is None and payload.doc_id is None:
        raise HTTPException(status_code=400, detail="Provide a job id, a document, or both.")
    try:
        task_id = mindmap.create_mindmap_job(db, job_id=payload.job_id, doc_id=payload.doc_id)
    except ValueError as exc:
        detail = str(exc)
        raise HTTPException(status_code=404 if "not found" in detail.lower() else 400, detail=detail)
    task = asyncio.create_task(mindmap.run_mindmap_job(task_id, job_id=payload.job_id, doc_id=payload.doc_id))
    _mindmap_tasks.add(task)
    task.add_done_callback(_mindmap_tasks.discard)
    return MindMapJobStartResponse(task_id=task_id, status="pending")


@router.get("/mindmap/latest", response_model=MindMapJobStatusResponse)
def get_latest_mindmap_job(
    job_id: int | None = Query(default=None, ge=1),
    doc_id: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> MindMapJobStatusResponse:
    result = mindmap.get_latest_mindmap_job(db, job_id=job_id, doc_id=doc_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No completed mind map job found")
    graph: MindMapGraphResponse | None = None
    if result.get("graph"):
        graph = MindMapGraphResponse.model_validate(result["graph"])
    return MindMapJobStatusResponse(
        task_id=str(result["task_id"]),
        status=str(result["status"]),
        error=result.get("error"),  # type: ignore[arg-type]
        graph=graph,
    )


@router.get("/mindmap/status/{task_id}", response_model=MindMapJobStatusResponse)
def get_mindmap_job_status(task_id: str, db: Session = Depends(get_db)) -> MindMapJobStatusResponse:
    try:
        result = mindmap.get_mindmap_job_status(db, task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    graph: MindMapGraphResponse | None = None
    if result.get("graph"):
        graph = MindMapGraphResponse.model_validate(result["graph"])
    return MindMapJobStatusResponse(
        task_id=str(result["task_id"]),
        status=str(result["status"]),
        error=result.get("error"),  # type: ignore[arg-type]
        graph=graph,
    )


@router.get("/mindmap/{task_id}/node/{node_id}", response_model=MindMapNodeInfoResponse)
async def get_node_info(task_id: str, node_id: str, db: Session = Depends(get_db)) -> MindMapNodeInfoResponse:
    try:
        result = await mindmap.get_or_generate_node_info(db, task_id=task_id, node_id=node_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return MindMapNodeInfoResponse(
        node_id=node_id,
        encyclopedic=str(result["encyclopedic"]),
        interview_prep=str(result["interview_prep"]),
        sources=[MindMapNodeSource.model_validate(s) for s in (result.get("sources") or [])],
    )


@router.post("/mindmap/{task_id}/chat", response_model=MindMapChatResponse)
async def chat_mindmap(
    task_id: str,
    payload: MindMapChatRequest,
    db: Session = Depends(get_db),
) -> MindMapChatResponse:
    try:
        result = await mindmap.chat_with_mindmap(db, task_id=task_id, message=payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return MindMapChatResponse(
        answer=str(result["answer"]),
        sources=[MindMapNodeSource.model_validate(s) for s in (result.get("sources") or [])],
    )


@router.get("/mindmap", response_model=MindMapResponse)
def get_mindmap(
    job_id: int = Query(..., ge=1),
    doc_id: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> MindMapResponse:
    try:
        result = mindmap.get_cached_mind_map(
            db,
            job_id=job_id,
            doc_id=doc_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if "not found" in detail.lower():
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=400, detail=detail)
    return MindMapResponse.model_validate({**result, "cached": True})
