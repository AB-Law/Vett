from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from ..database import get_db
from ..models.cv import CV
from ..models.score import ScoreHistory
from ..services.llm import score_cv_against_jd
from ..config import get_settings

router = APIRouter(prefix="/score", tags=["score"])


class ScoreRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = None
    company: Optional[str] = None


class ScoreResponse(BaseModel):
    id: Optional[int] = None
    fit_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    gap_analysis: str
    rewrite_suggestions: list[str]
    job_title: Optional[str] = None
    company: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class HistoryItem(BaseModel):
    id: int
    job_title: Optional[str]
    company: Optional[str]
    fit_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    gap_analysis: Optional[str]
    rewrite_suggestions: list[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


@router.post("/", response_model=ScoreResponse)
async def score_jd(req: ScoreRequest, db: Session = Depends(get_db)):
    cv = db.query(CV).order_by(CV.id.desc()).first()
    if not cv:
        raise HTTPException(400, "No CV uploaded. Please upload a CV first.")

    result = await score_cv_against_jd(cv.parsed_text, req.job_description)

    settings = get_settings()

    # Save to history if enabled
    if settings.save_history:
        entry = ScoreHistory(
            job_title=req.job_title,
            company=req.company,
            job_description=req.job_description,
            fit_score=result["fit_score"],
            matched_keywords=result["matched_keywords"],
            missing_keywords=result["missing_keywords"],
            gap_analysis=result["gap_analysis"],
            rewrite_suggestions=result["rewrite_suggestions"],
            llm_provider=settings.active_llm_provider,
            llm_model=_current_model(settings),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        result["id"] = entry.id

    result["llm_provider"] = settings.active_llm_provider
    result["llm_model"] = _current_model(settings)
    result.setdefault("job_title", req.job_title)
    result.setdefault("company", req.company)

    return ScoreResponse(**result)


@router.get("/history", response_model=list[HistoryItem])
def get_history(db: Session = Depends(get_db)):
    items = (
        db.query(ScoreHistory)
        .order_by(ScoreHistory.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        HistoryItem(
            id=i.id,
            job_title=i.job_title,
            company=i.company,
            fit_score=i.fit_score,
            matched_keywords=i.matched_keywords or [],
            missing_keywords=i.missing_keywords or [],
            gap_analysis=i.gap_analysis,
            rewrite_suggestions=i.rewrite_suggestions or [],
            llm_provider=i.llm_provider,
            llm_model=i.llm_model,
            created_at=str(i.created_at),
        )
        for i in items
    ]


@router.delete("/history/{item_id}")
def delete_history_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ScoreHistory).filter(ScoreHistory.id == item_id).first()
    if not item:
        raise HTTPException(404, "Not found")
    db.delete(item)
    db.commit()
    return {"deleted": item_id}


@router.delete("/history")
def clear_history(db: Session = Depends(get_db)):
    deleted = db.query(ScoreHistory).delete()
    db.commit()
    return {"deleted": deleted}


def _current_model(settings) -> str:
    p = settings.active_llm_provider.lower()
    mapping = {
        "claude": settings.claude_model,
        "openai": settings.openai_model,
        "azure_openai": settings.azure_openai_deployment,
        "ollama": settings.ollama_model,
        "lm_studio": settings.lm_studio_model,
    }
    return mapping.get(p, "unknown")
