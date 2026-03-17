from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional

from ..database import get_db
from ..models.cv import CV
from ..models.score import ScoreHistory
from ..services.llm import (
    extract_role_signal_map,
    generate_agent_score_plan,
    generate_cv_rewrite_proposals,
    score_cv_against_jd,
)
from ..config import get_settings

router = APIRouter(prefix="/score", tags=["score"])


class ScoreRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = None
    company: Optional[str] = None


class AgentPlan(BaseModel):
    """Structured follow-up plan created by the second local scoring pass."""

    role_signal_map: dict[str, Any] = Field(default_factory=dict)
    skills_to_fix_first: list[str] = Field(default_factory=list)
    concrete_edit_actions: list[str] = Field(default_factory=list)
    interview_topics_to_prioritize: list[str] = Field(default_factory=list)
    study_order: list[str] = Field(default_factory=list)


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
    agent_plan: Optional[AgentPlan] = None


class RewriteProposal(BaseModel):
    before: str
    after: str
    reason: str
    risk_or_uncertainty: str


class CVRewriteRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = None
    company: Optional[str] = None


class CVRewriteResponse(BaseModel):
    proposals: list[RewriteProposal]
    job_title: Optional[str] = None
    company: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class HistoryItem(BaseModel):
    id: int
    job_title: Optional[str]
    company: Optional[str]
    fit_score: float
    agent_plan: Optional[dict[str, Any]] = None
    matched_keywords: list[str]
    missing_keywords: list[str]
    gap_analysis: Optional[str]
    rewrite_suggestions: list[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    created_at: str

    model_config = ConfigDict(from_attributes=True)


def _fallback_agent_plan(
    role_signal_map: dict[str, Any] | None,
    score_result: dict[str, Any],
) -> dict[str, Any]:
    """Create a deterministic fallback plan from scoring artifacts."""
    safe_map = role_signal_map or {}
    missing = score_result.get("missing_keywords", [])
    gaps = missing if isinstance(missing, list) else []

    return {
        "role_signal_map": _coerce_object_map(safe_map),
        "skills_to_fix_first": _coerce_string_items(gaps)[:6],
        "concrete_edit_actions": [
            "Use one of the top 3 missing keywords in the professional summary.",
            "Add one quantified result for recent work aligned to missing requirements.",
            "Add one project bullet that demonstrates the strongest missing skill.",
        ],
        "interview_topics_to_prioritize": _coerce_string_items(score_result.get("missing_keywords"))[:6],
        "study_order": [
            "Identify the top 3 missing keywords",
            "Draft one STAR proof point per missing skill",
            "Run one mock interview with those topics in sequence",
        ],
    }


def _coerce_rewrite_proposals_values(values: Any) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []

    proposals: list[dict[str, str]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before", "")).strip()
        after = str(item.get("after", "")).strip()
        reason = str(item.get("reason", "")).strip()
        risk_or_uncertainty = str(item.get("risk_or_uncertainty", "")).strip()
        if not before or not after or not reason or not risk_or_uncertainty:
            continue
        proposals.append(
            {
                "before": before,
                "after": after,
                "reason": reason,
                "risk_or_uncertainty": risk_or_uncertainty,
            }
        )
    return proposals


def _coerce_string_items(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    return []


def _coerce_object_map(values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        return {str(k): v for k, v in values.items() if str(k).strip()}
    return {}


@router.post("/", response_model=ScoreResponse)
async def score_jd(req: ScoreRequest, db: Session = Depends(get_db)):
    cv = db.query(CV).order_by(CV.id.desc()).first()
    if not cv:
        raise HTTPException(400, "No CV uploaded. Please upload a CV first.")

    try:
        role_signal_map = await extract_role_signal_map(cv.parsed_text, req.job_description)
    except Exception:
        role_signal_map = {}
    result = await score_cv_against_jd(cv.parsed_text, req.job_description, role_signal_map=role_signal_map)
    try:
        agent_plan = await generate_agent_score_plan(
            cv_text=cv.parsed_text,
            jd_text=req.job_description,
            score_payload=result,
            role_signal_map=role_signal_map,
        )
    except Exception:
        agent_plan = _fallback_agent_plan(role_signal_map, result)

    result["agent_plan"] = agent_plan

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
            agent_plan=result.get("agent_plan"),
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


@router.post("/rewrite-proposals", response_model=CVRewriteResponse)
async def rewrite_proposals(req: CVRewriteRequest, db: Session = Depends(get_db)):
    cv = db.query(CV).order_by(CV.id.desc()).first()
    if not cv:
        raise HTTPException(400, "No CV uploaded. Please upload a CV first.")

    try:
        role_signal_map = await extract_role_signal_map(cv.parsed_text, req.job_description)
    except Exception:
        role_signal_map = {}

    try:
        score_payload = await score_cv_against_jd(cv.parsed_text, req.job_description, role_signal_map=role_signal_map)
    except Exception:
        score_payload = {}

    try:
        proposals = await generate_cv_rewrite_proposals(
            cv_text=cv.parsed_text,
            jd_text=req.job_description,
            role_signal_map=role_signal_map,
            score_payload=score_payload,
        )
        proposals = _coerce_rewrite_proposals_values(proposals)
    except Exception:
        proposals = [
            {
                "before": "Could not generate a structured rewrite proposal.",
                "after": "Retry after checking LLM connectivity and credentials.",
                "reason": "Agent execution failed during proposal generation.",
                "risk_or_uncertainty": "High risk of no-op if provider response is unavailable.",
            }
        ]

    settings = get_settings()
    return CVRewriteResponse(
        proposals=proposals,
        job_title=req.job_title,
        company=req.company,
        llm_provider=settings.active_llm_provider,
        llm_model=_current_model(settings),
    )


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
            agent_plan=i.agent_plan,
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
