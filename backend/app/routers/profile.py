"""Candidate profile endpoints."""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from sqlalchemy.orm import Session

from ..database import get_db
from ..models.user_profile import UserProfile


router = APIRouter(prefix="/profile", tags=["profile"])


class UserProfileResponse(BaseModel):
    id: int | None
    full_name: str
    headline_or_target_role: str
    current_company: str
    years_experience: int | None
    top_skills: list[str]
    location: str
    linkedin_url: str
    summary: str
    source: str


class UserProfileUpdate(BaseModel):
    full_name: str | None = None
    headline_or_target_role: str | None = None
    current_company: str | None = None
    years_experience: int | None = None
    top_skills: list[str] | None = None
    location: str | None = None
    linkedin_url: str | None = None
    summary: str | None = None
    source: str | None = None


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = str(raw).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(text)
    return output


def _coerce_profile_payload(payload: UserProfileUpdate | None) -> dict[str, Any]:
    if payload is None:
        return {}
    raw = payload.model_dump(exclude_unset=True)
    output: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "top_skills":
            output[key] = _coerce_string_list(value)
            continue
        if key == "years_experience":
            if value is None or str(value).strip() == "":
                output[key] = None
                continue
            try:
                output[key] = int(value)
            except Exception:
                output[key] = None
            continue
        output[key] = str(value).strip()
    return output


def _normalize_profile(profile: UserProfile | None) -> UserProfileResponse:
    if profile is None:
        return UserProfileResponse(
            id=None,
            full_name="",
            headline_or_target_role="",
            current_company="",
            years_experience=None,
            top_skills=[],
            location="",
            linkedin_url="",
            summary="",
            source="",
        )
    return UserProfileResponse(
        id=profile.id,
        full_name=profile.full_name or "",
        headline_or_target_role=profile.headline_or_target_role or "",
        current_company=profile.current_company or "",
        years_experience=profile.years_experience,
        top_skills=_coerce_string_list(profile.top_skills or []),
        location=profile.location or "",
        linkedin_url=profile.linkedin_url or "",
        summary=profile.summary or "",
        source=profile.source or "manual",
    )


def _get_profile(db: Session) -> UserProfile | None:
    return db.query(UserProfile).order_by(UserProfile.id.asc()).first()


@router.get("/", response_model=UserProfileResponse)
def get_profile(db: Session = Depends(get_db)):
    profile = _get_profile(db)
    return _normalize_profile(profile)


@router.post("/", response_model=UserProfileResponse)
def update_profile(payload: UserProfileUpdate, db: Session = Depends(get_db)):
    profile = _get_profile(db)
    if profile is None:
        profile = UserProfile(top_skills=[])
        db.add(profile)

    normalized = _coerce_profile_payload(payload)
    for field, value in normalized.items():
        setattr(profile, field, value)
    if normalized:
        profile.source = normalized.get("source") or "manual"

    db.commit()
    db.refresh(profile)
    return _normalize_profile(profile)
