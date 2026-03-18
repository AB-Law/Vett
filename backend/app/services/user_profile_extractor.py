"""LLM-based extraction utilities for global user profile."""

from __future__ import annotations

import json
import re
from typing import Any

from ..models.user_profile import UserProfile
from .llm import _extract_first_json_object, _sanitize_json_token_stream, _get_litellm_model
from sqlalchemy.orm import Session

PROFILE_EXTRACTION_SYSTEM_PROMPT = """You are a strict JSON extractor for candidate profile data.

Return a single JSON object with keys:
- full_name: string
- headline_or_target_role: string
- current_company: string
- years_experience: integer
- top_skills: array of strings
- location: string
- linkedin_url: string
- summary: string

Extract only what is explicitly supported by the CV text.
If a field is missing, set it to "" for strings and null for years_experience.
Keep top_skills concise and unique, max 12 entries.
"""


def _sanitize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:2000].strip()


def _sanitize_skills(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _sanitize_text(item)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
        if len(normalized) >= 12:
            break
    return normalized


def _sanitize_profile_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, Any] = {}
    full_name = _sanitize_text(raw.get("full_name"))
    headline = _sanitize_text(raw.get("headline_or_target_role"))
    current_company = _sanitize_text(raw.get("current_company"))
    location = _sanitize_text(raw.get("location"))
    linkedin_url = _sanitize_text(raw.get("linkedin_url"))
    summary = _sanitize_text(raw.get("summary"))
    years = raw.get("years_experience")
    parsed["full_name"] = full_name
    parsed["headline_or_target_role"] = headline
    parsed["current_company"] = current_company
    parsed["location"] = location
    parsed["linkedin_url"] = linkedin_url
    parsed["summary"] = summary
    parsed["top_skills"] = _sanitize_skills(raw.get("top_skills"))
    if years in (None, ""):
        parsed["years_experience"] = None
    else:
        try:
            parsed["years_experience"] = int(years)
        except Exception:
            parsed["years_experience"] = None
    return parsed


def _coerce_candidate_years(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        normalized = int(value)
    except Exception:
        match = re.search(r"(\d+)", str(value))
        if not match:
            return None
        normalized = int(match.group(1))
    return max(0, normalized)


def _fallback_profile_from_cv_text(profile_text: str) -> dict[str, Any]:
    text = profile_text.strip()
    if not text:
        return {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_lines = lines[:8]
    joined = " ".join(first_lines)
    return {
        "full_name": "",
        "headline_or_target_role": "",
        "current_company": "",
        "years_experience": None,
        "top_skills": _sanitize_skills(first_lines),
        "location": "",
        "linkedin_url": "",
        "summary": joined[:1600],
    }


def _extract_first_json(payload: str) -> dict[str, Any]:
    parsed = _extract_first_json_object(payload)
    if not parsed:
        return {}
    try:
        data = json.loads(parsed)
    except Exception:
        try:
            data = json.loads(_sanitize_json_token_stream(parsed))
        except Exception:
            return {}
    if isinstance(data, dict):
        return data
    return {}


async def extract_profile_from_cv_text(profile_text: str) -> dict[str, Any]:
    """Extract profile data from raw CV text.

    Returns a sanitized dict matching UserProfile model fields.
    """
    cv_text = profile_text.strip()
    if not cv_text:
        return {}
    try:
        import litellm  # type: ignore
    except Exception:
        return _sanitize_profile_payload(_fallback_profile_from_cv_text(cv_text))

    model, kwargs = _get_litellm_model()
    prompt = f"{PROFILE_EXTRACTION_SYSTEM_PROMPT}\n\nCV:\n{cv_text[:9000]}"
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured profile information from CVs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
            **kwargs,
        )
    except Exception:
        return _sanitize_profile_payload(_fallback_profile_from_cv_text(cv_text))

    raw = str(getattr(response.choices[0].message, "content", "") if response and response.choices else "")
    payload = _extract_first_json(raw)
    if not isinstance(payload, dict):
        payload = _fallback_profile_from_cv_text(cv_text)
    sanitized = _sanitize_profile_payload(payload)
    if not sanitized.get("summary"):
        fallback = _fallback_profile_from_cv_text(cv_text)
        sanitized["summary"] = fallback.get("summary", "")
    if sanitized.get("years_experience") is not None:
        sanitized["years_experience"] = _coerce_candidate_years(sanitized["years_experience"])
    return sanitized


def _coerce_source_value(value: str | None) -> str:
    if not value:
        return "cv_llm"
    return value.strip().lower() or "cv_llm"


def _get_or_create_profile(db: Session) -> UserProfile:
    profile = db.query(UserProfile).order_by(UserProfile.id.asc()).first()
    if profile is None:
        profile = UserProfile(top_skills=[])
        db.add(profile)
        db.flush()
    return profile


def upsert_user_profile_from_extraction(
    db: Session,
    payload: dict[str, Any],
    source: str = "cv_llm",
) -> UserProfile:
    profile = _get_or_create_profile(db)
    sanitized = _sanitize_profile_payload(payload)
    profile.full_name = sanitized.get("full_name", "")
    profile.headline_or_target_role = sanitized.get("headline_or_target_role", "")
    profile.current_company = sanitized.get("current_company", "")
    profile.years_experience = sanitized.get("years_experience")
    profile.top_skills = sanitized.get("top_skills", [])
    profile.location = sanitized.get("location", "")
    profile.linkedin_url = sanitized.get("linkedin_url", "")
    profile.summary = sanitized.get("summary", "")
    profile.source = _coerce_source_value(source)
    db.commit()
    db.refresh(profile)
    return profile
