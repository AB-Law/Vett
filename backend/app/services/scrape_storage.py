"""Helpers for normalizing and persisting scraped job payloads."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.score import Job, ScrapedJob
from ..scrapers.linkedin import map_item


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    if isinstance(value, str):
        normalised = value.strip()
        if not normalised:
            return None

        normalized = normalised
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass

        known_formats = (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
        )
        for fmt in known_formats:
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                pass

        relative = normalized.lower()
        if relative in {"today", "just now"}:
            return datetime.utcnow()
        if relative == "yesterday":
            return datetime.utcnow() - timedelta(days=1)

        match = re.match(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", relative)
        if match:
            value_str, unit = match.groups()
            amount = int(value_str)
            if unit == "second":
                return datetime.utcnow() - timedelta(seconds=amount)
            if unit == "minute":
                return datetime.utcnow() - timedelta(minutes=amount)
            if unit == "hour":
                return datetime.utcnow() - timedelta(hours=amount)
            if unit == "day":
                return datetime.utcnow() - timedelta(days=amount)
            if unit == "week":
                return datetime.utcnow() - timedelta(weeks=amount)
            if unit == "month":
                return datetime.utcnow() - timedelta(days=amount * 30)
            if unit == "year":
                return datetime.utcnow() - timedelta(days=amount * 365)
    return None


def _match_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return None
    return str(value).strip() or None


def _match_int(value: object) -> int | None:
    matched = _match_value(value)
    if matched is None:
        return None
    digits = "".join(ch for ch in matched if ch.isdigit())
    return int(digits) if digits else None


def _canonicalize_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    try:
        parts = urlsplit(stripped)
        if parts.scheme and parts.netloc:
            normalized_path = parts.path.rstrip("/") or "/"
            return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))
        return stripped
    except Exception:
        return stripped


def store_scrape_results(
    db: Session,
    scrape_request_id: int,
    source: str,
    items: list[dict[str, object]],
    store_mapped: bool = True,
) -> int:
    stored_count = 0
    for item in items:
        mapped = map_item(item)

        def _match_existing(
            mapped_external_id: str | None,
            mapped_url: str | None,
            mapped_canonical_url: str | None,
            source_value: str,
        ):
            if mapped_external_id:
                return and_(Job.source == source_value, Job.external_job_id == mapped_external_id)
            if mapped_canonical_url:
                return and_(Job.source == source_value, or_(Job.canonical_url == mapped_canonical_url, Job.url.like(f"{mapped_canonical_url}%")))
            if mapped_url:
                canonicalized = _canonicalize_url(mapped_url)
                matchers = [Job.url == mapped_url]
                if canonicalized is not None:
                    matchers.append(Job.url == canonicalized)
                    matchers.append(Job.canonical_url == canonicalized)
                    matchers.append(Job.url.like(f"{canonicalized}%"))
                return and_(Job.source == source_value, or_(*matchers))
            filters = [
                Job.source == source_value,
                Job.url.is_(None),
            ]
            identity_count = 0
            if mapped.get("title"):
                identity_count += 1
                filters.append(Job.title == mapped.get("title"))
            if mapped.get("company"):
                identity_count += 1
                filters.append(Job.company == mapped.get("company"))
            if mapped.get("location"):
                identity_count += 1
                filters.append(Job.location == mapped.get("location"))
            if identity_count == 0:
                return None
            return and_(*filters)

        if store_mapped:
            where_clause = _match_existing(
                _match_value(mapped.get("external_job_id")),
                _match_value(mapped.get("url")),
                _match_value(mapped.get("canonical_url")),
                source,
            )
            existing = db.query(Job).filter(where_clause).first() if where_clause is not None else None
            if existing is None:
                existing = Job(
                    external_job_id=_match_value(mapped.get("external_job_id")),
                    canonical_url=_match_value(mapped.get("canonical_url")),
                    title=mapped.get("title"),
                    company=mapped.get("company"),
                    location=mapped.get("location"),
                    url=mapped.get("url"),
                    description=mapped.get("description"),
                    source=source,
                    work_type=_match_value(mapped.get("work_type")),
                    employment_type=_match_value(mapped.get("employment_type")),
                    job_function=_match_value(mapped.get("job_function")),
                    industries=_match_value(mapped.get("industries")),
                    applicants_count=_match_value(mapped.get("applicants_count")),
                    benefits=mapped.get("benefits"),
                    salary=_match_value(mapped.get("salary")),
                    company_logo=_match_value(mapped.get("company_logo")),
                    company_linkedin_url=_match_value(mapped.get("company_linkedin_url")),
                    company_website=_match_value(mapped.get("company_website")),
                    company_address=mapped.get("company_address"),
                    company_employees_count=_match_int(mapped.get("company_employees_count")),
                    seniority=mapped.get("seniority"),
                    posted_at=_parse_datetime(mapped.get("posted_date")),
                    posted_at_raw=_match_value(mapped.get("posted_at_raw")),
                    fit_score=None,
                    matched_keywords=[],
                    missing_keywords=[],
                    gap_analysis=None,
                    scored_at=None,
                )
                db.add(existing)
            else:
                existing.title = mapped.get("title") or existing.title
                existing.company = mapped.get("company") or existing.company
                existing.location = mapped.get("location") or existing.location
                existing.external_job_id = _match_value(mapped.get("external_job_id")) or existing.external_job_id
                canonical_url = _match_value(mapped.get("canonical_url"))
                if canonical_url is not None:
                    existing.canonical_url = canonical_url
                existing.source = source
                existing.seniority = mapped.get("seniority") or existing.seniority
                existing.work_type = _match_value(mapped.get("work_type")) or existing.work_type
                existing.employment_type = _match_value(mapped.get("employment_type")) or existing.employment_type
                existing.job_function = _match_value(mapped.get("job_function")) or existing.job_function
                existing.industries = _match_value(mapped.get("industries")) or existing.industries
                existing.applicants_count = _match_value(mapped.get("applicants_count")) or existing.applicants_count
                if mapped.get("benefits"):
                    existing.benefits = mapped.get("benefits")
                existing.salary = _match_value(mapped.get("salary")) or existing.salary
                existing.company_logo = _match_value(mapped.get("company_logo")) or existing.company_logo
                existing.company_linkedin_url = _match_value(mapped.get("company_linkedin_url")) or existing.company_linkedin_url
                existing.company_website = _match_value(mapped.get("company_website")) or existing.company_website
                if mapped.get("company_address"):
                    existing.company_address = mapped.get("company_address")
                company_employee_count = _match_int(mapped.get("company_employees_count"))
                if company_employee_count is not None:
                    existing.company_employees_count = company_employee_count
                parsed_posted = _parse_datetime(mapped.get("posted_date"))
                if parsed_posted is not None:
                    existing.posted_at = parsed_posted
                if mapped.get("posted_at_raw") is not None:
                    existing.posted_at_raw = _match_value(mapped.get("posted_at_raw"))
                if mapped.get("job_poster_name"):
                    existing.job_poster_name = _match_value(mapped.get("job_poster_name"))
                if mapped.get("job_poster_title"):
                    existing.job_poster_title = _match_value(mapped.get("job_poster_title"))
                if mapped.get("job_poster_profile_url"):
                    existing.job_poster_profile_url = _match_value(mapped.get("job_poster_profile_url"))
                if mapped.get("description"):
                    existing.description = mapped["description"] or existing.description

        db.add(
            ScrapedJob(
                scrape_request_id=scrape_request_id,
                source=source,
                title=mapped.get("title"),
                company=mapped.get("company"),
                location=mapped.get("location"),
                url=mapped.get("url"),
                posted_at=_parse_datetime(mapped.get("posted_date")),
                seniority=mapped.get("seniority"),
                description=mapped.get("description"),
                raw_payload=item,
            )
        )
        stored_count += 1

    return stored_count
