#!/usr/bin/env python3
"""Backfill missing job metadata in the normalized jobs table from scraped_payload rows.

Usage:
  python backend/scripts/backfill_jobs.py --apply
  python backend/scripts/backfill_jobs.py --source linkedin --apply --limit 200
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal
from app.models.score import Job, ScrapedJob
from app.scrapers.linkedin import map_item


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


def _normalise_url(value: str | None) -> str | None:
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


def _to_int_or_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value))
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return m.group(0)
    return None


def _to_text_or_csv(value: object) -> str | None:
    if isinstance(value, (list, tuple)):
        out = [x for x in [_match_value(item) for item in value] if x]
        return ", ".join(out) or None
    return _match_value(value)


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
            amount_str, unit = match.groups()
            amount = int(amount_str)
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


def _extract_scraped_fields(payload: dict[str, object]) -> dict[str, object | None]:
    mapped = map_item(payload)
    if not isinstance(mapped, dict):
        mapped = {}

    # map_item is great for raw actor payloads; for already-mapped payloads we add fallback fields
    fields = {
        "external_job_id": _match_value(payload.get("external_job_id")) or _match_value(payload.get("id")),
        "canonical_url": _normalise_url(_match_value(payload.get("canonical_url")) or _match_value(payload.get("url"))),
        "url": _match_value(payload.get("url")),
        "title": _match_value(payload.get("title")),
        "company": _match_value(payload.get("company")),
        "location": _match_value(payload.get("location")),
        "work_type": _match_value(payload.get("work_type")) or _match_value(payload.get("employmentType")),
        "employment_type": _match_value(payload.get("employment_type")) or _match_value(payload.get("employmentType")),
        "job_function": _match_value(payload.get("job_function")),
        "industries": _to_text_or_csv(payload.get("industries")),
        "applicants_count": _to_int_or_str(payload.get("applicants_count")) or _to_int_or_str(payload.get("applicantsCount")),
        "benefits": payload.get("benefits") if isinstance(payload.get("benefits"), list) else None,
        "salary": _match_value(payload.get("salary")),
        "company_logo": _match_value(payload.get("company_logo")),
        "company_linkedin_url": _match_value(payload.get("company_linkedin_url")),
        "company_website": _match_value(payload.get("company_website")),
        "company_address": payload.get("company_address") if isinstance(payload.get("company_address"), dict) else None,
        "company_employees_count": _match_int(payload.get("company_employees_count"))
        or _match_int(payload.get("companyEmployeesCount")),
        "job_poster_name": _match_value(payload.get("job_poster_name")),
        "job_poster_title": _match_value(payload.get("job_poster_title")),
        "job_poster_profile_url": _match_value(payload.get("job_poster_profile_url")),
    }

    # Merge map_item output (useful for raw actor payloads)
    fields.update({
        "external_job_id": _match_value(fields["external_job_id"]) or _match_value(mapped.get("external_job_id")),
        "canonical_url": _normalise_url(_match_value(fields["canonical_url"]) or _match_value(mapped.get("canonical_url"))),
        "url": _match_value(fields["url"]) or _match_value(mapped.get("url")),
        "title": _match_value(fields["title"]) or _match_value(mapped.get("title")),
        "company": _match_value(fields["company"]) or _match_value(mapped.get("company")),
        "location": _match_value(fields["location"]) or _match_value(mapped.get("location")),
        "work_type": _match_value(fields["work_type"]) or _match_value(mapped.get("work_type")) or _match_value(mapped.get("employment_type")),
        "employment_type": _match_value(fields["employment_type"]) or _match_value(mapped.get("employment_type")) or _match_value(mapped.get("work_type")),
        "job_function": _match_value(fields["job_function"]) or _match_value(mapped.get("job_function")),
        "industries": _match_value(fields["industries"]) or _to_text_or_csv(mapped.get("industries")),
        "applicants_count": _match_value(fields["applicants_count"]) or _match_value(mapped.get("applicants_count")),
        "benefits": fields["benefits"] if isinstance(fields["benefits"], list) else mapped.get("benefits"),
        "salary": _match_value(fields["salary"]) or _match_value(mapped.get("salary")),
        "company_logo": _match_value(fields["company_logo"]) or _match_value(mapped.get("company_logo")),
        "company_linkedin_url": _match_value(fields["company_linkedin_url"]) or _match_value(mapped.get("company_linkedin_url")),
        "company_website": _match_value(fields["company_website"]) or _match_value(mapped.get("company_website")),
        "company_address": fields["company_address"] if isinstance(fields["company_address"], dict) else mapped.get("company_address"),
        "company_employees_count": fields["company_employees_count"] or _match_int(mapped.get("company_employees_count")),
        "job_poster_name": _match_value(fields["job_poster_name"]) or _match_value(mapped.get("job_poster_name")),
        "job_poster_title": _match_value(fields["job_poster_title"]) or _match_value(mapped.get("job_poster_title")),
        "job_poster_profile_url": _match_value(fields["job_poster_profile_url"]) or _match_value(mapped.get("job_poster_profile_url")),
    })

    # Posted date + raw source text
    posted_raw = (
        _match_value(payload.get("posted_at_raw"))
        or _match_value(payload.get("posted_date"))
        or _match_value(payload.get("postedAt"))
        or _match_value(payload.get("publishedAt"))
        or _match_value(payload.get("postedDate"))
        or _match_value(payload.get("date"))
        or _match_value(payload.get("datePosted"))
        or _match_value(mapped.get("posted_at_raw"))
        or _match_value(mapped.get("posted_date"))
    )
    fields["posted_at_raw"] = posted_raw
    fields["posted_at"] = _parse_datetime(posted_raw)

    return fields


def _score_scraped_match(job: Job, mapped: dict[str, object | None]) -> int:
    score = 0
    job_external = _match_value(job.external_job_id)
    mapped_external = _match_value(mapped.get("external_job_id"))
    if job_external and mapped_external and job_external == mapped_external:
        score += 120

    job_url = _match_value(job.url)
    job_canonical = _match_value(job.canonical_url) or _normalise_url(job_url)
    mapped_url = _match_value(mapped.get("url"))
    mapped_canonical = _match_value(mapped.get("canonical_url")) or _normalise_url(mapped_url)

    if job_canonical and mapped_canonical:
        if job_canonical == mapped_canonical:
            score += 100
        elif mapped_canonical.startswith(job_canonical) or job_canonical.startswith(mapped_canonical):
            score += 60
    if job_url and mapped_url and (mapped_url == job_url or mapped_url.startswith(job_url) or job_url.startswith(mapped_url)):
        score += 50

    title_job = _match_value(job.title)
    title_payload = _match_value(mapped.get("title"))
    company_job = _match_value(job.company)
    company_payload = _match_value(mapped.get("company"))
    if title_job and title_payload and title_job.lower() == title_payload.lower():
        score += 25
    if company_job and company_payload and company_job.lower() == company_payload.lower():
        score += 25
    if _match_value(job.location) and _match_value(mapped.get("location")) and _match_value(job.location).lower() == _match_value(mapped.get("location")).lower():
        score += 10

    return score


def _collect_scraped_rows(db, source: str) -> list[ScrapedJob]:
    return (
        db.query(ScrapedJob)
        .filter(ScrapedJob.source == source)
        .filter(ScrapedJob.raw_payload.isnot(None))
        .order_by(ScrapedJob.created_at.desc())
        .all()
    )


def backfill_jobs(
    source: str = "linkedin",
    dry_run: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    db = SessionLocal()
    try:
        q = db.query(Job).filter(Job.source == source)
        target_jobs = q.all()
        scraped_rows = _collect_scraped_rows(db, source)

        updated = 0
        scanned = 0
        for job in target_jobs:
            if limit is not None and scanned >= limit:
                break
            scanned += 1

            candidates = []
            for scraped in scraped_rows:
                raw_payload = scraped.raw_payload
                if not isinstance(raw_payload, dict):
                    continue
                mapped = _extract_scraped_fields(raw_payload)
                score = _score_scraped_match(job, mapped)
                if score > 0:
                    candidates.append((score, mapped))
            if not candidates:
                continue
            mapped = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]

            changed = False

            if job.posted_at_raw is None and mapped.get("posted_at_raw"):
                job.posted_at_raw = _match_value(mapped.get("posted_at_raw"))
                changed = True

            if job.posted_at is None and mapped.get("posted_at"):
                posted = mapped["posted_at"]
                if isinstance(posted, datetime):
                    job.posted_at = posted
                    changed = True

            for field in [
                "external_job_id",
                "canonical_url",
                "work_type",
                "employment_type",
                "job_function",
                "industries",
                "salary",
                "company_logo",
                "company_linkedin_url",
                "company_website",
                "job_poster_name",
                "job_poster_title",
                "job_poster_profile_url",
            ]:
                current_value = getattr(job, field)
                if current_value is None and mapped.get(field) is not None:
                    setattr(job, field, _match_value(mapped.get(field)))
                    changed = True

            if job.applicants_count is None and mapped.get("applicants_count"):
                job.applicants_count = _to_int_or_str(mapped.get("applicants_count"))
                changed = True

            if job.benefits is None and isinstance(mapped.get("benefits"), list):
                job.benefits = mapped.get("benefits")
                changed = True

            if job.company_address is None and isinstance(mapped.get("company_address"), dict):
                job.company_address = mapped.get("company_address")
                changed = True

            if job.company_employees_count is None and mapped.get("company_employees_count") is not None:
                job.company_employees_count = _match_int(mapped.get("company_employees_count"))
                changed = True

            if changed:
                updated += 1

        if dry_run:
            db.rollback()
            return updated, scanned

        db.commit()
        return updated, scanned
    finally:
        db.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill existing job rows with richer fields from scraped payload history."
    )
    parser.add_argument("--source", default="linkedin", help="Target source (default: linkedin)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to scan")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write backfilled data to DB. If omitted, runs as dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    updated, scanned = backfill_jobs(
        source=args.source,
        dry_run=not args.apply,
        limit=args.limit,
    )
    if args.apply:
        print(f"Backfill complete. Scanned {scanned} jobs, updated {updated}.")
    else:
        print(f"Dry-run complete. Scanned {scanned} jobs, would update {updated}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

