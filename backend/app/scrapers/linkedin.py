"""LinkedIn jobs scraper powered by Apify.

Uses the Apify actor hKByXkMQaC5Qt9UMN (LinkedIn Jobs Scraper).
Requires APIFY_API_TOKEN to be set in the environment.
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote_plus, urlsplit, urlunsplit
from collections.abc import Iterable

from typing import Any

try:
    from apify_client import ApifyClient
except ImportError as exc:
    ApifyClient = None
    _APIFY_IMPORT_ERROR: Any = exc
else:
    _APIFY_IMPORT_ERROR = None

ACTOR_ID = "hKByXkMQaC5Qt9UMN"
DEFAULT_LIMIT = 25
MAX_LIMIT = 100

_SENIORITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("intern", r"\bintern(?:ship)?\b"),
    ("junior", r"\b(junior|jr\.?)\b"),
    ("associate", r"\bassociate\b"),
    ("mid", r"\bmid(?:-|\s*)level\b"),
    ("senior", r"\b(senior|sr\.?)\b"),
    ("lead", r"\blead\b"),
    ("principal", r"\bprincipal\b"),
    ("manager", r"\bmanager\b"),
    ("director", r"\bdirector\b"),
    ("vp", r"\bvice president\b|\bvp\b"),
    ("executive", r"\bexecutive\b"),
)


def _infer_seniority(*parts: str | None) -> str | None:
    haystack = " ".join(p.lower() for p in parts if p).strip()
    if not haystack:
        return None
    for level, pattern in _SENIORITY_PATTERNS:
        if re.search(pattern, haystack):
            return level
    return None


def _build_search_url(role: str, location: str | None) -> str:
    params = f"keywords={quote_plus(role)}"
    if location:
        params += f"&location={quote_plus(location)}"
    return f"https://www.linkedin.com/jobs/search/?{params}&position=1&pageNum=0"


def _normalise_years(years_of_experience: int | str | None) -> int | None:
    if years_of_experience is None:
        return None
    if isinstance(years_of_experience, bool):
        # bool is an int subclass; not a sensible experience value here.
        return None
    if isinstance(years_of_experience, int):
        return years_of_experience
    if isinstance(years_of_experience, str):
        m = re.search(r"\d+", years_of_experience)
        if not m:
            return None
        return int(m.group(0))
    if isinstance(years_of_experience, float):
        return int(years_of_experience)
    return None


def _build_keywords(
    role: str | None,
    job: str | None,
    years_of_experience: int | None,
) -> str:
    """Build a clean LinkedIn search string from user inputs."""
    components: list[str] = []
    seen: set[str] = set()

    for text in (role, job):
        normalised = _normalise(text)
        if not normalised:
            continue
        lowered = normalised.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        components.append(normalised)

    if years_of_experience is not None and years_of_experience >= 0:
        suffix = "year" if years_of_experience == 1 else "years"
        components.append(f"{years_of_experience} {suffix} experience")

    return " ".join(components)


def _normalise(value: object) -> str | None:
    if not value:
        return None
    if isinstance(value, dict):
        return None
    if isinstance(value, (list, tuple)):
        for candidate in value:
            text_value = _normalise(candidate)
            if text_value:
                return text_value
        return None
    text = str(value)
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed or None


def _normalise_text_or_csv(value: object) -> str | None:
    if isinstance(value, (list, tuple)):
        normalized = [_normalise(item) for item in value]
        parts = [item for item in normalized if item]
        return ", ".join(parts) or None
    return _normalise(value)


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


def _pick_first(item: dict[str, object], *keys: str) -> str | None:
    """Read the first non-empty string value from nested or top-level keys."""

    for key in keys:
        cursor: object = item
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        value = _normalise(cursor)
        if value:
            return value
        if isinstance(cursor, Iterable) and not isinstance(cursor, (str, bytes, dict)):
            for entry in cursor:
                value = _normalise(entry)
                if value:
                    return value
    return None


def _infer_job_url(item: dict[str, object], mapped_url: str | None) -> str | None:
    if mapped_url:
        normalized = mapped_url.strip()
        if normalized.startswith("/"):
            return f"https://www.linkedin.com{normalized}"
        return normalized

    job_id = _pick_first(item, "jobId", "job_id", "jobId", "id")
    if job_id:
        return f"https://www.linkedin.com/jobs/view/{quote_plus(job_id)}/"
    return None


def _normalize_company_address(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    keys = (
        "streetAddress",
        "addressLocality",
        "addressRegion",
        "postalCode",
        "addressCountry",
        "type",
    )
    out: dict[str, str] = {}
    for key in keys:
        normalized = _normalise(value.get(key))
        if normalized:
            out[key] = normalized
    return out or None


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


def _normalise_benefits(value: object) -> list[str] | None:
    if isinstance(value, list):
        normalized = [_normalise(item) for item in value]
        return [item for item in normalized if item]
    return None


def _map_item(item: dict) -> dict[str, object | None]:
    """Map an Apify actor result item to our internal ScrapedJob shape."""
    title = _pick_first(item, "title", "jobTitle", "position")
    company = _pick_first(item, "companyName", "company", "organization")
    location = _pick_first(item, "location", "jobLocation", "locationName")
    url = _pick_first(item, "jobUrl", "url", "applyUrl", "link", "jobLink", "job_link")
    url = _infer_job_url(item, url)
    canonical_url = _canonicalize_url(url)
    posted_date = _pick_first(item, "postedAt", "publishedAt", "postedDate", "date", "datePosted")
    description = _normalise(
        _pick_first(item, "descriptionText", "description", "jobDescription", "jobDescriptionText")
        or item.get("description")
    )
    external_id = _normalise(item.get("id"))
    # Apify actor may already expose seniority; fall back to inference.
    seniority = _normalise(item.get("seniorityLevel") or item.get("seniority")) or _infer_seniority(
        title, description
    )

    return {
        "external_job_id": external_id,
        "title": title,
        "company": company,
        "location": location,
        "url": url or "",
        "canonical_url": canonical_url,
        "posted_date": posted_date,
        "posted_at_raw": posted_date,
        "seniority": seniority,
        "description": description,
        "work_type": _normalise(item.get("employmentType")),
        "employment_type": _normalise(item.get("employmentType")),
        "job_function": _normalise(item.get("jobFunction")),
        "industries": _normalise_text_or_csv(item.get("industries")),
        "applicants_count": _to_int_or_str(item.get("applicantsCount")),
        "benefits": _normalise_benefits(item.get("benefits")),
        "salary": _normalise(item.get("salary")),
        "company_logo": _normalise(item.get("companyLogo")),
        "company_linkedin_url": _normalise(item.get("companyLinkedinUrl")),
        "company_website": _normalise(item.get("companyWebsite")),
        "company_address": _normalize_company_address(item.get("companyAddress")),
        "company_employees_count": item.get("companyEmployeesCount"),
        "job_poster_name": _normalise(item.get("jobPosterName")),
        "job_poster_title": _normalise(item.get("jobPosterTitle")),
        "job_poster_profile_url": _normalise(item.get("jobPosterProfileUrl")),
    }


def map_item(item: dict) -> dict[str, object | None]:
    """Public helper for callers that want the mapped output."""
    return _map_item(item)


def scrape_linkedin(
    role: str | None,
    location: str | None = None,
    *,
    job: str | None = None,
    years_of_experience: int | str | None = None,
    limit: int = DEFAULT_LIMIT,
    num_records: int | None = None,
    return_raw: bool = False,
) -> list[dict[str, object | None]]:
    """Scrape LinkedIn jobs via Apify actor.

    Args:
        role: Primary role or job-function keyword.
        job: Optional narrower job title or alternate title keyword.
        location: Optional location string.
        years_of_experience: Optional years-of-experience filter used in search keywords.
        limit: Preferred max number of jobs to return.
        num_records: Optional explicit records count; takes precedence over `limit`.
        return_raw: If True, keep actor items unchanged; defaults to mapped fields.

    Returns:
        List of job dicts. By default, maps to:
        title, company, location, url, posted_date, seniority, description.

    Raises:
        EnvironmentError: If APIFY_API_TOKEN is not set.
        ValueError: If no role/job keywords are provided, or invalid experience value.
        RuntimeError: If the Apify actor run fails.
    """
    if ApifyClient is None:
        raise RuntimeError(
            "Apify SDK import failed. Fix dependency versions (apify_client/apify_shared) "
            "before using LinkedIn scraper endpoints."
        ) from _APIFY_IMPORT_ERROR

    api_token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise EnvironmentError(
            "APIFY_API_TOKEN is not set. Add it to your .env file."
        )

    role = (role or "").strip()
    job = (job or "").strip() if job else None
    if not role and not job:
        raise ValueError("At least one of role or job is required")

    years = _normalise_years(years_of_experience)
    if years_of_experience is not None and years is None:
        raise ValueError("years_of_experience must be a number or numeric string")

    requested_count = num_records if num_records is not None else limit
    requested_count = int(requested_count)
    if requested_count < 10:
        raise ValueError("number of records must be at least 10")

    requested_count = min(requested_count, MAX_LIMIT)

    keywords = _build_keywords(role=role, job=job, years_of_experience=years)
    if not keywords:
        raise ValueError("Could not build search keywords from provided role/job inputs")

    client = ApifyClient(api_token)

    run_input = {
        "urls": [_build_search_url(keywords, location)],
        "scrapeCompany": True,
        "count": requested_count,
        "splitByLocation": False,
    }

    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)
    except Exception as exc:
        raise RuntimeError(f"Apify actor run failed: {exc}") from exc

    if not run:
        raise RuntimeError("Apify actor run did not return a result.")

    if run.get("status") == "FAILED":
        raise RuntimeError(f"Apify actor run failed: {run.get('statusMessage', 'unknown reason')}")

    if not run.get("defaultDatasetId"):
        raise RuntimeError("Apify actor run did not return a dataset.")

    results: list[dict[str, object | None]] = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        if return_raw:
            results.append(item)
            if len(results) >= requested_count:
                break
            continue

        mapped = _map_item(item)
        if mapped.get("url"):  # skip items without a valid URL
            results.append(mapped)
        if len(results) >= requested_count:
            break

    return results
