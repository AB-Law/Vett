"""Sync + retrieval helpers for the local practice question repository."""

from __future__ import annotations

from collections.abc import Iterable
import csv
import difflib
import hashlib
import logging
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models.practice import PracticeQuestion, QuestionCompany

logger = logging.getLogger(__name__)

QUESTION_FILE_PRIORITY = [
    "thirty-days.csv",
    "three-months.csv",
    "six-months.csv",
    "one-year.csv",
    "all.csv",
]
WINDOW_GRANULARITY = {f.removesuffix(".csv"): i for i, f in enumerate(QUESTION_FILE_PRIORITY)}
# {"thirty-days": 0, "three-months": 1, "six-months": 2, "one-year": 3, "all": 4}
NON_ALPHA_NUMERIC = re.compile(r"[^a-z0-9]+")
NON_ALNUM_FOR_HEADERS = re.compile(r"[^a-z0-9]+")

PRACTICE_COMPANY_ALIAS = {
    "googleinc": "google",
    "google-llc": "google",
    "google llc": "google",
    "goldman-sachs": "goldman-sachs",
    "tata-consultancy-services": "tata-consultancy-services",
    "tata consultancy services": "tata-consultancy-services",
    "microsoftindia": "microsoft",
    "microsoft india": "microsoft",
    "globallogic": "globallogic",
    "global-logic": "globallogic",
    "global logic": "globallogic",
    "globallogic llc": "globallogic",
}


def _parse_frequency(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().rstrip("%")
    try:
        f = float(cleaned)
        return f / 100.0 if f > 1.0 else f
    except ValueError:
        return None


class PracticeSyncResult:
    def __init__(self, company: str | None = None) -> None:
        self.company = company
        self.inserted = 0
        self.updated = 0
        self.retired = 0


def normalize_company_slug(name: str) -> str:
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(
        r"\b(inc|incorporated|llc|ltd|corp|corporation|plc|pvt|private|co|ltd\.|co\.|llc\.)\b",
        "",
        cleaned,
    )
    cleaned = NON_ALPHA_NUMERIC.sub("-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-")


def _normalize_csv_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value).strip() or None


def _normalize_csv_field_name(field_name: Any) -> str:
    if not isinstance(field_name, str):
        return ""
    return NON_ALNUM_FOR_HEADERS.sub("", field_name.lower().strip())


def _row_signature(*parts: Any) -> str:
    payload = "|".join(_normalize_csv_value(p) or "" for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_git_command(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "Failed to run git command"
        raise RuntimeError(message)
    return (completed.stdout or "").strip()


def _prepare_repo_url(base_url: str, token: str) -> str:
    if not token or "@" in base_url or not base_url.startswith("https://"):
        return base_url
    return f"https://{token}@{base_url.removeprefix('https://')}"


def sync_repo() -> tuple[Path, str]:
    settings = get_settings()
    repo_path = Path(settings.practice_repo_path).expanduser().resolve()
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_url = _prepare_repo_url(settings.practice_repo_url, settings.practice_github_token)
    branch = settings.practice_repo_branch

    if not repo_path.exists():
        _run_git_command(["git", "clone", "--depth", "1", "-b", branch, repo_url, str(repo_path)])
    elif not (repo_path / ".git").exists():
        raise RuntimeError(f"Invalid repository path: {repo_path}")
    else:
        _run_git_command(["git", "-C", str(repo_path), "fetch", "origin", branch])
        _run_git_command(["git", "-C", str(repo_path), "checkout", branch])
        _run_git_command(["git", "-C", str(repo_path), "pull", "--ff-only", "origin", branch])

    commit = _run_git_command(["git", "-C", str(repo_path), "rev-parse", "--short", f"origin/{branch}"])
    if not commit:
        commit = _run_git_command(["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"])
    return repo_path, commit


def discover_company_dir(repo_path: Path, slug: str | None, db: Session | None = None) -> str | None:
    if not slug:
        return None

    normalized_slug = normalize_company_slug(slug)
    normalized_slug = PRACTICE_COMPANY_ALIAS.get(normalized_slug, normalized_slug)
    compact_slug = NON_ALPHA_NUMERIC.sub("", normalized_slug)

    directories = [p.name for p in repo_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    mapped = {normalize_company_slug(name): name for name in directories}
    mapped_compact = {NON_ALPHA_NUMERIC.sub("", key): value for key, value in mapped.items()}

    if normalized_slug in mapped:
        return mapped[normalized_slug]
    if compact_slug in mapped_compact:
        return mapped_compact[compact_slug]
    for alias in set(PRACTICE_COMPANY_ALIAS.values()):
        alias_slug = normalize_company_slug(alias)
        if alias_slug in mapped:
            return mapped[alias_slug]
        if NON_ALPHA_NUMERIC.sub("", alias_slug) in mapped_compact:
            return mapped_compact[NON_ALPHA_NUMERIC.sub("", alias_slug)]

    # Try compact folder keys for raw normalized input (e.g., global logic -> globallogic).
    for key, value in mapped_compact.items():
        if key == compact_slug:
            return value

    close = difflib.get_close_matches(normalized_slug, list(mapped.keys()), n=1, cutoff=0.8)
    if close:
        return mapped[close[0]]
    close_compact = difflib.get_close_matches(compact_slug, list(mapped_compact.keys()), n=1, cutoff=0.8)
    if close_compact:
        return mapped_compact[close_compact[0]]

    if db is None:
        return None
    known = [entry[0] for entry in db.query(QuestionCompany.company_slug).distinct().all() if entry[0]]
    close = difflib.get_close_matches(normalized_slug, known, n=1, cutoff=0.8)
    return close[0] if close else None


def resolve_company_slug(slug: str | None, db: Session | None = None) -> str:
    """
    Resolve a company slug into a canonical slug used by the DB.

    This applies alias normalization, then tries exact/direct DB matches, then
    compact match and fuzzy fallback.
    """
    normalized = normalize_company_slug(slug or "")
    if not normalized:
        return normalized

    canonical = PRACTICE_COMPANY_ALIAS.get(normalized, normalized)

    if db is None:
        return canonical

    known = [entry[0] for entry in db.query(QuestionCompany.company_slug).distinct().all() if entry[0]]
    if not known:
        return canonical

    if canonical in known:
        return canonical

    # Match compact variants, e.g. "global-logic" -> "globallogic".
    compact_canonical = NON_ALPHA_NUMERIC.sub("", canonical)
    compact_known = {
        NON_ALPHA_NUMERIC.sub("", entry): entry
        for entry in known
        if NON_ALPHA_NUMERIC.sub("", entry)
    }
    if compact_canonical in compact_known:
        return compact_known[compact_canonical]

    close = difflib.get_close_matches(canonical, known, n=1, cutoff=0.8)
    if close:
        return close[0]

    close_compact = difflib.get_close_matches(compact_canonical, list(compact_known.keys()), n=1, cutoff=0.8)
    if close_compact:
        return compact_known[close_compact[0]]

    return canonical


def available_question_file(company_dir: Path, preferred_window: str | None = None) -> tuple[Path | None, str | None]:
    requested = (preferred_window or "").lower().replace("_", "-")
    priority = QUESTION_FILE_PRIORITY.copy()
    if requested:
        for index, item in enumerate(priority):
            if requested in item:
                priority.pop(index)
                priority.insert(0, item)
                break

    for file_name in priority:
        path = company_dir / file_name
        if path.exists():
            return path, file_name.removesuffix(".csv")
    return None, None


def _parse_question_rows(file_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            return rows

        lower_fields = [
            _normalize_csv_field_name(name)
            for name in reader.fieldnames
            if _normalize_csv_field_name(name) != ""
        ]
        has_expected_header = "title" in lower_fields and "difficulty" in lower_fields
        fp.seek(0)
        if has_expected_header:
            dict_reader = csv.DictReader(fp)
            for row in dict_reader:
                normalized_row = {
                    _normalize_csv_field_name(key): _normalize_csv_value(value)
                    for key, value in row.items()
                }
                out = {
                    "id": _normalize_csv_value(
                        normalized_row.get("id")
                        or normalized_row.get("questionid")
                        or normalized_row.get("question_id")
                    ),
                    "url": _normalize_csv_value(
                        normalized_row.get("url")
                        or normalized_row.get("link")
                        or normalized_row.get("questionlink")
                    ),
                    "title": _normalize_csv_value(
                        normalized_row.get("title")
                        or normalized_row.get("questiontitle")
                    ),
                    "difficulty": _normalize_csv_value(normalized_row.get("difficulty")),
                    "acceptance": _normalize_csv_value(
                        normalized_row.get("acceptance")
                        or normalized_row.get("acceptancepercent")
                        or normalized_row.get("acceptancerate")
                    ),
                    "frequency": _normalize_csv_value(
                        normalized_row.get("frequency")
                        or normalized_row.get("frequencypercent")
                        or normalized_row.get("askedfrequency")
                    ),
                }
                if out["title"]:
                    rows.append(out)
        else:
            fallback_reader = csv.reader(fp)
            for row in fallback_reader:
                if not row or len(row) < 3:
                    continue
                out = {
                    "id": _normalize_csv_value(row[0]),
                    "url": _normalize_csv_value(row[1]),
                    "title": _normalize_csv_value(row[2]),
                    "difficulty": _normalize_csv_value(row[3]) if len(row) > 3 else None,
                    "acceptance": _normalize_csv_value(row[4]) if len(row) > 4 else None,
                    "frequency": _normalize_csv_value(row[5]) if len(row) > 5 else None,
                }
                if out["title"]:
                    rows.append(out)
    return rows


def _upsert_company_window_rows(
    db: Session,
    company_slug: str,
    source_window: str,
    source_commit: str,
    rows: Iterable[dict[str, str]],
    active_question_ids: set[int],
) -> PracticeSyncResult:
    result = PracticeSyncResult(company_slug)
    new_window_rank = WINDOW_GRANULARITY.get(source_window, 999)

    for row in rows:
        title = row.get("title") or ""
        if not title:
            continue

        # Find or create the PracticeQuestion by title (unique key).
        question = db.query(PracticeQuestion).filter(PracticeQuestion.title == title).first()
        if question is None:
            question = PracticeQuestion(
                title=title,
                url=row.get("url"),
                difficulty=row.get("difficulty"),
                acceptance=row.get("acceptance"),
                is_active=True,
                source_commit=source_commit,
            )
            db.add(question)
            db.flush()
            result.inserted += 1
        else:
            result.updated += 1

        # Find or create the QuestionCompany junction row.
        qc = (
            db.query(QuestionCompany)
            .filter(
                QuestionCompany.question_id == question.id,
                QuestionCompany.company_slug == company_slug,
            )
            .first()
        )
        parsed_freq = _parse_frequency(row.get("frequency"))
        if qc is None:
            qc = QuestionCompany(
                question_id=question.id,
                company_slug=company_slug,
                frequency=parsed_freq,
                source_window=source_window,
                is_active=True,
            )
            db.add(qc)
            db.flush()
        else:
            existing_rank = WINDOW_GRANULARITY.get(qc.source_window, 999)
            if new_window_rank <= existing_rank:
                qc.source_window = source_window
                qc.frequency = parsed_freq
            qc.is_active = True

        active_question_ids.add(question.id)

    return result


def sync_all(
    db: Session,
    company: str | None = None,
    preferred_window: str | None = None,
) -> dict[str, int | str | list[dict[str, int | str]]]:
    repo_path, commit = sync_repo()
    summary: dict[str, int | str | list[dict[str, int | str]]] = {
        "commit": commit,
        "inserted": 0,
        "updated": 0,
        "retired": 0,
        "companies": [],
    }

    if company:
        folder = discover_company_dir(repo_path, company, db)
        if folder is None:
            raise RuntimeError(f"No company folder mapped for '{company}'.")
        company_dirs = [repo_path / folder]
    else:
        company_dirs = [path for path in repo_path.iterdir() if path.is_dir() and not path.name.startswith(".")]

    company_stats: list[dict[str, int | str]] = []
    for company_dir in company_dirs:
        if not company_dir.is_dir():
            continue

        company_slug = normalize_company_slug(company_dir.name)
        total_result = PracticeSyncResult(company_slug)
        active_question_ids: set[int] = set()

        windows_synced = 0
        for window_file in QUESTION_FILE_PRIORITY:
            csv_path = company_dir / window_file
            if not csv_path.exists():
                continue
            source_window = window_file.removesuffix(".csv")

            # If caller requested a specific window, skip others.
            if preferred_window and preferred_window.replace("-", "_") not in window_file.replace("-", "_"):
                continue

            rows = _parse_question_rows(csv_path)
            if not rows:
                continue

            result = _upsert_company_window_rows(
                db=db,
                company_slug=company_slug,
                source_window=source_window,
                source_commit=commit,
                rows=rows,
                active_question_ids=active_question_ids,
            )
            total_result.inserted += result.inserted
            total_result.updated += result.updated
            total_result.retired += result.retired
            windows_synced += 1

        if windows_synced == 0:
            logger.info("No interview CSV files found for %s", company_dir.name)
            continue

        # Retire QuestionCompany rows for this company that were not seen in this sync.
        stale_qcs = (
            db.query(QuestionCompany)
            .filter(
                QuestionCompany.company_slug == company_slug,
                QuestionCompany.is_active.is_(True),
            )
            .all()
        )
        for qc in stale_qcs:
            if qc.question_id not in active_question_ids:
                qc.is_active = False
                total_result.retired += 1

        db.commit()

        summary["inserted"] = int(summary["inserted"]) + total_result.inserted
        summary["updated"] = int(summary["updated"]) + total_result.updated
        summary["retired"] = int(summary["retired"]) + total_result.retired
        company_stats.append(
            {
                "company_slug": company_slug,
                "inserted": total_result.inserted,
                "updated": total_result.updated,
                "retired": total_result.retired,
            },
        )

    summary["companies"] = company_stats
    return summary


def get_questions_for_company(
    db: Session,
    company_slug: str,
    job_session_id: int | None = None,
    limit: int = 8,
    difficulty: str | None = None,
    source_window: str | None = None,
    recent_window_minutes: int = 0,
    exclude_statuses: list[str] | None = None,
) -> list[PracticeQuestion]:
    from ..models.practice import PracticeSessionQuestion

    canonical_slug = normalize_company_slug(company_slug)

    query = (
        db.query(PracticeQuestion, QuestionCompany)
        .join(QuestionCompany, QuestionCompany.question_id == PracticeQuestion.id)
        .filter(
            QuestionCompany.company_slug == canonical_slug,
            QuestionCompany.is_active.is_(True),
            PracticeQuestion.is_active.is_(True),
        )
    )

    if difficulty:
        query = query.filter(PracticeQuestion.difficulty.ilike(f"%{difficulty}%"))
    if source_window:
        query = query.filter(QuestionCompany.source_window == source_window.replace(".csv", ""))

    if job_session_id is not None:
        status_filter = exclude_statuses or ["solved", "discarded", "ai-generated", "seen"]
        asked_subq = (
            db.query(PracticeSessionQuestion.question_id)
            .filter(PracticeSessionQuestion.practice_session_id == job_session_id)
        )
        if status_filter:
            asked_subq = asked_subq.filter(PracticeSessionQuestion.status.in_(status_filter))
        if recent_window_minutes > 0:
            cutoff = datetime.utcnow() - timedelta(minutes=recent_window_minutes)
            asked_subq = asked_subq.filter(PracticeSessionQuestion.asked_at >= cutoff)
        query = query.filter(~PracticeQuestion.id.in_(asked_subq))

    final_limit = max(1, int(limit))
    rows = (
        query
        .order_by(QuestionCompany.frequency.desc().nullsfirst())
        .limit(final_limit)
        .all()
    )

    # Attach company-specific metadata as Python instance attributes so callers
    # can access q.frequency and q.source_window without changing the interface.
    result: list[PracticeQuestion] = []
    for q, qc in rows:
        q.frequency = str(qc.frequency) if qc.frequency is not None else None
        q.source_window = qc.source_window
        result.append(q)
    return result
