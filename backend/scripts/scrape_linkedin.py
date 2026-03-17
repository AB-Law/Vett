#!/usr/bin/env python3

"""CLI entrypoint for testing the LinkedIn scraper.

Usage:
  python scripts/scrape_linkedin.py --role "Machine Learning Engineer" --location "Remote" --num-records 5
  python scripts/scrape_linkedin.py --job "Data Scientist" --role "Machine Learning" --raw --output jobs.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a script from repo root with:
#   python backend/scripts/scrape_linkedin.py ...
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.scrapers.linkedin import scrape_linkedin  # noqa: E402
from app.database import SessionLocal
from app.models.score import ScrapeRequest
from app.services.scrape_storage import store_scrape_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape LinkedIn jobs via Apify and write raw or mapped output."
    )
    parser.add_argument(
        "--role",
        default=None,
        help="Primary role/keyword, e.g. 'data scientist'",
    )
    parser.add_argument(
        "--job",
        default=None,
        help="Optional secondary job title/keyword, e.g. 'backend engineer'",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional location, e.g. 'Remote'",
    )
    parser.add_argument(
        "--years-of-experience",
        default=None,
        help="Optional years of experience, e.g. 3",
    )
    parser.add_argument(
        "--num-records",
        "--limit",
        dest="num_records",
        type=int,
        default=25,
        help="Max jobs to fetch (default: 25)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Return raw Apify items instead of mapped job records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON payload",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip persisting to Postgres (write output file only).",
    )
    return parser.parse_args()


def _default_output_path(args: argparse.Namespace) -> Path:
    output_dir = os.getenv("SCRAPE_OUTPUT_DIR")
    if not output_dir:
        output_dir = os.getenv("UPLOAD_DIR")
    if not output_dir:
        output_dir = str(Path(__file__).resolve().parent.parent / "data" / "scrapes")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    base = (args.role or args.job or "linkedin_jobs").strip().lower()
    safe_base = re.sub(r"[^a-z0-9_-]+", "_", base)[:40].strip("_") or "linkedin_jobs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir_path / f"{safe_base}_{timestamp}.json"


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not args.role and not args.job:
        raise SystemExit("Either --role or --job is required")

    results = scrape_linkedin(
        role=args.role,
        location=args.location,
        job=args.job,
        years_of_experience=args.years_of_experience,
        num_records=args.num_records,
        return_raw=args.raw,
    )

    payload = {
        "query": {
            "role": args.role,
            "job": args.job,
            "location": args.location,
            "years_of_experience": args.years_of_experience,
            "num_records": args.num_records,
            "raw": args.raw,
        },
        "count": len(results),
        "results": results,
    }

    if not args.skip_db:
        stored = _persist_to_postgres(args, results)
        payload["stored"] = {
            "method": "postgres",
            "request_id": stored["request_id"],
            "stored_count": stored["stored_count"],
        }
    else:
        payload["stored"] = {"method": "file_only"}

    output_path = args.output if args.output is not None else _default_output_path(args)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["stored"]["file_path"] = str(output_path)

    return payload


def _persist_to_postgres(
    args: argparse.Namespace,
    results: list[dict[str, object]],
) -> dict[str, object]:
    db = SessionLocal()
    try:
        scrape = ScrapeRequest(
            source="linkedin",
            role=(args.role or "").strip(),
            job=args.job,
            location=args.location,
            years_of_experience=_to_int(args.years_of_experience),
            num_records=args.num_records,
            requested_by="cli",
            return_raw=args.raw,
            result_count=len(results),
        )
        db.add(scrape)
        db.flush()

        stored_count = store_scrape_results(
            db=db,
            scrape_request_id=scrape.id,
            source="linkedin",
            items=results,
            store_mapped=True,
        )
        db.commit()
        return {"request_id": scrape.id, "stored_count": stored_count}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> int:
    args = _parse_args()
    payload = _run(args)
    print(json.dumps(payload, indent=2))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
