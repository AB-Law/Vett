from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
import logging
import time

from ..database import get_db
from ..models.cv import CV
from ..services.cv_parser import parse_cv
from ..services.user_profile_extractor import (
    extract_profile_from_cv_text,
    upsert_user_profile_from_extraction,
)

router = APIRouter(prefix="/cv", tags=["cv"])
logger = logging.getLogger(__name__)


class CVResponse(BaseModel):
    id: int
    filename: str
    file_size: int
    file_type: str
    parsed_text: str
    created_at: str

    model_config = ConfigDict(from_attributes=True)


@router.post("/upload", response_model=CVResponse)
async def upload_cv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start_ts = time.perf_counter()
    allowed = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}
    filename = file.filename or "uploaded-cv.txt"
    suffix = "." + filename.split(".")[-1].lower()
    if suffix not in allowed:
        logger.warning("Rejected CV upload due to unsupported extension: %s", suffix)
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")

    data = await file.read()
    logger.info(
        "Starting CV upload parse filename=%s size=%s content_type=%s",
        filename,
        len(data),
        file.content_type,
    )
    try:
        parsed_start = time.perf_counter()
        parsed_text = parse_cv(data, filename)
        logger.info(
            "CV parse complete filename=%s parsed_chars=%s duration_ms=%.1f",
            filename,
            len(parsed_text),
            (time.perf_counter() - parsed_start) * 1000,
        )
    except ValueError as e:
        logger.warning("CV upload failed validation filename=%s error=%s", filename, str(e))
        raise HTTPException(422, str(e))
    except Exception:
        logger.exception("CV upload parse failed unexpectedly filename=%s", filename)
        raise HTTPException(500, "Upload failed while parsing file.")

    logger.info("Replacing previous CV records before saving new CV: filename=%s", filename)
    # Replace any existing CV (single-CV model)
    db.query(CV).delete()

    cv = CV(
        filename=filename,
        file_size=len(data),
        file_type=suffix.lstrip("."),
        parsed_text=parsed_text,
    )
    db.add(cv)
    db.commit()
    db.refresh(cv)
    try:
        extracted_profile = await extract_profile_from_cv_text(parsed_text)
        upsert_user_profile_from_extraction(
            db,
            payload=extracted_profile,
            source="cv_llm",
        )
    except Exception:
        logger.exception("Failed to extract and persist user profile from CV")

    logger.info(
        "CV upload completed id=%s filename=%s parse_chars=%s duration_ms=%.1f",
        cv.id,
        filename,
        len(parsed_text),
        (time.perf_counter() - start_ts) * 1000,
    )

    return CVResponse(
        id=cv.id,
        filename=cv.filename,
        file_size=cv.file_size,
        file_type=cv.file_type,
        parsed_text=cv.parsed_text,
        created_at=str(cv.created_at),
    )


@router.get("/", response_model=CVResponse | None)
def get_cv(db: Session = Depends(get_db)):
    cv = db.query(CV).order_by(CV.id.desc()).first()
    if not cv:
        return None
    return CVResponse(
        id=cv.id,
        filename=cv.filename,
        file_size=cv.file_size,
        file_type=cv.file_type,
        parsed_text=cv.parsed_text,
        created_at=str(cv.created_at),
    )


@router.delete("/")
def delete_cv(db: Session = Depends(get_db)):
    deleted = db.query(CV).delete()
    db.commit()
    return {"deleted": deleted}
