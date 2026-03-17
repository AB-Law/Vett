from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict

from ..database import get_db
from ..models.cv import CV
from ..services.cv_parser import parse_cv

router = APIRouter(prefix="/cv", tags=["cv"])


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
    allowed = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}
    suffix = "." + file.filename.split(".")[-1].lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")

    data = await file.read()
    try:
        parsed_text = parse_cv(data, file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))

    # Replace any existing CV (single-CV model)
    db.query(CV).delete()

    cv = CV(
        filename=file.filename,
        file_size=len(data),
        file_type=suffix.lstrip("."),
        parsed_text=parsed_text,
    )
    db.add(cv)
    db.commit()
    db.refresh(cv)

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
