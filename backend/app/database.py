from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import get_settings
import os

settings = get_settings()

engine_kwargs: dict[str, object] = {}

if settings.database_url.startswith("sqlite:///"):
    # Ensure SQLite directory exists for local dev.
    db_path = settings.database_url.replace("sqlite:///", "", 1)
    if db_path.startswith("/"):
        db_path = db_path.replace("////", "/", 1)
    if not db_path.startswith(":"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
