from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import get_settings
import os

settings = get_settings()

engine_kwargs: dict[str, object] = {}
database_url = settings.database_url

if database_url.startswith("sqlite:///"):
    # Ensure SQLite directory exists for local dev.
    db_path = database_url.replace("sqlite:///", "", 1)
    if db_path.startswith("/"):
        db_path = db_path.replace("////", "/", 1)
    if not db_path.startswith(":"):
        db_directory = os.path.dirname(db_path)
        if db_directory:
            try:
                os.makedirs(db_directory, exist_ok=True)
            except OSError as exc:
                if exc.errno not in {13, 30}:
                    raise
                # CI and container environments can block /data; fall back to a writable path.
                fallback_dir = "/tmp/vett"
                os.makedirs(fallback_dir, exist_ok=True)
                db_path = os.path.join(fallback_dir, os.path.basename(db_path) or "vett.db")
                database_url = f"sqlite:///{db_path}"
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(database_url, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
