import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.user_profile import UserProfile
from app.routers.profile import UserProfileUpdate, get_profile, update_profile
from app.services.user_profile_extractor import upsert_user_profile_from_extraction


def _new_db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_get_profile_returns_default_when_none():
    db = _new_db_session()
    profile = get_profile(db)
    assert profile.id is None
    assert profile.full_name == ""
    assert profile.top_skills == []


def test_update_profile_normalizes_and_persists():
    db = _new_db_session()
    payload = UserProfileUpdate(
        full_name="  Ada Lovelace  ",
        headline_or_target_role="Software Engineer",
        current_company="Acme",
        years_experience=7,
        top_skills=["Python", "python", "Go"],
        location="London",
        linkedin_url="https://linkedin.com/in/ada",
        summary="  concise profile  ",
    )
    updated = update_profile(payload, db)
    assert updated.id is not None
    assert updated.full_name == "Ada Lovelace"
    assert updated.top_skills == ["Python", "Go"]
    assert updated.years_experience == 7
    assert updated.source == "manual"
    assert db.query(UserProfile).count() == 1


def test_upsert_user_profile_from_extraction_overwrites_existing():
    db = _new_db_session()
    upsert_user_profile_from_extraction(
        db,
        {
            "full_name": "First Candidate",
            "headline_or_target_role": "Engineer",
            "current_company": "OldCo",
            "years_experience": "3",
            "top_skills": ["Python", "SQL"],
            "location": "Paris",
            "linkedin_url": "",
            "summary": "First summary",
        },
        source="cv_llm",
    )

    upsert_user_profile_from_extraction(
        db,
        {
            "full_name": "Second Candidate",
            "headline_or_target_role": "Senior Engineer",
            "current_company": "NewCo",
            "years_experience": 6,
            "top_skills": ["Kubernetes", "Go"],
            "location": "Remote",
            "linkedin_url": "",
            "summary": "Second summary",
        },
        source="cv_llm",
    )

    persisted = db.query(UserProfile).order_by(UserProfile.id.asc()).first()
    assert persisted is not None
    assert persisted.full_name == "Second Candidate"
    assert persisted.current_company == "NewCo"
    assert persisted.years_experience == 6
    assert persisted.top_skills == ["Kubernetes", "Go"]
    assert persisted.source == "cv_llm"
