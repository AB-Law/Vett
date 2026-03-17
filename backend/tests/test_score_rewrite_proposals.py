import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.models.cv import CV
from app.routers import score as score_router


def _new_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_rewrite_proposals_returns_model_fallback_on_generation_error(monkeypatch):
    db = _new_session()
    db.add(CV(filename="candidate.pdf", parsed_text="Python engineering experience."))
    db.commit()

    async def _extract_role_signal_map(cv_text: str, jd_text: str):
        return {"role_summary": "backend engineer"}

    async def _score_cv_against_jd(cv_text: str, jd_text: str, role_signal_map=None):
        return {
            "fit_score": 72,
            "matched_keywords": ["python", "backend"],
            "missing_keywords": [],
            "gap_analysis": "Some gaps remain.",
            "rewrite_suggestions": ["Add metrics."],
        }

    async def _generate_cv_rewrite_proposals(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(score_router, "extract_role_signal_map", _extract_role_signal_map)
    monkeypatch.setattr(score_router, "score_cv_against_jd", _score_cv_against_jd)
    monkeypatch.setattr(score_router, "generate_cv_rewrite_proposals", _generate_cv_rewrite_proposals)

    result = await score_router.rewrite_proposals(
        req=score_router.CVRewriteRequest(
            job_description="Need a senior backend role",
            job_title="Backend Lead",
            company="Acme",
        ),
        db=db,
    )

    assert len(result.proposals) == 1
    assert (
        result.proposals[0].before
        == "Could not generate a structured rewrite proposal."
    )
    assert result.job_title == "Backend Lead"
    assert result.company == "Acme"
