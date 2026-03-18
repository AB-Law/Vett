import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import Base
from app.config import get_settings
from app.services.interview_research.models import InterviewResearchQuestion, InterviewResearchQuestionBank
from app.services.interview_research.orchestrator import (
    _apply_minimums,
    _extract_candidate_profile_context,
    _fallback_questions_for_category,
    _deterministic_critic_decision,
    _parse_or_plan,
)
from app.services.interview_research.searxng_client import SearXNGClient
from app.services.interview_research.tools import search_web
from app.models.user_profile import UserProfile


@pytest.mark.asyncio
async def test_search_web_blocks_linkedin_jobs(monkeypatch):
    settings = get_settings()
    settings.interview_research_blocked_domains = "linkedin.com/jobs"
    settings.interview_research_allowed_domains = ""

    class FakeResult:
        def __init__(self, url: str, title: str, snippet: str):
            self.url = url
            self.title = title
            self.snippet = snippet

    async def fake_search(self, query: str, max_results: int):  # noqa: ARG001
        return [
            FakeResult("https://linkedin.com/jobs/123", "Job", "blocked"),
            FakeResult("https://engineering.example.com/posting", "Posting", "ok"),
        ]

    monkeypatch.setattr(SearXNGClient, "search", fake_search)
    results = await search_web("backend engineer interview questions")
    assert len(results) == 1
    assert "linkedin.com/jobs" not in results[0]["source_url"]
    assert results[0]["source_url"] == "https://engineering.example.com/posting"


def test_parse_or_plan_filters_only_supported_tools():
    tool_output = {
        "tool_calls": [
            {"tool": "search_web", "arguments": {"query": "x"}},
            {"tool": "extract_jd_facts", "arguments": {"job_description": "y"}},
            {"tool": "unsupported", "arguments": {}},
        ]
    }
    calls, used = _parse_or_plan(tool_output)
    assert len(calls) == 2
    assert calls[0]["tool"] == "search_web"
    assert calls[1]["tool"] == "extract_jd_facts"
    assert used == ["search_web", "extract_jd_facts"]


def test_apply_minimums_enforces_category_coverage_with_fallback():
    bank = InterviewResearchQuestionBank(
        behavioral=[
            InterviewResearchQuestion(
                question="Tell me about a team conflict.",
                tool="interview_documents",
                query="team",
                source_url="vector://one",
                source_title="Vector",
                source_type="vector",
                query_used="team",
                reason="seed",
                timestamp="2026-01-01T00:00:00Z",
                snippet="Team conflict response.",
                confidence_score=0.6,
            )
        ],
        technical=[],
        system_design=[],
        company_specific=[],
        source_urls=[],
    )
    fallback_used = _apply_minimums(bank, role="Backend Engineer", company="Acme")
    assert fallback_used is True
    assert len(bank.behavioral) >= 2
    assert len(bank.technical) >= 4
    assert len(bank.system_design) >= 2
    assert len(bank.company_specific) >= 3
    assert any(item.source_type == "fallback" for item in bank.all_questions())
    assert "local://fallback" in bank.source_urls


def test_apply_minimums_prefers_candidate_company_context():
    bank = InterviewResearchQuestionBank(
        behavioral=[],
        technical=[],
        system_design=[],
        company_specific=[],
        source_urls=[],
    )
    _apply_minimums(
        bank,
        role="Junior Backend Developer",
        company="Digitap.ai",
        candidate_company="Acme Fintech",
    )
    company_specific_text = " ".join(item.question_text for item in bank.company_specific).lower()
    assert "acme fintech" in company_specific_text
    assert "digitap.ai" in company_specific_text


def test_deterministic_critic_rejects_low_signal_sources():
    keep, reason = _deterministic_critic_decision(
        {
            "source_type": "search",
            "source_url": "https://letsintern.in/web-development-internship/",
            "source_title": "Web Development Internship",
            "snippet": "Apply now for internship opportunities.",
            "question_text": "Tell me about internship opportunities.",
        },
        role="Full Stack Intern",
        company="City Greens",
        job_text="Node.js Express.js React.js role responsibilities",
    )
    assert keep is False
    assert "low_signal_host" in reason or "job_listing_like_source" in reason

    keep2, reason2 = _deterministic_critic_decision(
        {
            "source_type": "search",
            "source_url": "https://citygreen.com/blog/engineering-stack",
            "source_title": "City Green engineering stack",
            "snippet": "Node.js and React systems used by the company.",
            "question_text": "How do you design Node.js APIs for this team?",
        },
        role="Full Stack Intern",
        company="City Greens",
        job_text="Node.js Express.js React.js role responsibilities",
    )
    assert keep2 is True
    assert reason2 == "context_aligned"


def _new_db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_extract_candidate_profile_context_prefers_stored_profile():
    db = _new_db_session()
    db.add(
        UserProfile(
            full_name="Ada Lovelace",
            headline_or_target_role="Research Engineer",
            current_company="NexGen Labs",
            years_experience=7,
            top_skills=["Python"],
            location="London",
            linkedin_url="https://example.com/ada",
            summary="Deep systems background.",
            source="manual",
        )
    )
    db.commit()
    company, summary = _extract_candidate_profile_context(db, "Experience at Old Corp")
    assert company == "NexGen Labs"
    assert summary == "Deep systems background."


def test_extract_candidate_profile_context_fallbacks_to_cv_text():
    db = _new_db_session()
    cv_text = "Software Developer at Fallbackly\nExperienced in Python."
    company, summary = _extract_candidate_profile_context(db, cv_text)
    assert company == "Fallbackly"
    assert "Software Developer at Fallbackly Experienced in Python." in summary


def test_company_specific_fallback_questions_are_answerable_without_inside_access():
    questions = _fallback_questions_for_category(
        category="company_specific",
        role="Backend Engineer",
        company="Digitap.ai",
        count=3,
        candidate_company="PwC India",
    )
    assert len(questions) == 3
    combined = " ".join(questions).lower()
    assert "public information" in combined
    assert "unknowns" in combined
    assert "evaluate engineering culture and execution differences" not in combined
