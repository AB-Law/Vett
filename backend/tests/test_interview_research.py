import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.interview_research.models import InterviewResearchQuestion, InterviewResearchQuestionBank
from app.services.interview_research.orchestrator import _detect_distributed_topics, _fallback_questions_for_category, _normalize_category_output
from app.services.interview_research.tools import build_company_culture_query, build_interview_questions_query, build_role_skills_query


def test_query_builders_include_expected_sources():
    assert "site:glassdoor.com" in build_interview_questions_query("Senior Backend Engineer", "Acme")
    assert "reddit.com" in build_interview_questions_query("Backend Engineer", None)
    assert "site:linkedin.com/jobs" in build_role_skills_query("Backend Engineer", "Acme")
    assert "site:medium.com" in build_company_culture_query("Acme", "Backend")


def test_fallback_questions_and_source_tracking():
    bank = InterviewResearchQuestionBank()
    _normalize_category_output(bank, "Backend Engineer", "Acme")
    assert len(bank.behavioral) >= 2
    assert len(bank.technical) >= 4
    assert len(bank.system_design) >= 2
    assert len(bank.company_specific) >= 3
    assert "local://fallback" in bank.source_urls


def test_kafka_driven_distributed_followups():
    bank = InterviewResearchQuestionBank(
        behavioral=[
            InterviewResearchQuestion(
                question="Can you describe how your team recovered a Kafka incident?",
                tool="unit",
                query="unit",
                source_url="https://example.com",
                source_title="Example",
                timestamp="2026-01-01T00:00:00Z",
                snippet="incident with Kafka",
                confidence_score=0.7,
            )
        ],
        technical=[],
        system_design=[],
        company_specific=[],
        source_urls=[],
    )
    queries = _detect_distributed_topics(bank, "Backend Engineer", "Acme")
    assert len(queries) >= 2
    assert any("Kafka" in q or "distributed systems" in q for q in queries)


def test_fallback_question_templates_by_category():
    technical = _fallback_questions_for_category("technical", "Backend Engineer", "Acme", count=2)
    company_specific = _fallback_questions_for_category("company_specific", "Backend Engineer", "Acme", count=2)
    assert len(technical) == 2
    assert len(company_specific) == 2
