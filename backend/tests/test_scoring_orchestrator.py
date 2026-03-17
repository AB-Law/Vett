import asyncio
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models.cv import CV
from app.models.score import (
    AGENT_STATE_COMPLETED,
    AGENT_STATE_EVIDENCE_SCAN,
    AGENT_STATE_FAILED,
    AGENT_STATE_GAP_AUDIT,
    AGENT_STATE_ACTION_PLAN,
    AGENT_STATE_REWRITE_PLAN,
    AGENT_STATE_ROLE_ANALYSIS,
    AGENT_STATE_SCORING,
    AGENT_STATE_COMPLETED as AGENT_STATE_COMPLETE,
    AgentRun,
    Job,
    RescoreRun,
)
from app.routers import jobs as jobs_router
from app.services import scoring_orchestrator as orchestrator


def _make_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _seed_cv(db):
    db.add(
        CV(
            id=1,
            filename="test.pdf",
            file_size=1024,
            file_type="application/pdf",
            parsed_text="Experienced backend engineer.",
        )
    )
    db.commit()


def test_orchestrator_records_full_step_transitions(monkeypatch):
    Session = _make_session_factory()
    db = Session()
    _seed_cv(db)

    async def fake_role_analysis(cv_text, jd_text):
        return {"role_summary": "backend"}, {"evidence_snippets": ["python", "apis"]}

    async def fake_evidence_scan(cv_text, jd_text, role_signal_map):
        return {
            "top_responsibilities": ["Build APIs"],
            "top_skills": ["Python"],
            "experience_signals": ["5+ years"],
        }, {}

    async def fake_scoring(cv_text, jd_text, role_signal_map):
        return {
            "fit_score": 87,
            "matched_keywords": ["python", "api"],
            "missing_keywords": ["k8s"],
            "gap_analysis": "Needs stronger systems scale signals.",
            "rewrite_suggestions": ["Add ownership example"],
        }, {"fit_score": 87}

    async def fake_gap_audit(role_signal_map, score_payload):
        return {
            "gaps_to_address": ["k8s"],
            "role_signal_focus": ["senior"],
        }, {}

    async def fake_action_plan(cv_text, jd_text, score_payload, role_signal_map):
        return {"skills_to_fix_first": ["k8s"]}, {"plan_fields": ["skills"]}

    def fake_rewrite_plan(score_payload):
        return {"rewrite_suggestions": ["Quantify delivery metrics"]}, {}

    monkeypatch.setattr(orchestrator, "_run_role_analysis_step", fake_role_analysis)
    monkeypatch.setattr(orchestrator, "_run_evidence_scan_step", fake_evidence_scan)
    monkeypatch.setattr(orchestrator, "_run_scoring_step", fake_scoring)
    monkeypatch.setattr(orchestrator, "_run_gap_audit_step", fake_gap_audit)
    monkeypatch.setattr(orchestrator, "_run_action_plan_step", fake_action_plan)
    monkeypatch.setattr(orchestrator, "_run_rewrite_plan_step", fake_rewrite_plan)

    result = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="fixed-orchestrator-key",
        )
    )

    triggers = [item.trigger for item in result.transitions]
    assert triggers == [
        AGENT_STATE_ROLE_ANALYSIS,
        AGENT_STATE_EVIDENCE_SCAN,
        AGENT_STATE_SCORING,
        AGENT_STATE_GAP_AUDIT,
        AGENT_STATE_ACTION_PLAN,
        AGENT_STATE_REWRITE_PLAN,
        "finalize",
    ]
    assert [item.step for item in result.artifacts] == [
        AGENT_STATE_ROLE_ANALYSIS,
        AGENT_STATE_EVIDENCE_SCAN,
        AGENT_STATE_SCORING,
        AGENT_STATE_GAP_AUDIT,
        AGENT_STATE_ACTION_PLAN,
        AGENT_STATE_REWRITE_PLAN,
    ]
    assert result.current_state == AGENT_STATE_COMPLETED
    assert result.status == AGENT_STATE_COMPLETED
    assert result.result["fit_score"] == 87
    assert result.result["run_attempt_count"] == 1

    run, _, artifacts = orchestrator.get_run_timeline(db, result.run_id)
    assert isinstance(run, AgentRun)
    assert run.attempt_count == 1
    assert len(artifacts) == 6


def test_orchestrator_preserves_scoring_evidence_rows(monkeypatch):
    Session = _make_session_factory()
    db = Session()
    _seed_cv(db)

    async def fake_role_analysis(cv_text, jd_text):
        return {"role_summary": "backend"}, {"evidence_snippets": ["python", "apis"]}

    async def fake_evidence_scan(cv_text, jd_text, role_signal_map):
        return {
            "top_responsibilities": ["Build APIs"],
            "top_skills": ["Python"],
            "experience_signals": ["5+ years"],
        }, {}

    async def fake_scoring(cv_text, jd_text, role_signal_map):
        return {
            "fit_score": 87,
            "matched_keywords": ["python", "api"],
            "missing_keywords": ["k8s"],
            "gap_analysis": "Needs stronger systems scale signals.",
            "rewrite_suggestions": ["Add ownership example"],
            "matched_keyword_evidence": [
                {
                    "value": "python",
                    "cv_citations": [
                        {
                            "section_id": "experience",
                            "line_start": 1,
                            "line_end": 1,
                            "snippet": "Experienced backend engineer.",
                        }
                    ],
                    "jd_phrase_citations": [
                        {
                            "phrase_id": "phrase-1",
                            "line_start": 1,
                            "line_end": 2,
                            "snippet": "Build APIs",
                        }
                    ],
                },
                {
                    "value": "api",
                    "cv_citations": [],
                    "jd_phrase_citations": [],
                    "evidence_missing_reason": "No matching API phrase was found in the JD source parse.",
                },
            ],
            "missing_keyword_evidence": [
                {
                    "value": "k8s",
                    "cv_citations": [],
                    "jd_phrase_citations": [
                        {"phrase_id": "phrase-2", "line_start": 7, "line_end": 7, "snippet": "Kubernetes ownership"},
                    ],
                }
            ],
            "rewrite_suggestion_evidence": [
                {
                    "value": "Add ownership example",
                    "cv_citations": [],
                    "jd_phrase_citations": [],
                    "evidence_missing_reason": "No rewrite-specific citation map returned.",
                }
            ],
        }, {"fit_score": 87}

    async def fake_gap_audit(role_signal_map, score_payload):
        return {
            "gaps_to_address": ["k8s"],
            "role_signal_focus": ["senior"],
        }, {}

    async def fake_action_plan(cv_text, jd_text, score_payload, role_signal_map):
        return {"skills_to_fix_first": ["k8s"]}, {"plan_fields": ["skills"]}

    def fake_rewrite_plan(score_payload):
        return {"rewrite_suggestions": ["Quantify delivery metrics"]}, {}

    monkeypatch.setattr(orchestrator, "_run_role_analysis_step", fake_role_analysis)
    monkeypatch.setattr(orchestrator, "_run_evidence_scan_step", fake_evidence_scan)
    monkeypatch.setattr(orchestrator, "_run_scoring_step", fake_scoring)
    monkeypatch.setattr(orchestrator, "_run_gap_audit_step", fake_gap_audit)
    monkeypatch.setattr(orchestrator, "_run_action_plan_step", fake_action_plan)
    monkeypatch.setattr(orchestrator, "_run_rewrite_plan_step", fake_rewrite_plan)

    result = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="with-evidence",
        )
    )

    assert result.result["matched_keyword_evidence"][0]["value"] == "python"
    assert len(result.result["missing_keyword_evidence"]) == 1
    assert result.result["rewrite_suggestion_evidence"][0]["value"] == "Add ownership example"

    scoring_payload = next(item for item in result.artifacts if item.step == AGENT_STATE_SCORING)
    assert len(scoring_payload.payload["matched_keyword_evidence"]) == 2
    assert len(scoring_payload.payload["rewrite_suggestion_evidence"]) == 1


def test_orchestrator_is_idempotent_with_same_idempotency_key(monkeypatch):
    Session = _make_session_factory()
    db = Session()
    _seed_cv(db)

    counters = {"count": 0}

    async def fake_role_analysis(cv_text, jd_text):
        counters["count"] += 1
        return {"role_summary": "backend"}, {}

    async def fake_evidence_scan(cv_text, jd_text, role_signal_map):
        counters["count"] += 1
        return {}, {}

    async def fake_scoring(cv_text, jd_text, role_signal_map):
        counters["count"] += 1
        return {
            "fit_score": 79,
            "matched_keywords": [],
            "missing_keywords": [],
            "gap_analysis": "",
            "rewrite_suggestions": [],
        }, {}

    async def fake_gap_audit(role_signal_map, score_payload):
        counters["count"] += 1
        return {"gaps_to_address": []}, {}

    async def fake_action_plan(cv_text, jd_text, score_payload, role_signal_map):
        counters["count"] += 1
        return {"skills_to_fix_first": []}, {}

    def fake_rewrite_plan(score_payload):
        counters["count"] += 1
        return {"rewrite_suggestions": []}, {}

    monkeypatch.setattr(orchestrator, "_run_role_analysis_step", fake_role_analysis)
    monkeypatch.setattr(orchestrator, "_run_evidence_scan_step", fake_evidence_scan)
    monkeypatch.setattr(orchestrator, "_run_scoring_step", fake_scoring)
    monkeypatch.setattr(orchestrator, "_run_gap_audit_step", fake_gap_audit)
    monkeypatch.setattr(orchestrator, "_run_action_plan_step", fake_action_plan)
    monkeypatch.setattr(orchestrator, "_run_rewrite_plan_step", fake_rewrite_plan)

    first = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="stable-key",
        )
    )
    second = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="stable-key",
        )
    )

    assert first.run_id == second.run_id
    assert second.status == AGENT_STATE_COMPLETED
    assert second.current_state == AGENT_STATE_COMPLETED
    assert len(first.transitions) == 7
    assert len(second.transitions) == 7
    assert counters["count"] == 6

    run, _, _ = orchestrator.get_run_timeline(db, second.run_id)
    assert run.attempt_count == 1


def test_orchestrator_retries_only_failed_step_and_completes(monkeypatch):
    Session = _make_session_factory()
    db = Session()
    _seed_cv(db)
    scoring_calls = {"count": 0}

    async def fake_role_analysis(cv_text, jd_text):
        return {"role_summary": "backend"}, {}

    async def fake_evidence_scan(cv_text, jd_text, role_signal_map):
        return {"top_responsibilities": []}, {}

    async def fake_scoring(cv_text, jd_text, role_signal_map):
        scoring_calls["count"] += 1
        if scoring_calls["count"] == 1:
            raise RuntimeError("temporary scoring outage")
        return {
            "fit_score": 91,
            "matched_keywords": ["python"],
            "missing_keywords": [],
            "gap_analysis": "",
            "rewrite_suggestions": [],
        }, {}

    async def fake_gap_audit(role_signal_map, score_payload):
        return {"gaps_to_address": []}, {}

    async def fake_action_plan(cv_text, jd_text, score_payload, role_signal_map):
        return {"skills_to_fix_first": []}, {}

    def fake_rewrite_plan(score_payload):
        return {"rewrite_suggestions": ["Quantify impact"]}, {}

    monkeypatch.setattr(orchestrator, "_run_role_analysis_step", fake_role_analysis)
    monkeypatch.setattr(orchestrator, "_run_evidence_scan_step", fake_evidence_scan)
    monkeypatch.setattr(orchestrator, "_run_scoring_step", fake_scoring)
    monkeypatch.setattr(orchestrator, "_run_gap_audit_step", fake_gap_audit)
    monkeypatch.setattr(orchestrator, "_run_action_plan_step", fake_action_plan)
    monkeypatch.setattr(orchestrator, "_run_rewrite_plan_step", fake_rewrite_plan)

    failed = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="resume-key",
        )
    )
    assert failed.status == AGENT_STATE_FAILED
    assert failed.current_state == AGENT_STATE_SCORING

    resumed = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="resume-key",
        )
    )
    assert resumed.status == AGENT_STATE_COMPLETE
    assert resumed.current_state == AGENT_STATE_COMPLETE
    assert scoring_calls["count"] == 2
    assert resumed.result["fit_score"] == 91

    run, transitions, artifacts = orchestrator.get_run_timeline(db, resumed.run_id)
    assert run.attempt_count == 2
    assert len([item for item in transitions if item.next_state == AGENT_STATE_FAILED]) == 1
    assert len([item for item in transitions if item.trigger == AGENT_STATE_SCORING]) == 2
    assert len(artifacts) == 6


def test_rescore_integration_invokes_orchestrator_for_each_job(monkeypatch):
    Session = _make_session_factory()
    seed = Session()
    seed.add(
        CV(
            id=1,
            filename="resume.pdf",
            file_size=512,
            file_type="application/pdf",
            parsed_text="Backend software engineer profile.",
        )
    )
    seed.add_all(
        [
            Job(
                id=10,
                title="Backend Developer",
                company="Acme",
                description="Build reliable APIs.",
                source="linkedin",
                fit_score=None,
            ),
            Job(
                id=11,
                title="ML Engineer",
                company="Acme",
                description="Build recommendation models.",
                source="linkedin",
                fit_score=None,
            ),
        ]
    )
    seed.add(
        RescoreRun(
            id="rescore-1",
            status="running",
            source="linkedin",
            only_unscored=False,
            total_jobs=2,
            processed_jobs=0,
            scored_count=0,
            failed_count=0,
            failed_job_ids=[],
            message="Queued.",
        )
    )
    seed.commit()
    seed.close()

    calls = []

    async def fake_orchestrator(
        db_session,
        *,
        cv_id,
        cv_text,
        job_title,
        company,
        job_description,
        idempotency_key,
        **kwargs,
    ):
        calls.append(idempotency_key)
        return SimpleNamespace(
            status=AGENT_STATE_COMPLETE,
            result={
                "fit_score": 88,
                "matched_keywords": ["python"],
                "missing_keywords": ["ml"],
                "gap_analysis": "N/A",
                "rewrite_suggestions": ["improve"],
                "run_id": "tmp",
                "run_state": AGENT_STATE_COMPLETE,
                "run_status": AGENT_STATE_COMPLETE,
                "run_attempt_count": 1,
            },
        )

    monkeypatch.setattr(jobs_router, "SessionLocal", Session)
    monkeypatch.setattr(jobs_router, "execute_scoring_orchestrator", fake_orchestrator)
    asyncio.run(jobs_router._process_rescore_run("rescore-1"))

    verify = Session()
    run = verify.query(RescoreRun).filter(RescoreRun.id == "rescore-1").first()
    scored_jobs = verify.query(Job).filter(Job.fit_score.isnot(None)).order_by(Job.id.asc()).all()
    assert len(calls) == 2
    assert any("job:10" in key for key in calls)
    assert any("job:11" in key for key in calls)
    assert run is not None
    assert run.status == "completed"
    assert run.processed_jobs == 2
    assert run.scored_count == 2
    assert run.failed_count == 0
    assert len(scored_jobs) == 2
    assert scored_jobs[0].fit_score == 88
    verify.close()


def test_rescore_run_marks_job_failed_when_orchestrator_fails(monkeypatch):
    Session = _make_session_factory()
    seed = Session()
    seed.add(
        CV(
            id=1,
            filename="resume.pdf",
            file_size=512,
            file_type="application/pdf",
            parsed_text="Backend software engineer profile.",
        )
    )
    seed.add_all(
        [
            Job(
                id=12,
                title="Backend Developer",
                company="Acme",
                description="Build reliable APIs.",
                source="linkedin",
                fit_score=None,
                created_at=datetime(2024, 1, 1, 12, 0),
            ),
            Job(
                id=13,
                title="Data Scientist",
                company="Acme",
                description="Scale experimentation systems.",
                source="linkedin",
                fit_score=None,
                created_at=datetime(2024, 1, 2, 12, 0),
            ),
        ]
    )
    seed.add(
        RescoreRun(
            id="rescore-2",
            status="running",
            source="linkedin",
            only_unscored=False,
            total_jobs=2,
            processed_jobs=0,
            scored_count=0,
            failed_count=0,
            failed_job_ids=[],
            message="Queued.",
        )
    )
    seed.commit()
    seed.close()

    async def fake_orchestrator(
        db_session,
        *,
        cv_id,
        cv_text,
        job_title,
        company,
        job_description,
        idempotency_key,
        **kwargs,
    ):
        if idempotency_key.endswith("job:13"):
            return SimpleNamespace(
                status=AGENT_STATE_FAILED,
                result={
                    "fit_score": 0,
                    "matched_keywords": [],
                    "missing_keywords": [],
                    "gap_analysis": "llm timeout",
                    "rewrite_suggestions": [],
                    "run_id": "tmp-failed",
                    "run_state": AGENT_STATE_FAILED,
                    "run_status": AGENT_STATE_FAILED,
                },
            )
        return SimpleNamespace(
            status=AGENT_STATE_COMPLETE,
            result={
                "fit_score": 77,
                "matched_keywords": ["python"],
                "missing_keywords": [],
                "gap_analysis": "",
                "rewrite_suggestions": [],
                "run_id": "tmp-complete",
                "run_state": AGENT_STATE_COMPLETE,
                "run_status": AGENT_STATE_COMPLETE,
            },
        )

    monkeypatch.setattr(jobs_router, "SessionLocal", Session)
    monkeypatch.setattr(jobs_router, "execute_scoring_orchestrator", fake_orchestrator)
    asyncio.run(jobs_router._process_rescore_run("rescore-2"))

    verify = Session()
    run = verify.query(RescoreRun).filter(RescoreRun.id == "rescore-2").first()
    assert run is not None
    assert run.status == "completed"
    assert run.processed_jobs == 2
    assert run.scored_count == 1
    assert run.failed_count == 1
    assert run.failed_job_ids == [13]
    verify.close()


def test_scoring_stage_critic_retries_once_and_completes(monkeypatch):
    Session = _make_session_factory()
    db = Session()
    _seed_cv(db)

    scoring_calls = {"count": 0}

    async def fake_role_analysis(cv_text, jd_text):
        return {"role_summary": "backend"}, {}

    async def fake_evidence_scan(cv_text, jd_text, role_signal_map):
        return {"top_responsibilities": ["Build APIs"], "top_skills": ["Python"], "experience_signals": ["5+ years"]}, {}

    async def fake_scoring_plan(cv_text, jd_text, role_signal_map):
        return {
            "missing_keyword_hypotheses": ["kubernetes", "observability"],
            "score_schema_fields": [
                "fit_score",
                "matched_keywords",
                "missing_keywords",
                "gap_analysis",
                "rewrite_suggestions",
            ],
        }

    async def fake_scoring_executor(cv_text, jd_text, role_signal_map, scoring_plan):
        scoring_calls["count"] += 1
        if scoring_calls["count"] == 1:
            return {
                "fit_score": 88,
                "matched_keywords": ["python", "api"],
                "missing_keywords": ["kubernetes"],
                "gap_analysis": "Focus on API stability and deployment.",
                "rewrite_suggestions": ["Add stronger observability notes."],
            }
        return {
            "fit_score": 89,
            "matched_keywords": ["python", "api"],
            "missing_keywords": ["kubernetes", "observability"],
            "gap_analysis": "Gap analysis identifies missing Kubernetes and observability skills.",
            "rewrite_suggestions": ["Add stronger observability notes."],
        }

    async def fake_gap_audit(role_signal_map, score_payload):
        return {"gaps_to_address": ["kubernetes", "observability"]}, {}

    async def fake_action_plan(cv_text, jd_text, score_payload, role_signal_map):
        return {"skills_to_fix_first": ["kubernetes", "observability"]}, {}

    def fake_rewrite_plan(score_payload):
        return {"rewrite_suggestions": ["Quantify delivery metrics"]}, {}

    monkeypatch.setattr(orchestrator, "_run_role_analysis_step", fake_role_analysis)
    monkeypatch.setattr(orchestrator, "_run_evidence_scan_step", fake_evidence_scan)
    monkeypatch.setattr(orchestrator, "_run_scoring_planner_step", fake_scoring_plan)
    monkeypatch.setattr(orchestrator, "_run_scoring_execution_step", fake_scoring_executor)
    monkeypatch.setattr(orchestrator, "_run_gap_audit_step", fake_gap_audit)
    monkeypatch.setattr(orchestrator, "_run_action_plan_step", fake_action_plan)
    monkeypatch.setattr(orchestrator, "_run_rewrite_plan_step", fake_rewrite_plan)

    result = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="critic-retry-key",
        )
    )

    assert result.status == AGENT_STATE_COMPLETED
    assert result.result["fit_score"] == 89
    assert scoring_calls["count"] == 2

    scoring_artifacts = [artifact for artifact in result.artifacts if artifact.step == AGENT_STATE_SCORING]
    assert len(scoring_artifacts) == 1
    assert scoring_artifacts[0].evidence["critic_retries"] == 1
    assert "Gap analysis does not mention missing keyword 'kubernetes'." in scoring_artifacts[0].evidence["critic_feedback"]


def test_scoring_stage_fails_after_critic_retry(monkeypatch):
    Session = _make_session_factory()
    db = Session()
    _seed_cv(db)

    scoring_calls = {"count": 0}

    async def fake_role_analysis(cv_text, jd_text):
        return {"role_summary": "backend"}, {}

    async def fake_evidence_scan(cv_text, jd_text, role_signal_map):
        return {"top_responsibilities": ["Build APIs"], "top_skills": ["Python"], "experience_signals": ["5+ years"]}, {}

    async def fake_scoring_plan(cv_text, jd_text, role_signal_map):
        return {"missing_keyword_hypotheses": ["kubernetes", "observability"]}

    async def fake_scoring_executor(cv_text, jd_text, role_signal_map, scoring_plan):
        scoring_calls["count"] += 1
        return {
            "fit_score": "not-a-number",
            "matched_keywords": "python",
            "missing_keywords": ["kubernetes"],
            "gap_analysis": "Focus on leadership and quality.",
            "rewrite_suggestions": ["Add stronger observability notes."],
        }

    monkeypatch.setattr(orchestrator, "_run_role_analysis_step", fake_role_analysis)
    monkeypatch.setattr(orchestrator, "_run_evidence_scan_step", fake_evidence_scan)
    monkeypatch.setattr(orchestrator, "_run_scoring_planner_step", fake_scoring_plan)
    monkeypatch.setattr(orchestrator, "_run_scoring_execution_step", fake_scoring_executor)

    result = asyncio.run(
        orchestrator.execute_scoring_orchestrator(
            db,
            cv_id=1,
            cv_text="Experienced backend engineer.",
            job_title="Senior backend engineer",
            company="Acme",
            job_description="Build robust API services.",
            actor="test",
            source="api",
            idempotency_key="critic-fail-key",
        )
    )

    assert result.status == AGENT_STATE_FAILED
    assert result.current_state == AGENT_STATE_SCORING
    assert scoring_calls["count"] == 2
    assert result.result["fit_score"] == 0

    run, transitions, artifacts = orchestrator.get_run_timeline(db, result.run_id)
    assert run is not None
    assert run.failed_step == AGENT_STATE_SCORING
    assert len([item for item in transitions if item.next_state == AGENT_STATE_FAILED]) == 1
    assert len([artifact for artifact in artifacts if artifact.step == AGENT_STATE_SCORING]) == 0
