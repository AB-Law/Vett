"""Scoring orchestrator with explicit state transitions and persisted artifacts."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..models.score import (
    AGENT_STATE_ACTION_PLAN,
    AGENT_STATE_CANCELLED,
    AGENT_STATE_COMPLETED,
    AGENT_STATE_EVIDENCE_SCAN,
    AGENT_STATE_FAILED,
    AGENT_STATE_GAP_AUDIT,
    AGENT_STATE_QUEUED,
    AGENT_STATE_RETRY_SCHEDULED,
    AGENT_STATE_REWRITE_PLAN,
    AGENT_STATE_ROLE_ANALYSIS,
    AGENT_STATE_SCORING,
    AGENT_TERMINAL_STATES,
    AgentRun,
    AgentRunArtifact,
    AgentRunTransition,
)
from ..models.score import (
    AgentRun as AgentRunModel,
    AgentRunArtifact as AgentRunArtifactModel,
    AgentRunTransition as AgentRunTransitionModel,
)
from .llm import (
    extract_role_signal_map,
    generate_agent_score_plan,
    score_cv_against_jd,
)

logger = logging.getLogger(__name__)

LIFECYCLE_STEPS = (
    AGENT_STATE_ROLE_ANALYSIS,
    AGENT_STATE_EVIDENCE_SCAN,
    AGENT_STATE_SCORING,
    AGENT_STATE_GAP_AUDIT,
    AGENT_STATE_ACTION_PLAN,
    AGENT_STATE_REWRITE_PLAN,
)

SCORING_EXECUTOR_MAX_ATTEMPTS = 2
SCORE_TERM_PATTERN = re.compile(r"[a-z0-9][a-z0-9+\-/]*")
MISSING_HINT_PATTERN = re.compile(
    r"\b(missing|lack(?:ing)?|gap(?:s)?|need(?:s|ed)?|insufficient|improve(?:ment)?|strengthen|requires?)\b",
    re.IGNORECASE,
)
SCORE_KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "its",
    "no",
    "not",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "with",
    "your",
    "you",
}


@dataclass
class OrchestratorResult:
    run_id: str
    current_state: str
    status: str
    result: dict[str, Any]
    transitions: list[AgentRunTransition]
    artifacts: list[AgentRunArtifact]
    failure_reason: str | None = None
    failed_step: str | None = None


def _coerce_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_evidence_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _coerce_string_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return []


def _keyword_terms(value: Any) -> set[str]:
    return {
        match.group(0)
        for match in SCORE_TERM_PATTERN.finditer(_coerce_text(value).lower())
        if match.group(0) not in SCORE_KEYWORD_STOPWORDS and len(match.group(0)) > 1
    }


def _has_term_overlap(reference: Any, candidate: Any) -> bool:
    reference_terms = _keyword_terms(reference)
    if not reference_terms:
        return False
    candidate_terms = _keyword_terms(candidate)
    if not candidate_terms:
        return False
    return bool(reference_terms.intersection(candidate_terms))


def _normalize_gap_analysis_claims(gap_analysis: str) -> list[str]:
    normalized = _coerce_text(gap_analysis)
    if not normalized:
        return []

    claims: list[str] = []
    for sentence in re.split(r"[.!?\n]", normalized):
        sentence = sentence.strip()
        if not sentence or not MISSING_HINT_PATTERN.search(sentence):
            continue

        stripped_sentence = MISSING_HINT_PATTERN.sub("", sentence).strip(" -,:").strip()
        for chunk in re.split(r"\band\b|,|;|\*", stripped_sentence, flags=re.IGNORECASE):
            candidate = _coerce_text(chunk)
            if not candidate or len(_keyword_terms(candidate)) < 1:
                continue
            if len(candidate) > 90:
                continue
            claims.append(candidate)

    if claims:
        return claims

    fallback = [
        _coerce_text(part)
        for part in re.split(r"[;,\n]", normalized)
        if _coerce_text(part) and MISSING_HINT_PATTERN.search(part)
    ]
    return fallback[:8]


def _validate_scoring_output_payload(
    score_payload: Any,
    scoring_plan: dict[str, Any],
) -> list[str]:
    """Validate the raw score payload before downstream normalization."""
    if not isinstance(score_payload, dict):
        return ["Score payload must be a mapping."]

    required_fields = (
        "fit_score",
        "matched_keywords",
        "missing_keywords",
        "gap_analysis",
        "rewrite_suggestions",
    )
    missing_fields = [field for field in required_fields if field not in score_payload]
    if missing_fields:
        return [f"Score payload missing field: {name}" for name in missing_fields]

    issues: list[str] = []
    try:
        int(score_payload["fit_score"])
    except Exception:
        issues.append("Score payload field 'fit_score' must be an integer-compatible value.")

    if not isinstance(score_payload["matched_keywords"], list):
        issues.append("Score payload field 'matched_keywords' must be a list.")
    if not isinstance(score_payload["missing_keywords"], list):
        issues.append("Score payload field 'missing_keywords' must be a list.")
    if not isinstance(score_payload["rewrite_suggestions"], list):
        issues.append("Score payload field 'rewrite_suggestions' must be a list.")
    if not isinstance(score_payload["gap_analysis"], str):
        issues.append("Score payload field 'gap_analysis' must be a string.")

    missing_keywords = _coerce_string_items(score_payload.get("missing_keywords"))
    gap_analysis = score_payload.get("gap_analysis")
    for keyword in missing_keywords:
        if not _has_term_overlap(gap_analysis, keyword):
            issues.append(f"Gap analysis does not mention missing keyword '{keyword}'.")

    gap_claims = _normalize_gap_analysis_claims(_coerce_text(gap_analysis))
    for claim in gap_claims:
        if not _has_term_overlap(" ".join(missing_keywords), claim):
            issues.append(f"Gap-analysis claim '{claim}' is not represented in missing_keywords.")

    role_signal_hypotheses = _coerce_string_items(scoring_plan.get("missing_keyword_hypotheses"))
    for hypothesis in role_signal_hypotheses:
        if not (
            _has_term_overlap(gap_analysis, hypothesis)
            or _has_term_overlap(missing_keywords, hypothesis)
        ):
            issues.append(f"Planned missing-signal hypothesis '{hypothesis}' is not represented in score outputs.")

    return issues


def _reconcile_scoring_plan_with_feedback(
    scoring_plan: dict[str, Any],
    critic_feedback: list[str],
) -> dict[str, Any]:
    reconciled = _coerce_dict(scoring_plan)
    reconciled["critic_feedback"] = critic_feedback[:3]
    return reconciled


def _normalize_request_key(
    *,
    cv_id: int,
    job_title: str | None,
    company: str | None,
    job_description: str,
    schema_version: str = "v1",
) -> str:
    parts = [
        schema_version,
        str(cv_id),
        _coerce_text(job_title),
        _coerce_text(company),
        _coerce_text(job_description)[:2000],
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:80]


def _coerce_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fit_score = payload.get("fit_score")
    try:
        fit_score_value = int(fit_score)
    except (TypeError, ValueError):
        fit_score_value = 0
    if fit_score_value < 0:
        fit_score_value = 0
    if fit_score_value > 100:
        fit_score_value = 100

    return {
        "fit_score": fit_score_value,
        "matched_keywords": _coerce_list(payload.get("matched_keywords")),
        "missing_keywords": _coerce_list(payload.get("missing_keywords")),
        "gap_analysis": _coerce_text(payload.get("gap_analysis")),
        "reason": _coerce_text(payload.get("reason")),
        "rewrite_suggestions": _coerce_list(payload.get("rewrite_suggestions")),
        "matched_keyword_evidence": _coerce_evidence_records(payload.get("matched_keyword_evidence")),
        "missing_keyword_evidence": _coerce_evidence_records(payload.get("missing_keyword_evidence")),
        "rewrite_suggestion_evidence": _coerce_evidence_records(payload.get("rewrite_suggestion_evidence")),
    }


def _build_request_payload(
    *,
    cv_id: int,
    job_title: str | None,
    company: str | None,
    job_description: str,
) -> dict[str, Any]:
    return {
        "cv_id": cv_id,
        "job_title": _coerce_text(job_title),
        "company": _coerce_text(company),
        "job_description": _coerce_text(job_description),
    }


def _get_or_create_run(
    db: Session,
    *,
    idempotency_key: str,
    cv_id: int,
    request_payload: dict[str, Any],
    actor: str,
    source: str = "api",
    score_history_id: int | None = None,
) -> AgentRun:
    existing = (
        db.query(AgentRunModel)
        .filter(AgentRunModel.idempotency_key == idempotency_key)
        .first()
    )
    if existing is not None:
        return existing

    run = AgentRun(
        id=uuid.uuid4().hex,
        cv_id=cv_id,
        score_history_id=score_history_id,
        idempotency_key=idempotency_key,
        current_state=AGENT_STATE_QUEUED,
        status=AGENT_STATE_QUEUED,
        actor=actor,
        source=source,
        request_payload=request_payload,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _latest_artifact(db: Session, run_id: str, step: str) -> AgentRunArtifact | None:
    return (
        db.query(AgentRunArtifactModel)
        .filter(AgentRunArtifactModel.run_id == run_id)
        .filter(AgentRunArtifactModel.step == step)
        .order_by(desc(AgentRunArtifactModel.id))
        .first()
    )


def _count_step_attempts(db: Session, run_id: str, trigger: str) -> int:
    return (
        db.query(AgentRunTransitionModel)
        .filter(AgentRunTransitionModel.run_id == run_id)
        .filter(AgentRunTransitionModel.trigger == trigger)
        .count()
    )


def _build_transition(
    *,
    run_id: str,
    previous_state: str | None,
    next_state: str,
    trigger: str,
    attempt: int,
    idempotency_key: str,
    actor: str,
    latency_ms: int,
    failure_reason: str | None = None,
    score_history_id: int | None = None,
    source: str = "scoring_orchestrator",
) -> AgentRunTransition:
    return AgentRunTransition(
        run_id=run_id,
        score_history_id=score_history_id,
        previous_state=previous_state,
        next_state=next_state,
        trigger=trigger,
        attempt=attempt,
        failure_reason=_coerce_text(failure_reason) or None,
        latency_ms=latency_ms,
        idempotency_key=idempotency_key,
        actor=actor,
        source=source,
    )


async def _run_role_analysis_step(cv_text: str, jd_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    role_signal_map = await extract_role_signal_map(cv_text, jd_text)
    artifact = _coerce_dict(role_signal_map)
    evidence = {
        "evidence_snippets": _coerce_list(artifact.get("evidence_snippets")),
        "required_skills_count": len(_coerce_list(artifact.get("required_skills"))),
    }
    return artifact, evidence


async def _run_evidence_scan_step(cv_text: str, jd_text: str, role_signal_map: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = {
        "cv_length": len(_coerce_text(cv_text)),
        "jd_length": len(_coerce_text(jd_text)),
        "role_signal_roles": _coerce_list(role_signal_map.get("role_summary")),
        "role_signal_skills": _coerce_list(role_signal_map.get("required_skills")),
    }
    artifact = {
        "top_responsibilities": _coerce_list(role_signal_map.get("responsibilities"))[:6],
        "top_skills": _coerce_list(role_signal_map.get("required_skills"))[:8],
        "experience_signals": _coerce_list(role_signal_map.get("experience_signals"))[:6],
    }
    return artifact, evidence


async def _run_scoring_step(
    cv_text: str,
    jd_text: str,
    role_signal_map: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scoring_plan = await _run_scoring_planner_step(
        cv_text,
        jd_text,
        role_signal_map,
    )
    score_payload = {}
    critic_retries = 0

    last_critic_feedback: list[str] = []
    for attempt in range(1, SCORING_EXECUTOR_MAX_ATTEMPTS + 1):
        score_payload = await _run_scoring_execution_step(
            cv_text=cv_text,
            jd_text=jd_text,
            role_signal_map=role_signal_map,
            scoring_plan=scoring_plan,
        )
        critic_feedback = _run_score_critic_step(score_payload, scoring_plan)
        if not critic_feedback:
            break

        last_critic_feedback = critic_feedback
        if attempt >= SCORING_EXECUTOR_MAX_ATTEMPTS:
            logger.warning(
                "Score payload did not fully satisfy critic after %s attempt(s); proceeding with best result. Issues: %s",
                attempt,
                "; ".join(critic_feedback),
            )
            break
        critic_retries += 1
        scoring_plan = _reconcile_scoring_plan_with_feedback(scoring_plan, critic_feedback)

    normalized = _coerce_score_payload(_coerce_dict(score_payload))
    evidence = {
        "fit_score": normalized["fit_score"],
        "matched_count": len(normalized["matched_keywords"]),
        "missing_count": len(normalized["missing_keywords"]),
    }
    if critic_retries:
        evidence["critic_retries"] = critic_retries
        evidence["critic_feedback"] = last_critic_feedback
    return normalized, evidence


async def _run_gap_audit_step(
    role_signal_map: dict[str, Any],
    score_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing_keywords = _coerce_list(score_payload.get("missing_keywords"))
    matched_keywords = _coerce_list(score_payload.get("matched_keywords"))
    artifact = {
        "fit_score": score_payload.get("fit_score"),
        "matched_count": len(matched_keywords),
        "missing_count": len(missing_keywords),
        "gaps_to_address": missing_keywords[:8],
        "role_signal_focus": _coerce_list(role_signal_map.get("role_summary")),
    }
    evidence = {
        "gaps_detected": len(missing_keywords),
        "coverage": f"{len(matched_keywords)}/{len(missing_keywords) + len(matched_keywords)}",
    }
    return artifact, evidence


async def _run_action_plan_step(
    cv_text: str,
    jd_text: str,
    score_payload: dict[str, Any],
    role_signal_map: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = await generate_agent_score_plan(
        cv_text=cv_text,
        jd_text=jd_text,
        score_payload=score_payload,
        role_signal_map=role_signal_map,
    )
    if not isinstance(plan, dict):
        raise ValueError("Action plan payload must be a mapping.")

    artifact = _coerce_dict(plan)
    artifact.setdefault("role_signal_map", _coerce_dict(role_signal_map))
    evidence = {
        "plan_fields": sorted(list(artifact.keys())),
    }
    return artifact, evidence


def _run_rewrite_plan_step(score_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    suggestions = _coerce_list(score_payload.get("rewrite_suggestions"))
    artifact = {
        "rewrite_suggestions": suggestions[:8]
        or [
            "Add measurable impact in top CV highlights.",
            "Lead with role-aligned outcomes.",
            "Quantify one project with clear metrics.",
        ]
    }
    evidence = {
        "rewrite_count": len(artifact["rewrite_suggestions"]),
    }
    return artifact, evidence


async def _run_scoring_planner_step(
    cv_text: str,
    jd_text: str,
    role_signal_map: dict[str, Any],
) -> dict[str, Any]:
    role_signal = _coerce_dict(role_signal_map)
    missing_hypotheses = (
        _coerce_string_items(role_signal.get("required_skills"))
        + _coerce_string_items(role_signal.get("secondary_skills"))
    )
    if not missing_hypotheses:
        missing_hypotheses = _coerce_string_items(role_signal.get("responsibilities"))
    if not missing_hypotheses and _coerce_text(role_signal.get("role_summary")):
        missing_hypotheses = [_coerce_text(role_signal.get("role_summary"))]

    return {
        "score_schema_fields": [
            "fit_score",
            "matched_keywords",
            "missing_keywords",
            "gap_analysis",
            "rewrite_suggestions",
        ],
        "missing_keyword_hypotheses": missing_hypotheses[:8],
        "score_context": {
            "role_summary": _coerce_text(role_signal.get("role_summary")),
            "jd_hint": _coerce_text(jd_text)[:1400],
            "cv_hint": _coerce_text(cv_text)[:1600],
        },
    }


async def _run_scoring_execution_step(
    cv_text: str,
    jd_text: str,
    role_signal_map: dict[str, Any],
    scoring_plan: dict[str, Any],
) -> Any:
    execution_context = _coerce_dict(role_signal_map)
    execution_context["scoring_plan"] = _coerce_dict(scoring_plan)
    score_payload = await score_cv_against_jd(
        cv_text=cv_text,
        jd_text=jd_text,
        role_signal_map=execution_context,
    )
    return score_payload


def _run_score_critic_step(
    score_payload: Any,
    scoring_plan: dict[str, Any],
) -> list[str]:
    return _validate_scoring_output_payload(score_payload, scoring_plan)


def _snapshot_result_context(db: Session, run_id: str) -> dict[str, Any]:
    context: dict[str, Any] = {
        "role_signal_map": {},
        "scoring_payload": {},
        "agent_plan": {},
        "run_artifacts": [],
    }

    for step in LIFECYCLE_STEPS:
        artifact = _latest_artifact(db, run_id, step)
        if artifact is None or not isinstance(artifact.payload, dict):
            continue
        context["run_artifacts"].append(artifact)
        if step == AGENT_STATE_ROLE_ANALYSIS:
            context["role_signal_map"] = _coerce_dict(artifact.payload)
        elif step == AGENT_STATE_SCORING:
            context["scoring_payload"] = _coerce_dict(artifact.payload)
        elif step == AGENT_STATE_ACTION_PLAN:
            context["agent_plan"] = _coerce_dict(artifact.payload)

    return context


def _normalize_final_response(
    run_id: str,
    db: Session,
    *,
    current_score: dict[str, Any],
    job_title: str | None,
    company: str | None,
) -> dict[str, Any]:
    scoring_payload = _coerce_dict(current_score.get("scoring_payload"))
    if not scoring_payload:
        scoring_payload = _coerce_dict(current_score.get("scoring_artifact"))

    gap_artifact = _latest_artifact(db, run_id, AGENT_STATE_GAP_AUDIT)
    rewrite_artifact = _latest_artifact(db, run_id, AGENT_STATE_REWRITE_PLAN)

    gap_analysis = _coerce_text(current_score.get("gap_analysis"))
    if not gap_analysis and gap_artifact and isinstance(gap_artifact.payload, dict):
        gap_analysis = _coerce_text(gap_artifact.payload.get("gaps_to_address"))

    rewrite_suggestions: list[str] = _coerce_list(current_score.get("rewrite_suggestions"))
    if not rewrite_suggestions and rewrite_artifact and isinstance(rewrite_artifact.payload, dict):
        rewrite_suggestions = _coerce_list(rewrite_artifact.payload.get("rewrite_suggestions"))

    normalized = _coerce_score_payload(scoring_payload)
    normalized["reason"] = _coerce_text(current_score.get("reason")) or normalized.get("reason")
    normalized["gap_analysis"] = gap_analysis
    normalized["rewrite_suggestions"] = rewrite_suggestions
    normalized["agent_plan"] = current_score.get("agent_plan") or _coerce_dict(current_score.get("role_signal_map"))
    normalized["job_title"] = _coerce_text(job_title)
    normalized["company"] = _coerce_text(company)
    normalized["run_id"] = run_id
    return normalized


async def _run_pipeline_with_idempotent_steps(
    *,
    db: Session,
    run: AgentRun,
    cv_text: str,
    job_title: str | None,
    company: str | None,
    job_description: str,
    actor: str,
    idempotency_key: str,
) -> OrchestratorResult:
    context = _snapshot_result_context(db, run.id)
    run.attempt_count = (run.attempt_count or 0) + 1
    run.status = "running"
    db.add(run)
    db.commit()

    transitions: list[AgentRunTransition] = []

    for step in LIFECYCLE_STEPS:
        if run.status in AGENT_TERMINAL_STATES and run.status != AGENT_STATE_FAILED:
            break

        if _latest_artifact(db, run.id, step) is not None:
            run.current_state = step
            continue

        attempt = _count_step_attempts(db, run.id, step) + 1
        previous_state = run.current_state or AGENT_STATE_QUEUED
        start_ns = time.perf_counter_ns()
        transition = _build_transition(
            run_id=run.id,
            previous_state=previous_state,
            next_state=AGENT_STATE_FAILED,
            trigger=step,
            attempt=attempt,
            idempotency_key=idempotency_key,
            actor=actor,
            latency_ms=0,
            score_history_id=run.score_history_id,
        )
        db.add(transition)
        db.flush()

        try:
            if step == AGENT_STATE_ROLE_ANALYSIS:
                payload, evidence = await _run_role_analysis_step(cv_text, job_description)
                context["role_signal_map"] = payload
            elif step == AGENT_STATE_EVIDENCE_SCAN:
                payload, evidence = await _run_evidence_scan_step(
                    cv_text,
                    job_description,
                    context["role_signal_map"],
                )
            elif step == AGENT_STATE_SCORING:
                payload, evidence = await _run_scoring_step(
                    cv_text,
                    job_description,
                    context["role_signal_map"],
                )
                context["scoring_payload"] = payload
            elif step == AGENT_STATE_GAP_AUDIT:
                payload, evidence = await _run_gap_audit_step(
                    context["role_signal_map"],
                    context["scoring_payload"],
                )
            elif step == AGENT_STATE_ACTION_PLAN:
                payload, evidence = await _run_action_plan_step(
                    cv_text=cv_text,
                    jd_text=job_description,
                    score_payload=context["scoring_payload"],
                    role_signal_map=context["role_signal_map"],
                )
                context["agent_plan"] = payload
            else:
                payload, evidence = _run_rewrite_plan_step(context["scoring_payload"])

            latency_ms = max(1, int((time.perf_counter_ns() - start_ns) / 1_000_000))
            transition.next_state = step
            transition.latency_ms = latency_ms
            transition.failure_reason = None
            db.add(transition)

            artifact = AgentRunArtifact(
                run_id=run.id,
                score_history_id=run.score_history_id,
                step=step,
                actor=actor,
                source="scoring_orchestrator",
                payload=_coerce_dict(payload),
                evidence=_coerce_dict(evidence),
                attempt=attempt,
                latency_ms=latency_ms,
                transition_id=transition.id,
            )
            db.add(artifact)
            db.flush()

            run.current_state = step
            run.status = step
            run.failed_step = None
            run.failure_reason = None
            run.updated_at = artifact.created_at
            db.add(run)
            db.commit()
            transitions.append(transition)
        except Exception as exc:  # noqa: BLE001
            latency_ms = max(1, int((time.perf_counter_ns() - start_ns) / 1_000_000))
            transition.next_state = AGENT_STATE_FAILED
            transition.latency_ms = latency_ms
            transition.failure_reason = _coerce_text(exc)
            run.status = AGENT_STATE_FAILED
            run.failed_step = step
            run.failure_reason = _coerce_text(exc)
            run.current_state = step
            db.add(transition)
            db.add(run)
            db.commit()
            logger.warning(
                "Agent run %s failed at step=%s attempt=%s reason=%r",
                run.id,
                step,
                attempt,
                _coerce_text(exc),
            )

            return OrchestratorResult(
                run_id=run.id,
                current_state=run.current_state,
                status=run.status,
                failure_reason=_coerce_text(exc),
                failed_step=step,
                result=_normalize_final_response(
                    run.id,
                    db,
                    current_score=_snapshot_result_context(db, run.id),
                    job_title=job_title,
                    company=company,
                ),
                transitions=transitions + [transition],
                artifacts=db.query(AgentRunArtifactModel)
                .filter(AgentRunArtifactModel.run_id == run.id)
                .order_by(AgentRunArtifactModel.id.asc())
                .all(),
            )

    run.current_state = AGENT_STATE_COMPLETED
    run.status = AGENT_STATE_COMPLETED
    run.failure_reason = None
    run.failed_step = None
    complete_transition = _build_transition(
        run_id=run.id,
        previous_state=run.current_state,
        next_state=AGENT_STATE_COMPLETED,
        trigger="finalize",
        attempt=1,
        idempotency_key=idempotency_key,
        actor=actor,
        latency_ms=0,
        score_history_id=run.score_history_id,
    )
    db.add(complete_transition)
    db.add(run)
    db.commit()
    transitions.append(complete_transition)

    return OrchestratorResult(
        run_id=run.id,
        current_state=run.current_state,
        status=run.status,
        failure_reason=run.failure_reason,
        failed_step=run.failed_step,
        result=_normalize_final_response(
            run.id,
            db,
            current_score=_snapshot_result_context(db, run.id),
            job_title=job_title,
            company=company,
        ),
        transitions=transitions,
        artifacts=db.query(AgentRunArtifactModel)
        .filter(AgentRunArtifactModel.run_id == run.id)
        .order_by(AgentRunArtifactModel.id.asc())
        .all(),
    )


async def execute_scoring_orchestrator(
    db: Session,
    *,
    cv_id: int,
    cv_text: str,
    job_title: str | None,
    company: str | None,
    job_description: str,
    actor: str,
    source: str = "api",
    idempotency_key: str | None = None,
    score_history_id: int | None = None,
) -> OrchestratorResult:
    request_payload = _build_request_payload(
        cv_id=cv_id,
        job_title=job_title,
        company=company,
        job_description=job_description,
    )
    resolved_key = idempotency_key or _normalize_request_key(
        cv_id=cv_id,
        job_title=job_title,
        company=company,
        job_description=job_description,
    )

    run = _get_or_create_run(
        db,
        idempotency_key=resolved_key,
        cv_id=cv_id,
        request_payload=request_payload,
        actor=actor,
        source=source,
        score_history_id=score_history_id,
    )

    if run.status == AGENT_STATE_CANCELLED:
        run.status = AGENT_STATE_QUEUED
        run.current_state = AGENT_STATE_QUEUED
        db.add(run)
        db.commit()
    if run.status == AGENT_STATE_FAILED:
        run.status = AGENT_STATE_QUEUED
        run.current_state = run.current_state or AGENT_STATE_QUEUED
        run.failed_step = None
        run.failure_reason = None
        run.updated_at = datetime.utcnow()
        db.add(run)
        db.commit()

    if run.status == AGENT_STATE_COMPLETED and run.current_state == AGENT_STATE_COMPLETED:
        return OrchestratorResult(
            run_id=run.id,
            current_state=run.current_state,
            status=run.status,
            failure_reason=_coerce_text(run.failure_reason),
            failed_step=run.failed_step,
            result=_normalize_final_response(
                run.id,
                db,
                current_score=_snapshot_result_context(db, run.id),
                job_title=job_title,
                company=company,
            ),
            transitions=db.query(AgentRunTransitionModel)
            .filter(AgentRunTransitionModel.run_id == run.id)
            .order_by(AgentRunTransitionModel.created_at.asc())
            .all(),
            artifacts=db.query(AgentRunArtifactModel)
            .filter(AgentRunArtifactModel.run_id == run.id)
            .order_by(AgentRunArtifactModel.id.asc())
            .all(),
        )

    result = await _run_pipeline_with_idempotent_steps(
        db=db,
        run=run,
        cv_text=cv_text,
        job_title=job_title,
        company=company,
        job_description=job_description,
        actor=actor,
        idempotency_key=resolved_key,
    )
    result.result["run_id"] = result.run_id
    result.result["run_state"] = result.current_state
    result.result["run_status"] = result.status
    result.result["run_attempt_count"] = run.attempt_count
    return result


def link_score_history_to_run(db: Session, run_id: str, score_history_id: int) -> None:
    run = db.query(AgentRunModel).filter(AgentRunModel.id == run_id).first()
    if run is None:
        return
    if run.score_history_id == score_history_id:
        return

    run.score_history_id = score_history_id
    db.add(run)
    db.query(AgentRunArtifactModel).filter(
        AgentRunArtifactModel.run_id == run_id,
        AgentRunArtifactModel.score_history_id == None,  # noqa: E711
    ).update(
        {AgentRunArtifactModel.score_history_id: score_history_id},
        synchronize_session=False,
    )
    db.query(AgentRunTransitionModel).filter(
        AgentRunTransitionModel.run_id == run_id,
        AgentRunTransitionModel.score_history_id == None,  # noqa: E711
    ).update(
        {AgentRunTransitionModel.score_history_id: score_history_id},
        synchronize_session=False,
    )
    db.commit()


def get_run_timeline(db: Session, run_id: str) -> tuple[AgentRun | None, list[AgentRunTransition], list[AgentRunArtifact]]:
    run = db.query(AgentRunModel).filter(AgentRunModel.id == run_id).first()
    if not run:
        return None, [], []
    transitions = (
        db.query(AgentRunTransitionModel)
        .filter(AgentRunTransitionModel.run_id == run_id)
        .order_by(AgentRunTransitionModel.created_at.asc())
        .all()
    )
    artifacts = (
        db.query(AgentRunArtifactModel)
        .filter(AgentRunArtifactModel.run_id == run_id)
        .order_by(AgentRunArtifactModel.id.asc())
        .all()
    )
    return run, transitions, artifacts

