"""LangGraph-style interview research runner with typed stage contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime
import time
from typing import Any
from collections.abc import Awaitable, Callable

from sqlalchemy.orm import Session

from ...config import get_settings
from ...models.cv import CV
from .agent_state import (
    InterviewResearchCandidate,
    InterviewResearchEvent,
    InterviewResearchPlan,
    PlannerNodeInput,
    PlannerNodeOutput,
    ToolExecutionNodeInput,
    ToolExecutionNodeOutput,
    EvidenceValidationNodeInput,
    EvidenceValidationNodeOutput,
    RecoveryNodeInput,
    RecoveryNodeOutput,
    SynthesisNodeInput,
    SynthesisNodeOutput,
    InterviewResearchStateSnapshot,
)
from .models import (
    InterviewResearchCitation,
    InterviewResearchQuestion,
    InterviewResearchQuestionBank,
    InterviewResearchResult,
)
from .orchestrator import (
    CRITIC_TIMEOUT_SECONDS,
    QUESTION_SYNTH_TIMEOUT_SECONDS,
    CLASSIFIER_TIMEOUT_SECONDS,
    PLANNER_TIMEOUT_SECONDS,
    PLANNER_PROMPT_VERSION,
    CLASSIFIER_PROMPT_VERSION,
    CRITIC_PROMPT_VERSION,
    CRITIC_RECOVERY_PROMPT_VERSION,
    QUESTION_SYNTH_PROMPT_VERSION,
    TARGET_WEB_SOURCE_PAGES,
    MAX_SOURCE_EXPANSION_ROUNDS,
    _append_to_category,
    _apply_minimums,
    _count_web_sources,
    _dedupe,
    _dedupe_candidate_rows,
    _extract_candidate_profile_context,
    _planner_call,
    _run_tool_calls,
    _critic_candidates,
    _critic_recovery_candidates,
    _synthesize_questions_from_candidates,
    _classify_candidates,
    InterviewResearchRunContext,
)
from .tools import search_web, query_vector_store, extract_jd_facts


def _iso_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_candidate_list(raw: list[dict[str, Any]]) -> list[InterviewResearchCandidate]:
    return [InterviewResearchCandidate.model_validate(item) for item in raw]


def _to_candidate_dicts(rows: list[InterviewResearchCandidate]) -> list[dict[str, Any]]:
    return [row.model_dump() for row in rows]


def _to_events(records: list[dict[str, Any]]) -> list[InterviewResearchEvent]:
    events: list[InterviewResearchEvent] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        events.append(
            InterviewResearchEvent(
                stage=str(item.get("stage", "")),
                tool=str(item.get("tool", "")),
                query=str(item.get("query", "")),
                status=str(item.get("status", "ok")),
                latency_ms=int(item.get("latency_ms", 0) or 0),
                result_count=int(item.get("result_count", 0) or 0),
                rejected_count=int(item.get("rejected_count", 0) or 0),
                error=str(item.get("error", "")),
                metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                timestamp=_iso_timestamp(),
            )
        )
    return events


class InterviewResearchAgentRunner:
    def __init__(self, db: Session, run_context: InterviewResearchRunContext):
        self.db = db
        self.run_context = run_context
        self.role = (run_context.role or "Interview role").strip()
        self.company = (run_context.company or "this company").strip()
        self.emit = run_context.emit
        self.timeout_seconds = max(2, int(run_context.timeout_seconds or 20))
        self.deadline_ts = time.perf_counter() + self.timeout_seconds
        self.cancel_event = getattr(run_context, "cancel_event", None)
        self.state = InterviewResearchStateSnapshot(
            session_id="",
            role=self.role,
            company=self.company,
        )
        self.settings = get_settings()
        self.tool_timeout_seconds = (
            self.settings.interview_research_tool_timeout_seconds
            if hasattr(self.settings, "interview_research_tool_timeout_seconds")
            else 10
        )
        self.node_retry_budget: dict[str, int] = {
            "planner": 2,
            "tool_execution": 2,
            "evidence": 1,
            "recovery": 2,
            "synthesis": 2,
            "classification": 2,
            "constraints": 1,
        }

    async def _emit(self, payload: dict[str, object]) -> None:
        if not self.emit:
            return
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise asyncio.CancelledError("Interview research run was cancelled by client request.")
        if self.deadline_ts and time.perf_counter() > self.deadline_ts:
            raise asyncio.TimeoutError("Interview research global timeout reached.")
        if payload.get("type") == "status":
            stage = str(payload.get("stage", ""))
            self.state.timeline.append(
                InterviewResearchEvent(
                    stage=stage,
                    tool=str(payload.get("tool", "")),
                    query=str(payload.get("query", "")),
                    status="ok",
                    latency_ms=int(payload.get("latency_ms", 0) or 0),
                    result_count=0,
                    rejected_count=0,
                    error="",
                    metadata={"message": str(payload.get("message", ""))},
                )
            )
        await self.emit(payload)

    async def _emit_status(
        self,
        stage: str,
        message: str,
        *,
        payload: dict[str, object] | None = None,
        status: str = "ok",
        tool: str = "",
        query: str = "",
        rejected_count: int = 0,
        result_count: int = 0,
    ) -> None:
        normalized = {
            "type": "status",
            "stage": stage,
            "message": message,
            "status": status,
            "tool": tool,
            "query": query,
            "rejected_count": rejected_count,
            "result_count": result_count,
            "payload": payload or {},
        }
        await self._emit(normalized)

    async def _emit_status_best_effort(
        self,
        stage: str,
        message: str,
        *,
        payload: dict[str, object] | None = None,
        status: str = "ok",
        tool: str = "",
        query: str = "",
        rejected_count: int = 0,
        result_count: int = 0,
    ) -> None:
        try:
            await self._emit_status(
                stage=stage,
                message=message,
                payload=payload,
                status=status,
                tool=tool,
                query=query,
                rejected_count=rejected_count,
                result_count=result_count,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Never let status emission failures override pipeline error handling.
            return

    def _node_budget_seconds(self, configured_seconds: int) -> float:
        hard_min = 0.5
        remaining = self.deadline_ts - time.perf_counter()
        if remaining <= 0:
            return 0
        candidate = min(configured_seconds, remaining)
        return max(candidate, hard_min)

    async def _run_node(
        self,
        node_name: str,
        action: Callable[[], Awaitable[Any]],
        configured_seconds: int,
    ) -> Any:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise asyncio.CancelledError("Interview research run was cancelled by client request.")

        attempts = max(1, self.node_retry_budget.get(node_name, 1))
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                await self._emit_status(
                    stage=node_name,
                    message=f"Starting {node_name} node (attempt {attempt + 1}/{attempts}).",
                    payload={"attempt": attempt + 1, "budget_seconds": configured_seconds},
                    tool=node_name,
                )
                return await asyncio.wait_for(
                    action(),
                    timeout=self._node_budget_seconds(configured_seconds),
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                await self._emit_status_best_effort(
                    stage=node_name,
                    message=f"{node_name} node timed out.",
                    status="timeout",
                    payload={"attempt": attempt + 1, "budget_seconds": configured_seconds},
                )
                break
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                if attempt + 1 >= attempts:
                    await self._emit_status_best_effort(
                        stage=node_name,
                        message=f"{node_name} node failed after {attempt + 1} attempts.",
                        status="error",
                        payload={"attempt": attempt + 1, "error": str(exc)},
                    )
                    break
                await self._emit_status_best_effort(
                    stage=node_name,
                    message=f"{node_name} node failed; retrying.",
                    status="retry",
                    payload={"attempt": attempt + 1, "error": str(exc)},
                )
                await asyncio.sleep(0.05)
        if isinstance(last_error, Exception):
            raise last_error
        raise RuntimeError(f"{node_name} node did not complete.")

    def _annotate_with_citations(self, question: object) -> None:
        try:
            if not isinstance(question, InterviewResearchQuestion):
                return
            if question.citations:
                return
            citation = InterviewResearchCitation(
                source_url=getattr(question, "source_url", ""),
                source_title=getattr(question, "source_title", ""),
                snippet=(getattr(question, "snippet", "") or "").strip(),
                confidence=min(max(float(getattr(question, "confidence_score", 0.0) or 0.0), 0.0), 1.0),
            )
            if citation.source_url or citation.source_title or citation.snippet:
                question.citations = [citation]
        except Exception:
            return

    async def constraints_node(
        self,
        question_bank: InterviewResearchQuestionBank,
        *,
        candidate_company: str | None,
    ) -> dict[str, object]:
        fallback_used = False
        if _apply_minimums(
            question_bank,
            role=self.role,
            company=self.company,
            candidate_company=candidate_company,
        ):
            fallback_used = True
        await self._emit_status(
            stage="constraints",
            message="Coverage constraints applied.",
            payload={
                "fallback_used": fallback_used,
                "behavioral_count": len(question_bank.behavioral),
                "technical_count": len(question_bank.technical),
                "system_design_count": len(question_bank.system_design),
                "company_specific_count": len(question_bank.company_specific),
            },
        )
        return {"fallback_used": fallback_used, "constraints_applied": True}

    async def planner_node(self, payload: PlannerNodeInput) -> PlannerNodeOutput:
        model_id, plan, planner_log = await _planner_call(
            role=payload.role,
            company=payload.company,
            job_text=payload.job_text,
            profile_text=payload.profile_text,
            jd_facts=payload.jd_facts,
            emit=self._emit,
            timeout_seconds=PLANNER_TIMEOUT_SECONDS,
        )
        plan_calls = plan.get("tool_calls", [])
        tool_calls = []
        for item in plan_calls:
            if not isinstance(item, dict):
                continue
            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                {
                    "tool": str(item.get("tool", "search_web")),
                    "arguments": arguments,
                }
            )
        return PlannerNodeOutput(
            plan=InterviewResearchPlan(
                model_id=model_id,
                prompt_version=PLANNER_PROMPT_VERSION,
                tool_calls=[{"tool": item["tool"], "arguments": item["arguments"]} for item in tool_calls],
                metadata={"model_id": model_id, "model_latency_ms": 0},
                emitted_status=_to_events(planner_log),
            ),
            logs=_to_events(planner_log),
            fallback_plan_used=any(item.get("tool") == "fallback" for item in plan_calls),
        )

    async def tool_execution_node(self, payload: ToolExecutionNodeInput) -> ToolExecutionNodeOutput:
        candidates, action_log, dropped_domains_count, search_ok, vector_ok = await _run_tool_calls(
            db=self.db,
            role=payload.role,
            company=payload.company,
            job=self.run_context.job,
            profile_text=payload.profile_text,
            plan=payload.plan.model_dump(),
            emit=self._emit,
            timeout_seconds=self.tool_timeout_seconds,
        )
        return ToolExecutionNodeOutput(
            candidates=_to_candidate_list(candidates),
            logs=_to_events(action_log),
            dropped_domains_count=int(dropped_domains_count),
            search_ok=bool(search_ok),
            vector_ok=bool(vector_ok),
        )

    async def evidence_validation_node(self, payload: EvidenceValidationNodeInput) -> EvidenceValidationNodeOutput:
        candidates_dicts = _to_candidate_dicts(payload.candidates)
        kept, critic_log, rejected_sources = await _critic_candidates(
            candidates_dicts,
            role=payload.role,
            company=payload.company,
            job_text=payload.job_text,
            emit=self._emit,
            timeout_seconds=CRITIC_TIMEOUT_SECONDS,
        )
        return EvidenceValidationNodeOutput(
            validated=_to_candidate_list(kept),
            logs=_to_events(critic_log),
            rejected_sources=[row if isinstance(row, dict) else {} for row in rejected_sources],
        )

    async def recovery_node(self, payload: RecoveryNodeInput) -> RecoveryNodeOutput:
        if not payload.rejected_sources:
            return RecoveryNodeOutput(candidates=[], logs=[], rationale="")
        recovered, recovery_log, rationale, dropped_domains_count = await _critic_recovery_candidates(
            self.db,
            role=payload.role,
            company=payload.company,
            job=self.run_context.job,
            job_text=payload.job_text,
            rejected_sources=payload.rejected_sources,
            emit=self._emit,
            timeout_seconds=self.tool_timeout_seconds,
        )
        return RecoveryNodeOutput(
            candidates=_to_candidate_list(recovered),
            logs=_to_events(recovery_log),
            rationale=str(rationale),
            dropped_domains_count=int(dropped_domains_count),
        )

    async def synthesis_node(self, payload: SynthesisNodeInput) -> SynthesisNodeOutput:
        candidates_dict = _to_candidate_dicts(payload.candidates)
        synthesized, synth_log, rejected_count = await _synthesize_questions_from_candidates(
            candidates_dict,
            role=payload.role,
            company=payload.company,
            job_text=payload.job_text,
            emit=self._emit,
            timeout_seconds=QUESTION_SYNTH_TIMEOUT_SECONDS,
        )
        return SynthesisNodeOutput(
            candidates=_to_candidate_list(synthesized),
            logs=_to_events(synth_log),
            rejected_count=int(rejected_count or 0),
        )

    async def classify_node(self, payload: SynthesisNodeOutput, *, job_text: str) -> list[InterviewResearchQuestion]:
        payload_dicts = _to_candidate_dicts(payload.candidates)
        classified = await _classify_candidates(
            payload_dicts,
            role=self.role,
            company=self.company,
            job_text=job_text,
            emit=self._emit,
            timeout_seconds=CLASSIFIER_TIMEOUT_SECONDS,
        )
        return classified

    async def run(self) -> InterviewResearchResult:
        job_text = (self.run_context.job.description if self.run_context.job else "").strip()
        latest_cv = self.db.query(CV).order_by(CV.id.desc()).first()
        profile_text = (latest_cv.parsed_text if latest_cv else "").strip()
        candidate_company, resolved_profile_summary = _extract_candidate_profile_context(self.db, profile_text)
        profile_text_for_planner = resolved_profile_summary or profile_text

        question_bank = InterviewResearchQuestionBank()
        all_log: list[dict[str, Any]] = []
        rejected_sources: list[dict[str, str]] = []
        fallback_used = False
        critic_recovery_rationale = ""
        question_synth_rejected_count = 0
        candidates: list[InterviewResearchCandidate] = []
        search_ok = False
        vector_ok = False
        dropped_domains_count = 0

        await self._emit_status(
            stage="initialized",
            message=f"Researching interview questions for {self.role} at {self.company}.",
        )

        try:
            if self.deadline_ts and time.perf_counter() > self.deadline_ts:
                raise TimeoutError("Interview research global timeout reached before start.")

            try:
                jd_facts = await asyncio.wait_for(
                    extract_jd_facts(self.role, self.company, job_text),
                    timeout=self._node_budget_seconds(self.tool_timeout_seconds),
                )
            except Exception:
                await self._emit_status(
                    stage="extract_jd_facts",
                    message="JD fact extraction skipped due timeout or failure.",
                    status="warn",
                    payload={"reason": "fallback"},
                )
                jd_facts = {}

            plan_input = PlannerNodeInput(
                role=self.role,
                company=self.company,
                job_text=job_text,
                profile_text=profile_text_for_planner,
                timeout_seconds=PLANNER_TIMEOUT_SECONDS,
                jd_facts=jd_facts,
                emit_session_id=self.state.session_id,
                job_id=self.run_context.job.id if self.run_context.job else None,
            )
            planner_output = await self._run_node(
                "planner",
                lambda: self.planner_node(plan_input),
                PLANNER_TIMEOUT_SECONDS,
            )
            self.state.plan = planner_output.plan
            self.state.timeline.extend(planner_output.logs)
            all_log.extend([event.model_dump() for event in planner_output.logs])
            await self._emit_status(
                stage="planner",
                message="Planner finished and emitted tool plan.",
                payload={"tool_calls": len(planner_output.plan.tool_calls), "fallback_plan_used": planner_output.fallback_plan_used},
            )

            tool_input = ToolExecutionNodeInput(
                role=self.role,
                company=self.company,
                job_text=job_text,
                profile_text=profile_text_for_planner,
                timeout_seconds=self.tool_timeout_seconds,
                emit_session_id=self.state.session_id,
                job_id=self.run_context.job.id if self.run_context.job else None,
                plan=planner_output.plan,
            )
            tool_output = await self._run_node(
                "tool_execution",
                lambda: self.tool_execution_node(tool_input),
                self.tool_timeout_seconds,
            )
            self.state.timeline.extend(tool_output.logs)
            all_log.extend([event.model_dump() for event in tool_output.logs])
            candidates = tool_output.candidates
            dropped_domains_count = tool_output.dropped_domains_count
            search_ok = tool_output.search_ok
            vector_ok = tool_output.vector_ok
            await self._emit_status(
                stage="tool_execution",
                message="Tool execution completed.",
                payload={
                    "candidate_count": len(candidates),
                    "search_ok": search_ok,
                    "vector_ok": vector_ok,
                    "dropped_domains_count": dropped_domains_count,
                },
            )

            if not candidates:
                fallback_used = True
                fallback_queries = [
                    f"{self.role} {self.company} interview process and engineering expectations".strip(),
                    f"{self.company} engineering blog {self.role}".strip(),
                    f"{self.role} {self.company} system design interview questions".strip(),
                    f"{self.role} technical interview questions".strip(),
                ]
                fallback_hits: list[dict[str, Any]] = []
                for fallback_query in fallback_queries:
                    try:
                        fetched = await asyncio.wait_for(search_web(fallback_query, max_items=4), timeout=self.tool_timeout_seconds)
                        fallback_hits.extend(fetched)
                    except Exception:
                        pass
                if self.run_context.job is not None:
                    try:
                        fallback_hits.extend(
                            await asyncio.wait_for(
                                query_vector_store(self.db, f"{self.role} interview questions", job_id=self.run_context.job.id, max_items=3, fetch_limit=2),
                                timeout=self.tool_timeout_seconds,
                            )
                        )
                    except Exception:
                        pass
                candidates = _to_candidate_list(fallback_hits)
                await self._emit_status(
                    stage="fallback",
                    message="No primary candidates found; fallback search used.",
                    payload={"candidate_count": len(candidates)},
                )

            validation_input = EvidenceValidationNodeInput(
                candidates=candidates,
                role=self.role,
                company=self.company,
                job_text=job_text,
                timeout_seconds=CRITIC_TIMEOUT_SECONDS,
            )
            validation = await self._run_node(
                "evidence",
                lambda: self.evidence_validation_node(validation_input),
                CRITIC_TIMEOUT_SECONDS,
            )
            self.state.timeline.extend(validation.logs)
            all_log.extend([event.model_dump() for event in validation.logs])
            rejected_sources = validation.rejected_sources
            candidates = _dedupe_candidate_rows(_to_candidate_dicts(validation.validated))
            candidates = _to_candidate_list(candidates)
            await self._emit_status(
                stage="evidence",
                message="Initial evidence validation completed.",
                payload={"candidate_count": len(candidates), "rejected_count": len(rejected_sources)},
            )

            if not candidates:
                recovery_payload = RecoveryNodeInput(
                    role=self.role,
                    company=self.company,
                    job_text=job_text,
                    profile_text=profile_text_for_planner,
                    timeout_seconds=self.tool_timeout_seconds,
                    emit_session_id=self.state.session_id,
                    rejected_sources=rejected_sources,
                    candidates=[],
                    job_id=self.run_context.job.id if self.run_context.job else None,
                )
                try:
                    recovery = await self._run_node(
                        "recovery",
                        lambda recovery_payload=recovery_payload: self.recovery_node(recovery_payload),
                        self.tool_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    await self._emit_status_best_effort(
                        stage="recovery",
                        message="Primary evidence recovery timed out; continuing with fallback constraints.",
                        status="warn",
                    )
                    recovery = RecoveryNodeOutput(candidates=[], logs=[], rationale="")
                dropped_domains_count += recovery.dropped_domains_count
                critic_recovery_rationale = recovery.rationale
                self.state.timeline.extend(recovery.logs)
                all_log.extend([event.model_dump() for event in recovery.logs])
                recovered = _dedupe_candidate_rows(_to_candidate_dicts(recovery.candidates))
                candidates = _to_candidate_list(recovered)
                await self._emit_status(
                    stage="recovery",
                    message="Primary evidence recovery completed.",
                    payload={"candidate_count": len(candidates)},
                )
                recovery_validation = await self.evidence_validation_node(
                    EvidenceValidationNodeInput(
                        candidates=candidates,
                        role=self.role,
                        company=self.company,
                        job_text=job_text,
                        timeout_seconds=CRITIC_TIMEOUT_SECONDS,
                    )
                )
                self.state.timeline.extend(recovery_validation.logs)
                all_log.extend([event.model_dump() for event in recovery_validation.logs])
                candidates = _to_candidate_list(_dedupe_candidate_rows(_to_candidate_dicts(recovery_validation.validated)))
                rejected_sources.extend(recovery_validation.rejected_sources)
                await self._emit_status(
                    stage="recovery",
                    message="Recovery evidence validation completed.",
                    payload={"candidate_count": len(candidates)},
                )

            web_source_count = _count_web_sources(_to_candidate_dicts(candidates))
            source_expansion_rounds = 0
            while web_source_count < TARGET_WEB_SOURCE_PAGES and source_expansion_rounds < MAX_SOURCE_EXPANSION_ROUNDS:
                source_expansion_rounds += 1
                expansion_input = RecoveryNodeInput(
                    role=self.role,
                    company=self.company,
                    job_text=job_text,
                    profile_text=profile_text_for_planner,
                    timeout_seconds=self.tool_timeout_seconds,
                    emit_session_id=self.state.session_id,
                    rejected_sources=rejected_sources,
                    candidates=candidates,
                    job_id=self.run_context.job.id if self.run_context.job else None,
                )
                try:
                    expansion = await self._run_node(
                        "recovery",
                        lambda expansion_input=expansion_input: self.recovery_node(expansion_input),
                        self.tool_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    await self._emit_status_best_effort(
                        stage="recovery",
                        message="Source expansion timed out; proceeding with available sources.",
                        status="warn",
                        payload={"round": source_expansion_rounds, "web_source_count": web_source_count},
                    )
                    break
                if not expansion.candidates:
                    break
                dropped_domains_count += expansion.dropped_domains_count
                self.state.timeline.extend(expansion.logs)
                all_log.extend([event.model_dump() for event in expansion.logs])
                recovered = _dedupe_candidate_rows(_to_candidate_dicts(expansion.candidates))
                candidates = _to_candidate_list(_dedupe_candidate_rows(_to_candidate_dicts(candidates + _to_candidate_list(recovered))))
                web_source_count = _count_web_sources(_to_candidate_dicts(candidates))
                await self._emit_status(
                    stage="recovery",
                    message="Source expansion round completed.",
                    payload={"web_source_count": web_source_count, "round": source_expansion_rounds, "candidate_count": len(candidates)},
                )

            synth_output = await self._run_node(
                "synthesis",
                lambda: self.synthesis_node(
                    SynthesisNodeInput(
                        candidates=candidates,
                        role=self.role,
                        company=self.company,
                        job_text=job_text,
                        timeout_seconds=QUESTION_SYNTH_TIMEOUT_SECONDS,
                    )
                ),
                QUESTION_SYNTH_TIMEOUT_SECONDS,
            )
            self.state.timeline.extend(synth_output.logs)
            all_log.extend([event.model_dump() for event in synth_output.logs])
            candidates = _to_candidate_list(_dedupe_candidate_rows(_to_candidate_dicts(synth_output.candidates)))
            question_synth_rejected_count = synth_output.rejected_count
            await self._emit_status(
                stage="synthesis",
                message="Question synthesis completed.",
                payload={"candidate_count": len(candidates)},
            )

            enriched = await self._run_node(
                "classification",
                lambda: self.classify_node(synth_output, job_text=job_text),
                CLASSIFIER_TIMEOUT_SECONDS,
            )
            for question in enriched:
                if not question.source_type:
                    question.source_type = "search"
                if not question.reason:
                    question.reason = "Classified from model/heuristics."
                if not question.query_used:
                    question.query_used = question.query
                if not question.confidence_score:
                    question.confidence_score = 0.25
                self._annotate_with_citations(question)
                _append_to_category(question_bank, question)
                question_bank.register_source_url(question.source_url)
            await self._emit_status(
                stage="classification",
                message="Question classification completed.",
                payload={"total_questions": len(question_bank.all_questions())},
            )

            _dedupe(question_bank)
            constraints_result = await self._run_node(
                "constraints",
                lambda: self.constraints_node(
                    question_bank,
                    candidate_company=candidate_company,
                ),
                5,
            )
            if bool(constraints_result.get("fallback_used")):
                fallback_used = True

            if not search_ok and not vector_ok and len(question_bank.all_questions()) < (2 + 4 + 2 + 3):
                fallback_used = True
                await self._emit_status(
                    stage="constraints",
                    message="Fallback strategy marked due to source coverage.",
                    payload={"search_ok": search_ok, "vector_ok": vector_ok},
                )

            return InterviewResearchResult(
                session_id="",
                role=self.role,
                company=self.company,
                status="completed",
                question_bank=question_bank,
                fallback_used=fallback_used,
                message="Interview questions generated.",
                metadata={
                    "total_questions": len(question_bank.all_questions()),
                    "research_log": all_log,
                    "model_input_prompt_version": PLANNER_PROMPT_VERSION,
                    "classification_version": CLASSIFIER_PROMPT_VERSION,
                    "critic_version": CRITIC_PROMPT_VERSION,
                    "critic_recovery_version": CRITIC_RECOVERY_PROMPT_VERSION,
                    "critic_recovery_rationale": critic_recovery_rationale,
                    "question_synth_version": QUESTION_SYNTH_PROMPT_VERSION,
                    "question_synth_rejected_count": question_synth_rejected_count,
                    "target_web_source_pages": TARGET_WEB_SOURCE_PAGES,
                    "achieved_web_source_pages": _count_web_sources(_to_candidate_dicts(candidates)),
                    "tools_used": sorted(
                        {
                            item.get("tool")
                            for item in all_log
                            if isinstance(item.get("tool"), str)
                        }
                    ),
                    "dropped_domains_count": int(dropped_domains_count),
                    "rejected_sources": rejected_sources[:30],
                    "rejected_source_count": len(rejected_sources),
                    "question_bank_snapshot": {
                        "behavioral": len(question_bank.behavioral),
                        "technical": len(question_bank.technical),
                        "system_design": len(question_bank.system_design),
                        "company_specific": len(question_bank.company_specific),
                    },
                    "progress_timeline": [event.model_dump() for event in self.state.timeline],
                },
            )
        except Exception as exc:
            message = str(exc)
            if isinstance(exc, asyncio.CancelledError):
                message = "Interview research cancelled by user."
            if isinstance(exc, asyncio.TimeoutError):
                fallback_used = True
                _apply_minimums(
                    question_bank,
                    role=self.role,
                    company=self.company,
                    candidate_company=candidate_company,
                )
                await self._emit_status_best_effort(
                    stage="constraints",
                    message="Global timeout reached; returning fallback-safe question set.",
                    status="warn",
                    payload={"error": message},
                )
                return InterviewResearchResult(
                    session_id="",
                    role=self.role,
                    company=self.company,
                    status="completed",
                    question_bank=question_bank,
                    fallback_used=True,
                    message="Interview questions generated with timeout fallback.",
                    metadata={
                        "total_questions": len(question_bank.all_questions()),
                        "research_log": all_log,
                        "model_input_prompt_version": PLANNER_PROMPT_VERSION,
                        "classification_version": CLASSIFIER_PROMPT_VERSION,
                        "critic_version": CRITIC_PROMPT_VERSION,
                        "critic_recovery_version": CRITIC_RECOVERY_PROMPT_VERSION,
                        "critic_recovery_rationale": critic_recovery_rationale,
                        "question_synth_version": QUESTION_SYNTH_PROMPT_VERSION,
                        "question_synth_rejected_count": question_synth_rejected_count,
                        "target_web_source_pages": TARGET_WEB_SOURCE_PAGES,
                        "achieved_web_source_pages": _count_web_sources(_to_candidate_dicts(candidates))
                        if "candidates" in locals()
                        else 0,
                        "tools_used": sorted(
                            {
                                item.get("tool")
                                for item in all_log
                                if isinstance(item.get("tool"), str)
                            }
                        ),
                        "dropped_domains_count": int(dropped_domains_count),
                        "rejected_sources": rejected_sources[:30],
                        "rejected_source_count": len(rejected_sources),
                        "question_bank_snapshot": {
                            "behavioral": len(question_bank.behavioral),
                            "technical": len(question_bank.technical),
                            "system_design": len(question_bank.system_design),
                            "company_specific": len(question_bank.company_specific),
                        },
                        "progress_timeline": [event.model_dump() for event in self.state.timeline],
                        "timeout_degraded": True,
                        "timeout_error": message,
                    },
                )
            await self._emit_status_best_effort(
                stage="failed",
                message=message,
                status="error",
                payload={"error": message},
            )
            return InterviewResearchResult(
                session_id="",
                role=self.role,
                company=self.company,
                status="failed",
                question_bank=question_bank,
                fallback_used=True,
                message=f"Recovered from error: {message}",
                metadata={
                    "total_questions": len(question_bank.all_questions()),
                    "research_log": all_log,
                    "model_input_prompt_version": PLANNER_PROMPT_VERSION,
                    "critic_version": CRITIC_PROMPT_VERSION,
                    "critic_recovery_version": CRITIC_RECOVERY_PROMPT_VERSION,
                    "critic_recovery_rationale": critic_recovery_rationale,
                    "question_synth_version": QUESTION_SYNTH_PROMPT_VERSION,
                    "question_synth_rejected_count": question_synth_rejected_count,
                    "target_web_source_pages": TARGET_WEB_SOURCE_PAGES,
                    "achieved_web_source_pages": _count_web_sources(_to_candidate_dicts(candidates)) if "candidates" in locals() else 0,
                    "tools_used": ["fallback"],
                    "rejected_sources": rejected_sources[:30],
                    "rejected_source_count": len(rejected_sources),
                },
            )


async def run_interview_research(db: Session, run_context: InterviewResearchRunContext) -> InterviewResearchResult:
    runner = InterviewResearchAgentRunner(db, run_context)
    return await runner.run()
