"""Typed state and node I/O contracts for the DeepAgent interview research graph."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import InterviewResearchQuestion, InterviewResearchQuestionBank, InterviewResearchCitation

InterviewResearchToolName = Literal[
    "search_web",
    "query_vector_store",
    "extract_jd_facts",
    "extract_candidate_profile",
    "fallback",
]

InterviewResearchStage = Literal[
    "planner",
    "tool_execution",
    "critic",
    "critic_recovery",
    "source_expansion",
    "question_synth",
    "classify",
    "fallback",
    "finalizing",
]


def _utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class InterviewResearchEvent(BaseModel):
    stage: str
    tool: str = ""
    query: str = ""
    status: str
    latency_ms: int = 0
    result_count: int = 0
    rejected_count: int = 0
    error: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utc_timestamp)

    model_config = ConfigDict(extra="ignore")


class InterviewResearchToolCall(BaseModel):
    tool: InterviewResearchToolName
    arguments: dict[str, object] = Field(default_factory=dict)
    status: str = "queued"
    latency_ms: int = 0
    result_count: int = 0

    model_config = ConfigDict(extra="ignore")


class InterviewResearchCandidate(BaseModel):
    question: str = ""
    question_text: str = ""
    tool: str = ""
    query: str = ""
    query_used: str = ""
    source_url: str = ""
    source_title: str = ""
    source_type: str = "search"
    snippet: str = ""
    confidence_score: float = 0.5
    reason: str = ""
    category: str = "technical"
    citations: list[InterviewResearchCitation] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _sync_question_fields(self) -> "InterviewResearchCandidate":
        if self.question and not self.question_text:
            object.__setattr__(self, "question_text", self.question)
        if self.question_text and not self.question:
            object.__setattr__(self, "question", self.question_text)
        if not self.query_used:
            object.__setattr__(self, "query_used", self.query)
        return self


class InterviewResearchPlan(BaseModel):
    model_id: str = ""
    prompt_version: str = "interview-research-planner-v1"
    tool_calls: list[InterviewResearchToolCall] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    emitted_status: list[InterviewResearchEvent] = Field(default_factory=list)


class InterviewResearchNodeInput(BaseModel):
    role: str
    company: str
    job_text: str
    profile_text: str
    timeout_seconds: int
    emit_session_id: str = ""
    job_id: int | None = None


class PlannerNodeInput(InterviewResearchNodeInput):
    jd_facts: dict[str, object] | None = None


class PlannerNodeOutput(BaseModel):
    plan: InterviewResearchPlan
    logs: list[InterviewResearchEvent] = Field(default_factory=list)
    fallback_plan_used: bool = False


class ToolExecutionNodeInput(InterviewResearchNodeInput):
    plan: InterviewResearchPlan


class ToolExecutionNodeOutput(BaseModel):
    candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    logs: list[InterviewResearchEvent] = Field(default_factory=list)
    dropped_domains_count: int = 0
    search_ok: bool = False
    vector_ok: bool = False


class EvidenceValidationNodeInput(BaseModel):
    candidates: list[InterviewResearchCandidate]
    role: str
    company: str
    job_text: str
    timeout_seconds: int


class EvidenceValidationNodeOutput(BaseModel):
    validated: list[InterviewResearchCandidate] = Field(default_factory=list)
    logs: list[InterviewResearchEvent] = Field(default_factory=list)
    rejected_sources: list[dict[str, str]] = Field(default_factory=list)
    dropped_count: int = 0


class RecoveryNodeInput(InterviewResearchNodeInput):
    rejected_sources: list[dict[str, str]]
    candidates: list[InterviewResearchCandidate]


class RecoveryNodeOutput(BaseModel):
    candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    logs: list[InterviewResearchEvent] = Field(default_factory=list)
    rationale: str = ""
    dropped_domains_count: int = 0


class SynthesisNodeInput(BaseModel):
    candidates: list[InterviewResearchCandidate]
    role: str
    company: str
    job_text: str
    timeout_seconds: int


class SynthesisNodeOutput(BaseModel):
    candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    logs: list[InterviewResearchEvent] = Field(default_factory=list)
    rejected_count: int = 0


class ClassificationNodeInput(BaseModel):
    candidates: list[InterviewResearchCandidate]
    role: str
    company: str
    job_text: str
    timeout_seconds: int


class ClassificationNodeOutput(BaseModel):
    classified: list[InterviewResearchQuestion] = Field(default_factory=list)
    logs: list[InterviewResearchEvent] = Field(default_factory=list)


class InterviewResearchStateSnapshot(BaseModel):
    session_id: str
    role: str
    company: str
    status: str = "initialized"
    stage: str = "initialized"
    plan: InterviewResearchPlan | None = None
    candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    validated_candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    recovered_candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    synthesized_candidates: list[InterviewResearchCandidate] = Field(default_factory=list)
    classified_questions: list[InterviewResearchQuestion] = Field(default_factory=list)
    rejected_sources: list[dict[str, str]] = Field(default_factory=list)
    question_bank: InterviewResearchQuestionBank = Field(default_factory=InterviewResearchQuestionBank)
    fallback_used: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)
    timeline: list[InterviewResearchEvent] = Field(default_factory=list)
    message: str = ""
    failure_reason: str | None = None
    created_at: str = Field(default_factory=_utc_timestamp)
    updated_at: str = Field(default_factory=_utc_timestamp)

    model_config = ConfigDict(arbitrary_types_allowed=True)
