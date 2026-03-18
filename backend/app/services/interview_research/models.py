from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

InterviewResearchCategory = Literal["behavioral", "technical", "system_design", "company_specific"]


def _utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class InterviewResearchQuestion(BaseModel):
    question: str
    tool: str
    query: str
    source_url: str
    source_title: str
    timestamp: str = Field(default_factory=_utc_timestamp)
    snippet: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class InterviewResearchQuestionBank(BaseModel):
    behavioral: list[InterviewResearchQuestion] = Field(default_factory=list)
    technical: list[InterviewResearchQuestion] = Field(default_factory=list)
    system_design: list[InterviewResearchQuestion] = Field(default_factory=list)
    company_specific: list[InterviewResearchQuestion] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)

    def all_questions(self) -> list[InterviewResearchQuestion]:
        return (
            self.behavioral
            + self.technical
            + self.system_design
            + self.company_specific
        )

    def register_source_url(self, url: str | None) -> None:
        if not url:
            return
        normalized = url.strip()
        if not normalized or normalized in self.source_urls:
            return
        self.source_urls.append(normalized)

    class Config:
        arbitrary_types_allowed = True


class InterviewResearchResult(BaseModel):
    session_id: str
    role: str
    company: str
    status: str
    question_bank: InterviewResearchQuestionBank
    fallback_used: bool = False
    message: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "company": self.company,
            "status": self.status,
            "fallback_used": self.fallback_used,
            "message": self.message,
            "question_bank": self.question_bank.model_dump(),
            "metadata": self.metadata,
        }


class StageEvent(BaseModel):
    type: str
    stage: str
    message: str
    session_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
