from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any, Callable, Literal
import time

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.cv import CV
from ..models.score import Job, LocalContextToolCallAudit, ScoreHistory


RiskLevel = Literal["low", "medium", "high"]


class ToolDispatchError(Exception):
    """Base error raised for all deterministic tooling failures."""

    def __init__(self, code: str, message: str, recovery_hint: str | None = None, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery_hint = recovery_hint
        self.status_code = status_code


class ToolAuthorizationError(ToolDispatchError):
    """Raised when caller context is not allowlisted for the requested tool."""


class ToolSchemaError(ToolDispatchError):
    """Raised when tool input fails schema validation."""


class ToolExecutionError(ToolDispatchError):
    """Raised when tool runtime execution cannot be completed."""


@dataclass(frozen=True)
class ToolExecutionContext:
    request_uuid: str
    model_id: str
    actor_role: str
    request_source: str
    environment: str
    session_id: str | None
    user_id: str | None = None


class GetJdSimilarityInput(BaseModel):
    job_description_id: str = Field(..., min_length=1)
    candidate_profile_id: str = Field(..., min_length=1)
    top_k: int = Field(..., ge=1, le=50)
    include_explanation: bool = False

    model_config = ConfigDict(extra="forbid")


class JdSimilarityMatch(BaseModel):
    section: str
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str

    model_config = ConfigDict(extra="forbid")


class GetJdSimilarityOutput(BaseModel):
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    matches: list[JdSimilarityMatch]
    generated_at: str

    model_config = ConfigDict(extra="forbid")


class FetchRecentHistoryInput(BaseModel):
    user_id: str = Field(..., min_length=1)
    window_hours: int = Field(..., ge=1, le=720)
    max_items: int = Field(..., ge=1, le=200)
    include_internal_notes: bool = False

    model_config = ConfigDict(extra="forbid")


class RecentEvent(BaseModel):
    timestamp: str
    actor: str
    action: str
    metadata: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class FetchRecentHistoryOutput(BaseModel):
    events: list[RecentEvent]
    cursor: str | None

    model_config = ConfigDict(extra="forbid")


class GetCompanyProfileInput(BaseModel):
    domain: str = Field(..., min_length=1)
    include_owners: bool = True
    include_hierarchy: bool = False

    model_config = ConfigDict(extra="forbid")


class CompanyProfile(BaseModel):
    name: str
    size: str | None
    industry: str | None
    location: str | None

    model_config = ConfigDict(extra="forbid")


class OwnerProfile(BaseModel):
    name: str
    role: str

    model_config = ConfigDict(extra="forbid")


class GetCompanyProfileOutput(BaseModel):
    domain: str
    profile: CompanyProfile
    owners: list[OwnerProfile] | None
    updated_at: str
    source: str

    model_config = ConfigDict(extra="forbid")


class SuggestNextConstraintInput(BaseModel):
    case_id: str = Field(..., min_length=1)
    candidate_state: dict[str, Any]
    prior_steps: list[str]
    risk_level: RiskLevel

    model_config = ConfigDict(extra="forbid")


class ConstraintSuggestion(BaseModel):
    constraint_key: str
    reason: str
    priority: RiskLevel
    expires_at: str

    model_config = ConfigDict(extra="forbid")


class SuggestNextConstraintOutput(BaseModel):
    suggestion: ConstraintSuggestion
    alternatives: list[ConstraintSuggestion]

    model_config = ConfigDict(extra="forbid")


ToolInputModel = (
    GetJdSimilarityInput
    | FetchRecentHistoryInput
    | GetCompanyProfileInput
    | SuggestNextConstraintInput
)
ToolOutputModel = (
    GetJdSimilarityOutput
    | FetchRecentHistoryOutput
    | GetCompanyProfileOutput
    | SuggestNextConstraintOutput
)
ToolHandler = Callable[[ToolExecutionContext, ToolInputModel, Session], ToolOutputModel]


ALLOWED_TOOL_ROLES: dict[str, dict[str, dict[str, set[str]]]] = {
    "get_jd_similarity": {
        "dev": {
            "api": {"agent", "admin", "system"},
            "model": {"agent", "admin", "system"},
            "internal": {"agent", "admin", "system"},
        },
        "staging": {
            "api": {"admin"},
            "model": {"agent", "admin", "system"},
            "internal": {"admin", "system"},
        },
        "prod": {
            "api": {"admin"},
            "model": {"admin", "system"},
            "internal": {"admin", "system"},
        },
    },
    "fetch_recent_history": {
        "dev": {
            "api": {"agent", "admin", "system", "auditor"},
            "model": {"agent", "admin", "system"},
            "internal": {"agent", "admin", "system", "auditor"},
        },
        "staging": {
            "api": {"admin", "auditor"},
            "model": {"agent", "admin", "system"},
            "internal": {"admin", "system", "auditor"},
        },
        "prod": {
            "api": {"admin", "auditor"},
            "model": {"admin", "system"},
            "internal": {"admin", "system", "auditor"},
        },
    },
    "get_company_profile": {
        "dev": {
            "api": {"agent", "admin", "system"},
            "model": {"agent", "admin", "system"},
            "internal": {"agent", "admin", "system"},
        },
        "staging": {
            "api": {"agent", "admin"},
            "model": {"agent", "admin", "system"},
            "internal": {"admin", "system"},
        },
        "prod": {
            "api": {"admin"},
            "model": {"admin", "system"},
            "internal": {"admin", "system"},
        },
    },
    "suggest_next_constraint": {
        "dev": {
            "api": {"agent", "admin", "system"},
            "model": {"agent", "admin", "system"},
            "internal": {"agent", "admin", "system"},
        },
        "staging": {
            "api": {"admin"},
            "model": {"agent", "admin", "system"},
            "internal": {"admin", "system"},
        },
        "prod": {
            "api": {"admin"},
            "model": {"admin", "system"},
            "internal": {"admin", "system"},
        },
    },
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _deterministic_timestamp(seed: str) -> str:
    offset_minutes = int(_hash_payload(seed)[:16], 16) % 31536000
    marker = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)
    return marker.isoformat().replace("+00:00", "Z")


def _coerce_input_hash(tool_name: str, arguments: Mapping[str, Any]) -> str:
    payload = {"tool": tool_name, "arguments": arguments}
    return _hash_payload(payload)


def _coerce_output_hash(tool_name: str, output: Mapping[str, Any]) -> str:
    payload = {"tool": tool_name, "output": output}
    return _hash_payload(payload)


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (value or "").lower())


def _to_serializable_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_stable_json(value))


def _extract_json_candidates(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    matches: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        start = text.find("{", start)
        if start == -1:
            break

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if current == "\\":
                    escaped = True
                    continue
                if current == "\"":
                    in_string = False
                continue
            if current == "\"":
                in_string = True
                continue
            if current == "{":
                depth += 1
                continue
            if current == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        start = index + 1
                        break
                    if isinstance(parsed, dict):
                        matches.append(parsed)
                    start = index + 1
                    break
        else:
            break

    return matches


def _pick_tool_from_prompt_payload(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    raw_tool_name = (
        payload.get("tool")
        or payload.get("tool_name")
        or payload.get("name")
        or payload.get("function")
        or payload.get("fn")
    )
    if not isinstance(raw_tool_name, str):
        raise ToolSchemaError(
            "prompt.tool_missing",
            "Prompt tool-call payload must contain `tool`, `tool_name`, `name`, `function`, or `fn`.",
            "Return tool-call JSON as: {\"tool\": \"get_jd_similarity\", \"arguments\": {...}}",
            status_code=400,
        )

    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ToolSchemaError(
            "prompt.arguments_missing",
            "tool-call payload must include an arguments object.",
            "Send tool inputs under `arguments`.",
            status_code=400,
        )

    return raw_tool_name, arguments


def _resolve_tool_input(tool_name: str, raw_arguments: Mapping[str, Any]) -> ToolInputModel:
    try:
        if tool_name == "get_jd_similarity":
            return GetJdSimilarityInput(**raw_arguments)
        if tool_name == "fetch_recent_history":
            return FetchRecentHistoryInput(**raw_arguments)
        if tool_name == "get_company_profile":
            return GetCompanyProfileInput(**raw_arguments)
        if tool_name == "suggest_next_constraint":
            return SuggestNextConstraintInput(**raw_arguments)
        raise ToolExecutionError(
            "tool.unsupported",
            f"Tool '{tool_name}' is not in the safe tool registry.",
            "Only approved tools can be executed. Use a supported allowlisted tool name.",
            status_code=400,
        )
    except ValidationError as exc:
        raise ToolSchemaError(
            "tool.schema_error",
            f"Tool '{tool_name}' arguments failed schema validation.",
            "Fix argument names/types/ranges and retry.",
            status_code=422,
        ) from exc


def _authorize(context: ToolExecutionContext, tool_name: str) -> None:
    tool_name = tool_name.strip()
    environment = context.environment.lower()
    source = context.request_source.lower()
    role = context.actor_role.lower()

    allowed_for_tool = ALLOWED_TOOL_ROLES.get(tool_name)
    if not allowed_for_tool:
        raise ToolAuthorizationError(
            "tool.unknown",
            f"Tool '{tool_name}' is not recognized in allowlist.",
            "Use a supported allowlisted tool and a recognized name.",
            status_code=404,
        )

    allowed_for_env = allowed_for_tool.get(environment)
    if not allowed_for_env:
        raise ToolAuthorizationError(
            "authorization.environment_blocked",
            f"Environment '{environment}' does not allow tool '{tool_name}'.",
            "Use a supported environment or change LOCAL_CONTEXT_ENVIRONMENT in configuration.",
            status_code=403,
        )

    allowed_for_source = allowed_for_env.get(source)
    if not allowed_for_source:
        raise ToolAuthorizationError(
            "authorization.source_blocked",
            f"Source '{source}' is not allowed for tool '{tool_name}' in environment '{environment}'.",
            "Route through an allowlisted source.",
            status_code=403,
        )

    if role not in allowed_for_source:
        raise ToolAuthorizationError(
            "authorization.role_blocked",
            f"Role '{role}' is not allowed to execute '{tool_name}'.",
            "Switch to an allowlisted actor role before re-running.",
            status_code=403,
        )


def _safe_job_id(value: str) -> int | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.isdigit():
        return None
    return int(stripped)


def _build_similarity_output(job_text: str, cv_text: str, top_k: int, include_explanation: bool) -> GetJdSimilarityOutput:
    job_terms = _tokenize(job_text)
    cv_terms = _tokenize(cv_text)
    job_set = set(job_terms)
    cv_set = set(cv_terms)

    section_buckets: list[tuple[str, list[str]]] = [
        ("core_skills", _tokenize(" ".join(job_terms[:400]))),
        ("experience", _tokenize(" ".join(job_terms[-400:]))),
        ("education", _tokenize("education experience certification project")),
        ("responsibilities", _tokenize("responsibility ownership impact")),
    ]

    matches: list[JdSimilarityMatch] = []
    for section, section_terms in section_buckets:
        section_set = set(section_terms)
        if not section_set or not cv_set:
            score = 0.0
        else:
            shared = len(section_set.intersection(cv_set))
            union = len(section_set.union(cv_set))
            score = float(shared / union) if union else 0.0
        if section_set and job_set:
            shared_terms = sorted(section_set.intersection(cv_set))
            rationale = ", ".join(shared_terms[:6]) if shared_terms else "No direct overlap found."
        else:
            rationale = "No matching section text to compare."
        if not include_explanation:
            rationale = rationale.split("No direct overlap found.")[0] or "Computed deterministically from lexical overlap."
        matches.append(
            JdSimilarityMatch(
                section=section,
                score=round(min(max(score, 0.0), 1.0), 4),
                rationale=rationale,
            )
        )

    matches.sort(key=lambda item: (item.score, item.section), reverse=True)
    top_matches = matches[: max(1, min(top_k, len(matches)))]
    effective_scores = [match.score for match in top_matches]
    average_score = round(sum(effective_scores) / max(len(effective_scores), 1), 6)

    return GetJdSimilarityOutput(
        similarity_score=average_score,
        matches=_to_serializable_dict([match.model_dump() for match in top_matches]),
        generated_at=_deterministic_timestamp(job_text + "|" + cv_text + "|v1"),
    )


def _get_jd_similarity(
    context: ToolExecutionContext,
    arguments: GetJdSimilarityInput,
    db: Session,
) -> GetJdSimilarityOutput:
    job_id = _safe_job_id(arguments.job_description_id)
    profile_id = _safe_job_id(arguments.candidate_profile_id)

    if job_id is not None:
        job = db.query(Job).filter(Job.id == job_id).first()
        job_description = getattr(job, "description", "")
    else:
        job = None
        job_description = ""

    if profile_id is not None:
        cv = db.query(CV).filter(CV.id == profile_id).first()
        profile_text = getattr(cv, "parsed_text", "")
    else:
        profile = db.query(CV).order_by(CV.id.desc()).first()
        profile_text = getattr(profile, "parsed_text", "") if profile else ""

    if not job and not profile_text:
        return GetJdSimilarityOutput(
            similarity_score=0.0,
            matches=[
                {
                    "section": "fallback",
                    "score": 0.0,
                    "rationale": "No retrievable JD or candidate profile data for deterministic local computation.",
                }
            ],
            generated_at=_deterministic_timestamp(_hash_payload(arguments.model_dump())),
        )

    job_title = getattr(job, "title", "")
    return _build_similarity_output(
        f"{job_title}\n{job_description}",
        profile_text,
        arguments.top_k,
        arguments.include_explanation,
    )


def _fetch_recent_history(
    context: ToolExecutionContext,
    arguments: FetchRecentHistoryInput,
    db: Session,
) -> FetchRecentHistoryOutput:
    window_start = datetime.now(timezone.utc) - timedelta(hours=arguments.window_hours)
    rows = (
        db.query(ScoreHistory)
        .filter(
            ScoreHistory.created_at >= window_start,
            ScoreHistory.created_at.isnot(None),
        )
        .order_by(ScoreHistory.created_at.desc())
        .limit(arguments.max_items)
        .all()
    )

    events: list[RecentEvent] = []
    for item in rows:
        metadata = {
            "fit_score": float(item.fit_score),
            "job_title": item.job_title,
            "company": item.company,
            "llm_provider": item.llm_provider,
        }
        if arguments.include_internal_notes:
            metadata["history_id"] = item.id
            metadata["raw_description"] = item.job_description[:256] if item.job_description else ""
        events.append(
            RecentEvent(
                timestamp=item.created_at.isoformat().replace("+00:00", "Z") if item.created_at else _deterministic_timestamp(str(item.id)),
                actor=str(arguments.user_id),
                action="score_jd",
                metadata=metadata,
            )
        )

    last_event = str(rows[-1].id) if rows else None
    return FetchRecentHistoryOutput(events=events, cursor=last_event)


def _owners_from_domain(domain: str, count: int = 2) -> list[dict[str, str]]:
    token = _hash_payload(domain)
    titles = ["Hiring Lead", "Recruiter", "Engineering Manager"]
    surnames = ["Sato", "Reyes", "Zhang", "Kim", "Novak", "Patel", "Klein", "Rossi"]
    given = ["Ari", "Casey", "Jordan", "Rowan", "Morgan", "Taylor", "Sky", "Kai"]

    output: list[dict[str, str]] = []
    for index in range(count):
        name = f"{given[int(token[index * 4:(index + 1) * 4], 16) % len(given)]} {surnames[int(token[(index + 2) * 4:(index + 3) * 4], 16) % len(surnames)]}"
        role = titles[int(token[index * 3:(index * 3) + 3], 16) % len(titles)]
        output.append({"name": name, "role": role})
    return output


def _get_company_profile(
    context: ToolExecutionContext,
    arguments: GetCompanyProfileInput,
    db: Session,
) -> GetCompanyProfileOutput:
    normalized = arguments.domain.strip().lower().rstrip(".")
    if not normalized:
        raise ToolSchemaError(
            "tool.domain_invalid",
            "domain must be a non-empty value.",
            "Provide a valid domain string like example.com.",
            status_code=422,
        )

    parsed = normalized.replace("https://", "").replace("http://", "")
    parsed = parsed.lstrip("www.").strip("/")
    domain_parts = parsed.split("/")
    domain = domain_parts[0] if domain_parts else parsed

    try:
        candidate = db.query(Job).filter(
            func.lower(func.coalesce(Job.company_website, "")).contains(domain)
        ).first()
    except Exception:
        candidate = None

    if candidate is not None:
        company_address = candidate.company_address
        if not isinstance(company_address, dict):
            company_address = {}
        profile = CompanyProfile(
            name=candidate.company or domain,
            size=str(candidate.company_employees_count) if candidate.company_employees_count else None,
            industry=(candidate.industries or None),
            location=str((company_address or {}).get("city") or candidate.company_address or None),
        )
    else:
        index = int(_hash_payload(domain)[:10], 16) % 4
        names = [
            domain.split(".")[0].replace("-", " ").title(),
            f"{domain.split('.')[0].title()} Labs",
            f"{domain.split('.')[0].title()} Group",
            f"{domain.split('.')[0].title()} Technologies",
        ]
        industries = ["Software", "Finance", "Health", "Professional Services", "AI Infrastructure"]
        locations = ["Remote", "Boston, MA", "London, UK", "Singapore", "Berlin, DE"]
        profile = CompanyProfile(
            name=names[index],
            size=str(((int(_hash_payload(domain)[:12], 16) % 3_000) + 50)),
            industry=industries[index % len(industries)],
            location=locations[index % len(locations)],
        )
    owners = _owners_from_domain(domain, count=1) if arguments.include_owners else None
    if arguments.include_hierarchy and owners is not None:
        owners = [
            {"name": owner["name"], "role": f"{owner['role']} (hierarchy-aware)"}
            for owner in owners
        ]

    source = "local_profile_registry_v1"
    return GetCompanyProfileOutput(
        domain=domain,
        profile=profile,
        owners=[OwnerProfile(**item) for item in owners] if owners else None,
        updated_at=_deterministic_timestamp(domain + "|" + str(arguments.include_owners)),
        source=source,
    )


def _pick_constraints(seed: int, risk: RiskLevel) -> list[ConstraintSuggestion]:
    templates = [
        ("restrict_outcomes", "Constrain output format and avoid verbosity.", "medium"),
        ("require_examples", "Require concrete examples in the answer.", "low"),
        ("add_scale_constraint", "Add a dataset or input-size stress condition.", "high"),
        ("insert_edge_case", "Inject an edge-case guard or failure mode.", "high"),
        ("prioritize_depth", "Shift prompts toward deeper reasoning depth.", "medium"),
        ("minimize_ambiguity", "Reduce ambiguity with explicit constraints.", "low"),
        ("add_validation", "Add validation criteria and acceptance thresholds.", "medium"),
    ]
    ordered = sorted(range(len(templates)), key=lambda index: (templates[index][2], index))
    if risk == "high":
        ordered = sorted(ordered, key=lambda i: (0 if templates[i][2] == "high" else 1, i))
    elif risk == "medium":
        ordered = sorted(ordered, key=lambda i: (0 if templates[i][2] == "medium" else 1, i))
    else:
        ordered = sorted(ordered, key=lambda i: (0 if templates[i][2] == "low" else 1, i))

    ordered = [ordered[(seed + idx) % len(ordered)] for idx in range(len(ordered))]

    output: list[ConstraintSuggestion] = []
    for index in ordered:
        key, reason, priority = templates[index]
        output.append(
            ConstraintSuggestion(
                constraint_key=key,
                reason=reason,
                priority=priority,  # type: ignore[arg-type]
                expires_at=_deterministic_timestamp(str(index) + "|" + str(seed)),
            )
        )
    return output


def _suggest_next_constraint(
    context: ToolExecutionContext,
    arguments: SuggestNextConstraintInput,
    db: Session,
) -> SuggestNextConstraintOutput:
    state_signature = _hash_payload(_to_serializable_dict(arguments.candidate_state))
    prior_signature = _hash_payload(arguments.prior_steps)
    risk_signature = _hash_payload(arguments.risk_level)
    suggestion_seed = int(f"{state_signature}{prior_signature}{risk_signature}"[:16], 16)

    ranked = _pick_constraints(suggestion_seed, arguments.risk_level)
    if not ranked:
        raise ToolExecutionError(
            "tool.suggest_failed",
            "No suggestions were available for this request.",
            "Retry with explicit candidate_state and prior_steps.",
            status_code=422,
        )

    return SuggestNextConstraintOutput(
        suggestion=ranked[0],
        alternatives=ranked[1: min(4, len(ranked))],
    )


def _coerce_tool_output(tool_name: str, output: ToolOutputModel) -> dict[str, Any]:
    if not isinstance(output, (GetJdSimilarityOutput, FetchRecentHistoryOutput, GetCompanyProfileOutput, SuggestNextConstraintOutput)):
        raise ToolExecutionError(
            "tool.output_type_invalid",
            f"Tool '{tool_name}' returned an unexpected result object.",
            "Contact backend maintainers; output schema is not compliant.",
            status_code=500,
        )
    return _to_serializable_dict(output.model_dump())


TOOL_REGISTRY: dict[str, tuple[type[BaseModel], ToolHandler]] = {
    "get_jd_similarity": (GetJdSimilarityInput, _get_jd_similarity),
    "fetch_recent_history": (FetchRecentHistoryInput, _fetch_recent_history),
    "get_company_profile": (GetCompanyProfileInput, _get_company_profile),
    "suggest_next_constraint": (SuggestNextConstraintInput, _suggest_next_constraint),
}


def _record_audit(
    db: Session,
    context: ToolExecutionContext,
    tool_name: str,
    input_payload: Mapping[str, Any],
    output_payload: Mapping[str, Any] | None,
    input_hash: str,
    result_hash: str,
    latency_ms: int,
    status: str,
    decision_rationale: str,
    error_message: str | None = None,
) -> int | None:
    try:
        entry = LocalContextToolCallAudit(
            request_uuid=context.request_uuid,
            tool_name=tool_name,
            model_id=context.model_id,
            actor_role=context.actor_role,
            environment=context.environment,
            request_source=context.request_source,
            session_id=context.session_id,
            user_id=context.user_id,
            input_hash=input_hash,
            result_hash=result_hash,
            latency_ms=latency_ms,
            status=status,
            decision_rationale={"reason": decision_rationale},
            error_message=error_message,
            input_payload=_to_serializable_dict(dict(input_payload)),
            output_payload=_to_serializable_dict(dict(output_payload or {})),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return int(entry.id)
    except Exception:
        db.rollback()
        return None


@dataclass(frozen=True)
class ToolInvocationResult:
    tool_name: str
    output: dict[str, Any]
    input_hash: str
    result_hash: str
    latency_ms: int
    decision_rationale: str
    audit_id: int | None = None


def execute_tool(
    *,
    db: Session,
    context: ToolExecutionContext,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> ToolInvocationResult:
    normalized_tool = tool_name.strip()
    if not normalized_tool:
        raise ToolSchemaError(
            "tool.empty",
            "tool_name is required.",
            "Send a non-empty tool name from the approved registry.",
            status_code=400,
        )

    _authorize(context, normalized_tool)
    resolved = _resolve_tool_input(normalized_tool, arguments)

    input_hash = _coerce_input_hash(normalized_tool, resolved.model_dump())
    start = time.perf_counter()
    try:
        _, handler = TOOL_REGISTRY[normalized_tool]
    except KeyError as exc:
        raise ToolExecutionError(
            "tool.unsupported",
            f"Tool '{normalized_tool}' is not in the dispatch registry.",
            "Use one of: get_jd_similarity, fetch_recent_history, get_company_profile, suggest_next_constraint.",
            status_code=400,
        ) from exc

    try:
        output = handler(context, resolved, db)
        if isinstance(output, BaseModel):
            payload = _coerce_tool_output(normalized_tool, output)
        else:
            raise ToolExecutionError(
                "tool.output_type_invalid",
                "Handler returned an invalid output payload.",
                "Contact backend maintainers.",
                status_code=500,
            )
        result_hash = _coerce_output_hash(normalized_tool, payload)
        latency_ms = int((time.perf_counter() - start) * 1000)
        rationale = f"Executed {normalized_tool} via allowlisted registry"
        audit_id = _record_audit(
            db=db,
            context=context,
            tool_name=normalized_tool,
            input_payload=resolved.model_dump(),
            output_payload=payload,
            input_hash=input_hash,
            result_hash=result_hash,
            latency_ms=latency_ms,
            status="allowed",
            decision_rationale=rationale,
        )
        return ToolInvocationResult(
            tool_name=normalized_tool,
            output=payload,
            input_hash=input_hash,
            result_hash=result_hash,
            latency_ms=latency_ms,
            decision_rationale=rationale,
            audit_id=audit_id,
        )
    except ToolDispatchError:
        raise
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        result_payload = {"error": str(exc)}
        result_hash = _coerce_output_hash(normalized_tool, result_payload)
        rationale = f"Execution failed while running {normalized_tool}"
        _record_audit(
            db=db,
            context=context,
            tool_name=normalized_tool,
            input_payload=resolved.model_dump(),
            output_payload=result_payload,
            input_hash=input_hash,
            result_hash=result_hash,
            latency_ms=latency_ms,
            status="error",
            decision_rationale=rationale,
            error_message=str(exc),
        )
        raise ToolExecutionError(
            "tool.execution_error",
            f"Failed to execute tool '{normalized_tool}'.",
            "Review tool inputs and try again.",
            status_code=500,
        ) from exc


def execute_from_prompt(
    *,
    db: Session,
    context: ToolExecutionContext,
    prompt: str,
) -> ToolInvocationResult:
    candidates = _extract_json_candidates(prompt)
    for candidate in candidates:
        try:
            tool_name, arguments = _pick_tool_from_prompt_payload(candidate)
            return execute_tool(db=db, context=context, tool_name=tool_name, arguments=arguments)
        except ToolDispatchError:
            raise
        except Exception:
            continue

    raise ToolSchemaError(
        "prompt.tool_not_found",
        "No supported tool-call JSON payload found in prompt text.",
        "Use format: {\"tool\":\"suggest_next_constraint\",\"arguments\":{...}}",
        status_code=400,
    )

