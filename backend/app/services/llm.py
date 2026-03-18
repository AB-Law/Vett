"""LiteLLM-backed LLM service with multi-provider support."""
import logging
import json
import re
import asyncio
import os
from html import unescape
from urllib.parse import urlparse
from typing import Any
from sqlalchemy.orm import Session
from ..config import get_settings
from .interview_docs import fetch_context_from_interview_documents

logger = logging.getLogger(__name__)

_LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
_LLM_SEMAPHORE: asyncio.Semaphore | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = asyncio.Semaphore(_LLM_CONCURRENCY)
    return _LLM_SEMAPHORE


def _get_litellm_model() -> tuple[str, dict[str, Any]]:
    """Return (model_string, extra_kwargs) for the active provider."""
    s = get_settings()
    p = s.active_llm_provider.lower()

    if p == "claude":
        return s.claude_model, {"api_key": s.anthropic_api_key}

    elif p == "openai":
        return s.openai_model, {"api_key": s.openai_api_key}

    elif p == "azure_openai":
        return (
            f"azure/{s.azure_openai_deployment}",
            {
                "api_key": s.azure_openai_api_key,
                "api_base": s.azure_openai_endpoint,
                "api_version": s.azure_openai_api_version,
            },
        )

    elif p == "ollama":
        return (
            f"ollama/{s.ollama_model}",
            {"api_base": s.ollama_base_url},
        )

    elif p == "lm_studio":
        # LiteLLM OpenAI-compatible: model must NOT contain slashes in the prefix.
        # Strip any org/namespace prefix (e.g. "google/gemma-3-4b" → use as-is but
        # route through the openai provider via api_base).
        model_id = s.lm_studio_model
        # Ensure the base URL ends without a trailing slash
        base_url = s.lm_studio_base_url.rstrip("/")
        return (
            f"openai/{model_id}",
            {
                "api_base": base_url,
                "api_key": "lm-studio",
            },
        )

    raise ValueError(f"Unknown LLM provider: {p}")


def _get_litellm_embedding_model() -> tuple[str, dict[str, Any]]:
    """Return (model_string, extra_kwargs) for embedding calls."""
    s = get_settings()
    provider = s.active_llm_provider.lower()
    embedding_model = s.practice_embedding_model

    if provider == "openai":
        return embedding_model, {"api_key": s.openai_api_key}

    if provider == "azure_openai":
        return (
            f"azure/{s.azure_openai_deployment}",
            {
                "api_key": s.azure_openai_api_key,
                "api_base": s.azure_openai_endpoint,
                "api_version": s.azure_openai_api_version,
            },
        )

    if provider == "ollama":
        return (
            f"ollama/{embedding_model}",
            {"api_base": s.ollama_base_url},
        )

    if provider == "lm_studio":
        return (
            f"openai/{embedding_model}",
            {
                "api_base": s.lm_studio_base_url.rstrip("/"),
                "api_key": "lm-studio",
            },
        )

    if provider == "claude":
        # Anthropic does not expose a standard embeddings API in this stack.
        return embedding_model, {"api_key": s.openai_api_key}

    raise ValueError(f"Unknown LLM provider: {provider}")


def _extract_embedding(response: Any) -> list[float]:
    """Extract an embedding vector from a LiteLLM response."""
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        raise ValueError("Empty embedding response")

    first = data[0]
    if hasattr(first, "embedding"):
        vector = first.embedding
    elif isinstance(first, dict):
        vector = first.get("embedding")
    else:
        raise ValueError("Unexpected embedding response shape")

    return [float(item) for item in vector]



SCORE_PROMPT = """You are an expert career coach and resume reviewer.

Given the following CV and job description, analyse how well the CV matches the job.

Return ONLY valid JSON with this exact structure:
{{
  "score_payload_version": "evidence-grounded-v1",
  "fit_score": <integer 0-100>,
  "matched_keywords": [<list of strings>],
  "missing_keywords": [<list of strings>],
  "gap_analysis": "<2-4 sentence narrative>",
  "rewrite_suggestions": [<list of 3-5 specific suggestion strings>],
  "matched_keyword_evidence": [
    {{
      "value": "<matched keyword>",
      "cv_citations": [
        {{
          "section_id": "<stable section id>",
          "line_start": <1-based int>,
          "line_end": <1-based int>,
          "snippet": "<short 1-2 line snippet>"
        }}
      ],
      "jd_phrase_citations": [
        {{
          "phrase_id": "<stable jd phrase id>",
          "line_start": <1-based int>,
          "line_end": <1-based int>,
          "snippet": "<short 1-2 line snippet>"
        }}
      ],
      "evidence_missing_reason": null | "<why evidence was not available>"
    }}
  ],
  "missing_keyword_evidence": [
    {{
      "value": "<missing keyword>",
      "cv_citations": [
        {{
          "section_id": "<stable section id>",
          "line_start": <1-based int>,
          "line_end": <1-based int>,
          "snippet": "<short 1-2 line snippet>"
        }}
      ],
      "jd_phrase_citations": [
        {{
          "phrase_id": "<stable jd phrase id>",
          "line_start": <1-based int>,
          "line_end": <1-based int>,
          "snippet": "<short 1-2 line snippet>"
        }}
      ],
      "evidence_missing_reason": "<required if evidence is not available>"
    }}
  ],
  "rewrite_suggestion_evidence": [
    {{
      "value": "<rewrite suggestion>",
      "cv_citations": [
        {{
          "section_id": "<stable section id>",
          "line_start": <1-based int>,
          "line_end": <1-based int>,
          "snippet": "<short 1-2 line snippet>"
        }}
      ],
      "jd_phrase_citations": [
        {{
          "phrase_id": "<stable jd phrase id>",
          "line_start": <1-based int>,
          "line_end": <1-based int>,
          "snippet": "<short 1-2 line snippet>"
        }}
      ],
      "evidence_missing_reason": null | "<why evidence was not available>"
    }}
  ]
}}

CV:
{cv_text}

Job Description:
{jd_text}

Role/Signal map:
{role_signal_map}
"""

SCORE_FAST_PROMPT = """You are a resume screening assistant.

Given the CV and job description below, return a quick fit assessment.

Return ONLY valid JSON with this exact structure:
{{
  "fit_score": <integer 0-100>,
  "matched_keywords": [<up to 8 strings>],
  "missing_keywords": [<up to 8 strings>],
  "gap_analysis": "<1-2 sentence summary>"
}}

CV:
{cv_text}

Job Description:
{jd_text}
"""

CV_REWRITE_PROPOSAL_PROMPT = """
You are a conservative CV rewrite agent supporting local-first manual review.

Input:
CV:
{cv_text}

Job Description:
{jd_text}

Role/Signal map:
{role_signal_map}

Score context:
{score_snapshot}

Return ONLY valid JSON with this exact structure:
{{
  "proposals": [
    {{
      "before": "<short, exact or near-exact snippet from the CV, 1-2 lines>",
      "after": "<proposed replacement snippet, same scope/intent>",
      "reason": "<why this improves JD alignment>",
      "risk_or_uncertainty": "<risk or uncertainty about correctness / overclaim>"
    }}
  ]
}}

Guidelines:
- Keep snippets grounded in the CV style and do not invent dates, companies, or achievements.
- Provide exactly 3-6 proposals unless evidence is weak.
- Do not include any markdown, bullets, or extra prose outside JSON.
"""


ROLE_SIGNAL_MAP_PROMPT = """You are an extraction agent for career-role matching.

Input:
CV:
{cv_text}

Job description:
{jd_text}

Return ONLY valid JSON using this exact structure:
{{
  "role_summary": "<one-line summary of the likely target role>",
  "role_tier": "<senior|mid|lead|principal|director|manager|unknown>",
  "responsibilities": ["<important duty 1>", "<important duty 2>"],
  "required_skills": ["<high-confidence required skill>", "..."],
  "secondary_skills": ["<helpful but optional skill>", "..."],
  "experience_signals": ["<years, domain, stack, delivery scale signals>", "..."],
  "interview_focus": ["<topic likely to appear in interview>", "..."],
  "evidence_snippets": ["<short JD signal 1>", "..."]
}}

Keep each list to ~2-5 high-value entries.
"""


AGENT_PLAN_PROMPT = """You are a career agent that converts a scoring result into an action plan.

Input:
Role/Signal map:
{role_signal_map}

Score result:
{score_payload}

CV:
{cv_text}

Job description:
{jd_text}

Return ONLY valid JSON using this exact structure:
{{
  "role_signal_map": {{}},
  "skills_to_fix_first": ["<most important skill to improve>", "..."],
  "concrete_edit_actions": ["<specific change to CV or prep>", "..."],
  "interview_topics_to_prioritize": ["<topic name>", "..."],
  "study_order": ["<order of study for next 1-2 weeks>", "..."]
}}

Rules:
- Skills must be ordered by priority (earliest item highest priority).
- Interview topics should be distinct and role-relevant.
- Provide study order as an explicit sequence, not generic labels.
- Keep every list actionable and concrete.
"""


def _coerce_signal_map(value: object) -> dict[str, str | list[str]]:
    """Normalise role map structure to safe, JSON-serializable values."""
    if not isinstance(value, dict):
        return {}

    output: dict[str, str | list[str]] = {}
    for raw_key, raw_value in value.items():
        safe_key = str(raw_key).strip()
        if not safe_key:
            continue
        if isinstance(raw_value, list):
            values = _coerce_string_list(raw_value)
            if values:
                output[safe_key] = values
            continue
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if text:
                output[safe_key] = text
            continue
        if raw_value is None:
            continue
        output[safe_key] = str(raw_value).strip()
    return output


def _coerce_list_block(value: object) -> list[str]:
    """Coerce to a bounded list of actionable bullets."""
    return _coerce_string_list(value)[:8]


def _coerce_int(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_evidence_snippet(value: object, max_chars: int = 220) -> str:
    raw = _coerce_non_empty_string(value)
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return raw.strip()[:max_chars]
    return " | ".join(lines[:2])[:max_chars]


def _coerce_line_range(value: object) -> tuple[int, int]:
    values = value if isinstance(value, dict) else {}
    line_start = _coerce_int(values.get("start"), 1)
    line_end = _coerce_int(values.get("end"), line_start)
    if line_start < 1:
        line_start = 1
    if line_end < line_start:
        line_end = line_start
    return line_start, line_end


def _coerce_cv_citation(value: object, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    section_id = _coerce_non_empty_string(value.get("section_id")) or _coerce_non_empty_string(value.get("id"))
    if not section_id:
        section_id = f"cv_section_{index + 1}"
    line_start = _coerce_int(value.get("line_start"), index + 1)
    line_end = _coerce_int(value.get("line_end"), line_start)
    if isinstance(value.get("line_range"), dict):
        line_start, line_end = _coerce_line_range(value.get("line_range"))
    if line_start < 1:
        line_start = 1
    if line_end < line_start:
        line_end = line_start
    return {
        "section_id": section_id,
        "line_start": line_start,
        "line_end": line_end,
        "snippet": _normalize_evidence_snippet(value.get("snippet")),
    }


def _coerce_jd_citation(value: object, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    phrase_id = _coerce_non_empty_string(value.get("phrase_id")) or _coerce_non_empty_string(value.get("id"))
    if not phrase_id:
        phrase_id = f"jd_phrase_{index + 1}"
    line_start = _coerce_int(value.get("line_start"), index + 1)
    line_end = _coerce_int(value.get("line_end"), line_start)
    if isinstance(value.get("line_range"), dict):
        line_start, line_end = _coerce_line_range(value.get("line_range"))
    if line_start < 1:
        line_start = 1
    if line_end < line_start:
        line_end = line_start
    return {
        "phrase_id": phrase_id,
        "phrase": _coerce_non_empty_string(value.get("phrase")) or "",
        "line_start": line_start,
        "line_end": line_end,
        "snippet": _normalize_evidence_snippet(value.get("snippet")),
    }


def _coerce_citation_list(values: object, kind: str, index_offset: int = 0) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    output: list[dict[str, Any]] = []
    for offset, value in enumerate(values):
        if kind == "cv":
            citation = _coerce_cv_citation(value, index_offset + offset)
        else:
            citation = _coerce_jd_citation(value, index_offset + offset)
        if citation is None:
            continue
        if _normalize_evidence_snippet(citation.get("snippet")):
            output.append(citation)
    return output


def _coerce_score_evidence_record(value: object, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    entry_value = _coerce_non_empty_string(
        value.get("value")
        or value.get("keyword")
        or value.get("suggestion")
        or value.get("item")
    )
    if not entry_value:
        return None
    record = {
        "value": entry_value,
        "cv_citations": _coerce_citation_list(
            value.get("cv_citations"),
            kind="cv",
            index_offset=index,
        ),
        "jd_phrase_citations": _coerce_citation_list(
            value.get("jd_phrase_citations"),
            kind="jd",
            index_offset=index,
        ),
        "evidence_missing_reason": _coerce_non_empty_string(value.get("evidence_missing_reason")),
    }
    if not record["cv_citations"] and not record["jd_phrase_citations"] and not record["evidence_missing_reason"]:
        record["evidence_missing_reason"] = (
            f"Evidence could not be attached for \"{entry_value}\". "
            "Re-run scoring with citation capture enabled."
        )
    return record


def _coerce_score_evidence_rows(items: object, evidence_values: object, *, item_label: str) -> list[dict[str, Any]]:
    item_values = _coerce_string_list(items)
    raw_records: list[dict[str, Any]] = []
    evidence_list = evidence_values if isinstance(evidence_values, list) else []
    for index, raw_record in enumerate(evidence_list):
        try:
            parsed = _coerce_score_evidence_record(raw_record, index)
        except Exception:
            logger.exception(
                "Malformed %s evidence row at index=%s (raw=%r).",
                item_label,
                index,
                raw_record,
            )
            parsed = None
        if parsed is not None:
            raw_records.append(parsed)

    records_by_value: dict[str, list[dict[str, Any]]] = {}
    for record in raw_records:
        key = _coerce_non_empty_string(record.get("value")).strip().lower()
        if not key:
            logger.warning(
                "Evidence row missing fallback value in %s evidence block, skipping row=%r",
                item_label,
                record,
            )
            continue
        records_by_value.setdefault(key, []).append(record)

    output: list[dict[str, Any]] = []
    for item in item_values:
        normalized_item = item.strip()
        if not normalized_item:
            continue
        normalized_key = normalized_item.lower()
        row = records_by_value.get(normalized_key, []).pop(0) if normalized_key in records_by_value else None
        if row:
            row["value"] = normalized_item
            output.append(row)
            continue
        output.append(
            {
                "value": normalized_item,
                "cv_citations": [],
                "jd_phrase_citations": [],
                "evidence_missing_reason": (
                    f"No evidence row was returned for \"{normalized_item}\" in the {item_label} block."
                ),
            }
        )
    return output


def _coerce_score_payload(score_payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact payload for downstream prompts."""
    matched_keyword_evidence = _coerce_score_evidence_rows(
        score_payload.get("matched_keywords"),
        score_payload.get("matched_keyword_evidence"),
        item_label="matched keyword",
    )
    missing_keyword_evidence = _coerce_score_evidence_rows(
        score_payload.get("missing_keywords"),
        score_payload.get("missing_keyword_evidence"),
        item_label="missing keyword",
    )
    rewrite_suggestion_evidence = _coerce_score_evidence_rows(
        score_payload.get("rewrite_suggestions"),
        score_payload.get("rewrite_suggestion_evidence"),
        item_label="rewrite suggestion",
    )
    fit_score = _coerce_int(score_payload.get("fit_score"), 0)
    if fit_score < 0:
        fit_score = 0
    if fit_score > 100:
        fit_score = 100
    return {
        "score_payload_version": _coerce_non_empty_string(score_payload.get("score_payload_version")) or "v1",
        "fit_score": fit_score,
        "matched_keywords": _coerce_list_block(score_payload.get("matched_keywords")),
        "missing_keywords": _coerce_list_block(score_payload.get("missing_keywords")),
        "gap_analysis": _coerce_non_empty_string(score_payload.get("gap_analysis")) or "",
        "rewrite_suggestions": _coerce_list_block(score_payload.get("rewrite_suggestions")),
        "matched_keyword_evidence": matched_keyword_evidence,
        "missing_keyword_evidence": missing_keyword_evidence,
        "rewrite_suggestion_evidence": rewrite_suggestion_evidence,
    }


def _coerce_agent_plan_payload(
    value: object,
    fallback_role_signal_map: dict[str, Any],
) -> dict[str, object]:
    """Build a complete agent-plan payload and enforce output shape."""
    if not isinstance(value, dict):
        return {
            "role_signal_map": fallback_role_signal_map,
            "skills_to_fix_first": [],
            "concrete_edit_actions": [],
            "interview_topics_to_prioritize": [],
            "study_order": [],
        }

    role_signal_map = _coerce_signal_map(
        value.get("role_signal_map")
        or value.get("role_map")
        or value.get("role_signal")
        or fallback_role_signal_map
    )

    skills_to_fix_first = _coerce_list_block(
        value.get("skills_to_fix_first")
        or value.get("priority_skills")
        or value.get("priority_skill_gaps")
        or value.get("skills_to_improve_first")
    )
    interview_topics = _coerce_list_block(
        value.get("interview_topics_to_prioritize")
        or value.get("interview_topics")
        or value.get("interview_focus")
    )
    study_order = _coerce_list_block(value.get("study_order") or value.get("study_path") or value.get("learning_order"))
    concrete_edit_actions = _coerce_list_block(
        value.get("concrete_edit_actions")
        or value.get("concrete_actions")
        or value.get("edit_actions")
    )

    return {
        "role_signal_map": role_signal_map,
        "skills_to_fix_first": skills_to_fix_first,
        "concrete_edit_actions": concrete_edit_actions,
        "interview_topics_to_prioritize": interview_topics,
        "study_order": study_order,
    }


def _coerce_rewrite_proposals(
    value: object,
    score_payload: dict[str, Any] | None,
    role_signal_map: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return []

    proposal_list = value.get("proposals")
    if not isinstance(proposal_list, list):
        return []

    proposals = []
    for item in proposal_list:
        if not isinstance(item, dict):
            continue
        before = _coerce_non_empty_string(item.get("before"))
        after = _coerce_non_empty_string(item.get("after"))
        if not before or not after:
            continue
        reason = _coerce_non_empty_string(item.get("reason")) or "Adds a focused alignment edit for CV reviewability."
        risk = _coerce_non_empty_string(item.get("risk_or_uncertainty")) or _coerce_non_empty_string(item.get("uncertainty")) or "Moderate confidence due to ambiguity or parser variance."
        proposals.append(
            {
                "before": before,
                "after": after,
                "reason": reason,
                "risk_or_uncertainty": risk,
            }
        )

    if proposals:
        return proposals

    fallback_missing = _coerce_string_list((score_payload or {}).get("missing_keywords")) if isinstance(score_payload, dict) else []
    summary = _coerce_non_empty_string((role_signal_map or {}).get("role_summary"))
    fallback_prefix = summary or "targeted role requirements"

    if not fallback_missing:
        return [
            {
                "before": "Current CV text is already the only source for safe edits.",
                "after": "Add one concrete, evidence-backed bullet that mirrors a high-priority requirement from the job description.",
                "reason": f"The model did not return structured rewrite candidates; this keeps updates constrained to verifiable sections.",
                "risk_or_uncertainty": f"Fallback is generic for {fallback_prefix}. Review carefully against the original CV data.",
            }
        ]

    return [
        {
            "before": f"Gap area for \"{missing}\" currently not visible.",
            "after": f"Add one concise CV bullet demonstrating quantified ownership in \"{missing}\".",
            "reason": f"Job description emphasis indicates this signal is likely expected for {fallback_prefix}.",
            "risk_or_uncertainty": "Risk of overclaim if this skill is not directly supported in CV evidence.",
        }
        for missing in fallback_missing[:4]
    ]


async def extract_role_signal_map(cv_text: str, jd_text: str) -> dict[str, str | list[str]]:
    """Extract role and signal metadata for downstream scoring and planning."""
    import litellm

    model, kwargs = _get_litellm_model()
    prompt = ROLE_SIGNAL_MAP_PROMPT.format(
        cv_text=cv_text[:6000],
        jd_text=jd_text[:4000],
    )

    async with _get_llm_semaphore():
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
            **kwargs,
        )
    raw = response.choices[0].message.content.strip()
    match = _extract_first_json_object(raw)
    if match:
        raw = match

    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            parsed = {}

    return _coerce_signal_map(
        parsed
        if isinstance(parsed, dict)
        else {
            "role_summary": "Unable to parse role/signal map output.",
            "role_tier": "unknown",
        }
    )


async def score_cv_against_jd(
    cv_text: str,
    jd_text: str,
    role_signal_map: dict[str, Any] | None = None,
) -> dict:
    import litellm

    model, kwargs = _get_litellm_model()
    safe_role_signal_map = _coerce_signal_map(role_signal_map)
    role_signal_text = (
        json.dumps(safe_role_signal_map, ensure_ascii=False)
        if safe_role_signal_map
        else "No structured role/signal map available."
    )
    prompt = SCORE_PROMPT.format(
        cv_text=cv_text[:6000],
        jd_text=jd_text[:4000],
        role_signal_map=role_signal_text,
    )

    async with _get_llm_semaphore():
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            **kwargs,
        )

    raw = response.choices[0].message.content.strip()

    # Extract JSON even if wrapped in markdown code block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    return _coerce_score_payload(data)


async def score_cv_fast(cv_text: str, jd_text: str) -> dict:
    """Single-call fast scoring for bulk operations. No role analysis, no critic."""
    import litellm

    model, kwargs = _get_litellm_model()
    prompt = SCORE_FAST_PROMPT.format(
        cv_text=cv_text[:4000],
        jd_text=jd_text[:3000],
    )
    async with _get_llm_semaphore():
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            **kwargs,
        )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    return _coerce_score_payload(data)


async def generate_cv_rewrite_proposals(
    cv_text: str,
    jd_text: str,
    role_signal_map: dict[str, Any] | None = None,
    score_payload: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    import litellm

    model, kwargs = _get_litellm_model()
    safe_map = _coerce_signal_map(role_signal_map)
    safe_score = _coerce_score_payload(score_payload or {})

    prompt = CV_REWRITE_PROPOSAL_PROMPT.format(
        cv_text=cv_text[:7000],
        jd_text=jd_text[:4500],
        role_signal_map=json.dumps(safe_map, ensure_ascii=False) if safe_map else "No role/signal map available.",
        score_snapshot=json.dumps(safe_score, ensure_ascii=False),
    )

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        max_tokens=1200,
        **kwargs,
    )

    raw = response.choices[0].message.content.strip()
    match = _extract_first_json_object(raw)
    if match:
        raw = match

    parsed: object
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            parsed = {}

    return _coerce_rewrite_proposals(parsed, score_payload=score_payload or {}, role_signal_map=safe_map)


async def generate_agent_score_plan(
    cv_text: str,
    jd_text: str,
    score_payload: dict[str, Any],
    role_signal_map: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate a second-pass local plan for skill improvements and interview prep."""
    import litellm

    model, kwargs = _get_litellm_model()
    safe_map = _coerce_signal_map(role_signal_map)
    map_text = json.dumps(safe_map, ensure_ascii=False) if safe_map else "No role/signal map available."
    score_text = json.dumps(_coerce_score_payload(score_payload), ensure_ascii=False)

    async with _get_llm_semaphore():
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": AGENT_PLAN_PROMPT.format(
                cv_text=cv_text[:6000],
                jd_text=jd_text[:4000],
                role_signal_map=map_text,
                score_payload=score_text,
            )}],
            temperature=0.25,
            max_tokens=1200,
            **kwargs,
        )

    raw = response.choices[0].message.content.strip()
    match = _extract_first_json_object(raw)
    if match:
        raw = match

    parsed: object
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            parsed = {}

    return _coerce_agent_plan_payload(parsed, safe_map)


async def test_connection() -> dict:
    """Quick connectivity test – sends a minimal prompt."""
    import litellm

    model, kwargs = _get_litellm_model()
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
            **kwargs,
        )
        reply = response.choices[0].message.content.strip()
        return {"ok": True, "reply": reply}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for text using the active provider."""
    import litellm

    model, kwargs = _get_litellm_embedding_model()
    response = await litellm.aembedding(model=model, input=[text], **kwargs)
    return _extract_embedding(response)


FOLLOWUP_PROMPT = """
You are an interview coach that rewrites a base question into a constrained follow-up prompt.
Language policy:
- If `language` is provided in constraints, align the prompt to that language.
- If `language` is not provided, keep the prompt language-agnostic and avoid naming any concrete programming language or language-specific APIs.

Input:
Base Question:
- title: {title}
- url: {url}
- difficulty: {difficulty}
- accepted rate: {acceptance}
- asked frequency: {frequency}
- base company: {company}

Constraints requested:
{constraints}
Language constraint:
{language_policy}

Return exactly one JSON object and nothing else.
Use plain text only in `transformed_prompt`.
Do not use LaTeX escapes like "\\(" or "\\)".
Do not include markdown fences or explanatory prose.
The `transformed_prompt` must be a new follow-up variant, not a restatement of the original statement.
Output schema:
{{
  "base_question_link": "same as base url",
  "transformed_prompt": "question prompt text (short, 1-3 paragraphs)",
  "constraint_metadata": {{
    "difficulty_delta": <int|null>,
    "language": "<string|null>",
    "technique": "<string|null>",
    "complexity": "<string|null>",
    "time_pressure_minutes": <int|null>,
    "pattern": "<string|null>"
  }},
  "reason": "short reason for prompt relevance"
}}

Few-shot style examples:

Example 1
Input
title: Two Sum
constraints: language=python, difficulty_delta=1, pattern=two-sum, time_pressure_minutes=20
Output
{{
  "base_question_link": "https://leetcode.com/problems/two-sum",
  "transformed_prompt": "You are given an array `nums` and a target integer. Return two indices of values that sum to the target. Add a twist: enforce stricter variable naming and one additional edge case. Provide a solution approach only, not the full code.",
  "constraint_metadata": {{
    "difficulty_delta": 1,
    "language": "python",
    "technique": null,
    "complexity": null,
    "time_pressure_minutes": 20,
    "pattern": "two-sum"
  }},
  "reason": "Raises difficulty by forcing stricter implementation discipline and a deeper edge-case check."
}}

Example 2
Input
title: Valid Parentheses
constraints: language=python, difficulty_delta=0
Output
{{
  "base_question_link": "https://leetcode.com/problems/valid-parentheses",
  "transformed_prompt": "Determine whether a string of parentheses is valid under the classic bracket matching rules. Add a requirement to explain one optimization point and a minimal proof idea after the algorithm description.",
  "constraint_metadata": {{
    "difficulty_delta": 0,
    "language": "python",
    "technique": null,
    "complexity": null,
    "time_pressure_minutes": null,
    "pattern": null
  }},
  "reason": "Maintains core structure while adding a short proof and implementation rigor under the same target language."
}}

Example 3
Input
title: Two Sum
constraints: pattern=10 million records
Output
{{
  "base_question_link": "https://leetcode.com/problems/two-sum",
  "transformed_prompt": "You are designing an analytics service that receives up to 10 million numbers per batch. Given an integer array `nums` and a target integer, return two distinct indices of elements that sum to the target while describing how your approach remains efficient at this scale.",
  "constraint_metadata": {{
    "difficulty_delta": null,
    "language": null,
    "technique": null,
    "complexity": null,
    "time_pressure_minutes": null,
    "pattern": "10 million records"
  }},
  "reason": "Adds a realistic scale constraint so the solution needs to address high-throughput behavior."
}}

Hard requirements:
- Always produce a transformed prompt that adds at least one concrete new constraint, scenario, or performance requirement (e.g., scale, memory cap, language rule, or data ordering rule).
- Do not rephrase the original prompt unchanged.
- Start with a fresh constraint/variant line before the task if helpful.
- Keep the core concept intact, but make the question clearly distinct from the base form.
- If no language is provided, do not mention JavaScript, Python, Java, TypeScript, C++, Rust, Go, Ruby, or any other language names in `transformed_prompt`.
"""

INTERVIEW_CHAT_PROMPT = """
You are a technical interviewer running a mock interview.
Be encouraging, practical, and strict. Ask one meaningful probing question or hint.
Do not provide complete code unless explicitly requested as a scaffold.

Base question:
- title: {title}
- url: {url}
- difficulty: {difficulty}
- company: {company}
- acceptance: {acceptance}
- frequency: {frequency}

Candidate message:
{message}

Recent conversation:
{conversation}

Current candidate draft:
{draft_solution}

Response instructions:
- Return plain text only.
- Do not include JSON or markdown.
- Ask only one follow-up question or one concise hint.
"""

ADAPTIVE_INTERVIEW_QUESTION_PROMPT = """
You are a senior technical interviewer.
Generate exactly one high-quality interview question in polished English.

Return strict JSON only with this schema:
{{
  "question": "single interview question text",
  "rationale": "1-2 sentence reason for choosing this question now"
}}

Hard requirements:
- The question must be grammatically correct and natural.
- Do NOT include bullet lists, markdown, or multiple questions.
- Keep it concise: 1-3 sentences.
- Respect category: {category}
- Role: {role}
- Company: {company}
- Performance signal: {performance_signal}
- Dev focus: {dev_focus}
- Avoid repeating prior questions or close paraphrases from this list:
{asked_questions}
- Use this context when relevant:
  - Job requirements: {job_requirements}
  - CV signals: {cv_signals}
  - Interview memory: {memory_facts}
  - Retrieved context snippets: {context_snippets}
"""

REVIEW_AND_VARIANT_PROMPT = """
You are a senior engineering interviewer reviewing a candidate answer.
Be direct and specific. Return strict JSON only.

Base question:
- title: {title}
- url: {url}
- company: {company}
- difficulty: {difficulty}
- acceptance: {acceptance}
- frequency: {frequency}

Candidate language: {language}

Submitted solution:
{solution}

Return one JSON object only.
{{
  "review_summary": "2-4 sentence overall review on correctness, complexity, tradeoffs",
  "strengths": ["strength 1", "strength 2"],
  "concerns": ["concern 1", "concern 2"],
  "follow_up_prompt": "A concrete and distinct variant prompt to validate the approach",
  "follow_up_reason": "Short reason for why this variant is a good robustness test"
}}
"""


def _normalize_followup_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return value


def _coerce_constraint_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _coerce_non_empty_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _coerce_test_case_items(value: object, max_cases: int = 8) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    cases: list[dict[str, str]] = []
    for item in value[:max_cases]:
        if not isinstance(item, dict):
            continue
        name = _coerce_non_empty_string(item.get("name")) or f"Case {len(cases) + 1}"
        input_value = _coerce_non_empty_string(item.get("input"))
        output_value = _coerce_non_empty_string(item.get("expected_output"))
        if not input_value or not output_value:
            continue
        rationale = _coerce_non_empty_string(item.get("rationale")) or "Covers edge or edge-adjacent behavior."
        cases.append(
            {
                "name": name,
                "input": input_value,
                "expected_output": output_value,
                "rationale": rationale,
            }
        )
    return cases


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [text.strip() for text in [str(item).strip() for item in value] if text.strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _contains_followup_twist(prompt: str, constraints: dict[str, object]) -> bool:
    normalized = " ".join(prompt.lower().split())
    if any(token in normalized for token in ("suppose", "assume", "consider", "what if", "how would", "scale", "handle", "optimize", "design", "under", "without", "without using", "must", "edge case", "constraint", "time complexity", "space complexity")):
        return True
    pattern = _coerce_constraint_value(constraints.get("pattern"))
    technique = _coerce_constraint_value(constraints.get("technique"))
    complexity = _coerce_constraint_value(constraints.get("complexity"))
    language = _coerce_constraint_value(constraints.get("language"))
    for token in (pattern, technique, complexity, language):
        if token and token.lower() in normalized:
            return True
    return False


def _build_followup_twist(constraints: dict[str, object], difficulty_delta: int | None) -> str:
    pattern = _coerce_constraint_value(constraints.get("pattern"))
    if pattern:
        return f" Add a concrete constraint: {pattern}."

    language = _coerce_constraint_value(constraints.get("language"))
    if language:
        return f" Ask for a solution outline that is written only in {language}."

    technique = _coerce_constraint_value(constraints.get("technique"))
    if technique:
        return f" Add a requirement to apply {technique} in the approach."

    complexity = _coerce_constraint_value(constraints.get("complexity"))
    if complexity:
        return f" Enforce a complexity target of {complexity} where possible."

    time_pressure = constraints.get("time_pressure_minutes")
    if time_pressure is not None:
        return f" Ask for a solution that is explicitly designed for a {time_pressure}-minute interview window."

    if difficulty_delta in (-1, -2):
        return " Make it simpler by adding one strong guardrail and a smaller ambiguity surface."
    if difficulty_delta in (1, 2):
        return " Increase difficulty by requiring an implementation detail that changes behavior under high volume."
    return " Reframe this as a production-oriented follow-up: assume this runs on up to 10 million records in one run and explain how to keep it efficient at that scale."


def _coerce_transformed_prompt(value: object) -> str:
    if not isinstance(value, str):
        return ""

    prompt = value.strip()
    if not prompt:
        return ""

    try:
        nested = json.loads(prompt)
    except Exception:
        try:
            nested = json.loads(_sanitize_json_token_stream(prompt))
        except Exception:
            return value

    if not isinstance(nested, dict):
        return value

    inner_prompt = nested.get("transformed_prompt")
    if isinstance(inner_prompt, str) and inner_prompt.strip():
        return inner_prompt.strip()
    return value


def _sanitize_json_token_stream(payload: str) -> str:
    # Some model payloads include sequences like "\\(" or "\\)" that are not valid
    # JSON escapes, so escape only unsupported backslash sequences.
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", payload)


def _extract_first_json_object(payload: str) -> str | None:
    cursor = 0
    length = len(payload)
    while cursor < length and payload[cursor] != "{":
        cursor += 1
    if cursor >= length:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(cursor, length):
        char = payload[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return payload[cursor:index + 1]
    return None


def _extract_leetcode_slug(source_url: str) -> str | None:
    normalized = (source_url or "").strip()
    if not normalized:
        return None
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    if "leetcode" not in host:
        return None
    path = parsed.path or ""
    match = re.search(r"/problems/([^/\\?]+)", path)
    if not match:
        return None
    slug = match.group(1).strip("/")
    return slug or None


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""

    sanitized = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.I)
    sanitized = re.sub(r"</\s*(p|div|li|ul|ol|pre|code|h[1-6])\s*>", "\n", sanitized, flags=re.I)
    sanitized = re.sub(r"<[^>]+>", "", sanitized)
    sanitized = unescape(sanitized)
    sanitized = re.sub(r"\r\n", "\n", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    sanitized = re.sub(r"[ \t]+", " ", sanitized)
    return sanitized.strip()


async def _fetch_leetcode_problem_statement(source_url: str) -> str | None:
    """Fetch and clean the LeetCode problem statement for a given URL."""
    slug = _extract_leetcode_slug(source_url)
    if not slug:
        return None

    try:
        import httpx
    except Exception:
        return None

    query = """query problemDetails($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        title
        content
        difficulty
      }
    }"""
    payload = {
        "query": query,
        "variables": {"titleSlug": slug},
    }
    headers = {
        "User-Agent": "Vett/1.0 (contact for details)",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
            response = await client.post("https://leetcode.com/graphql", json=payload)
    except Exception:
        return None

    if response.status_code < 200 or response.status_code >= 300:
        return None

    try:
        data = response.json()
    except Exception:
        return None

    question = data.get("data", {}).get("question")
    if not isinstance(question, dict):
        return None

    content = _coerce_non_empty_string(question.get("content"))
    if not content:
        return None

    text = _strip_html_to_text(content)
    return text[:7000] if len(text) > 7000 else text


def _coerce_review_payload(value: dict[str, object]) -> dict[str, object]:
    return {
        "review_summary": _coerce_non_empty_string(value.get("review_summary")) or "Could not parse a clear review summary.",
        "strengths": _coerce_string_list(value.get("strengths")),
        "concerns": _coerce_string_list(value.get("concerns")),
        "follow_up_prompt": _coerce_non_empty_string(value.get("follow_up_prompt"))
        or "Keep the same core idea and add one realistic scale or edge-case constraint.",
        "follow_up_reason": _coerce_non_empty_string(value.get("follow_up_reason"))
        or "Validate whether the approach handles added robustness constraints.",
        "llm_model": str(value.get("llm_model") or ""),
        "llm_provider": str(value.get("llm_provider") or ""),
    }


def _history_lines(history: list[dict[str, str]]) -> str:
    if not history:
        return "- no prior messages"
    lines = []
    for item in history[-12:]:
        role = str(item.get("role", "user")).strip().lower()
        if role not in {"user", "assistant", "interviewer"}:
            role = "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "- no prior messages"


TEST_CASE_PROMPT = """
You are a senior engineer preparing platform-quality test cases for coding interviews.

Return ONLY valid JSON in this exact shape:
{{
  "test_cases": [
    {{
      "name": "short case name",
      "input": "exact input expression or object used for validation",
      "expected_output": "expected output or return value",
      "rationale": "why this case is important"
    }}
  ]
}}

Constraints:
- Use language-independent notation for inputs/outputs if exact language syntax differs.
- Keep cases concrete and minimal but meaningful.
- Include normal, edge, and failure-mode checks whenever possible.
- Do not include markdown fences or prose.

Problem context:
- Title: {title}
- URL/Source: {url}
- Company: {company}
- Difficulty: {difficulty}
- Acceptance: {acceptance}
- Frequency: {frequency}
- Preferred language: {language}

Examples:
Example 1
Input
- Title: Two Sum
- Difficulty: easy
- Acceptance: solved with O(n) method preferred
- Frequency: medium
- Preferred language: python
Output
{{
  "test_cases": [
    {{
      "name": "Simple pair",
      "input": "([2, 7, 11, 15], 9)",
      "expected_output": "[0, 1]",
      "rationale": "Checks base behavior with valid pair at start."
    }},
    {{
      "name": "No valid pair",
      "input": "([1, 2, 3, 4], 100)",
      "expected_output": "[]",
      "rationale": "Validates graceful handling when no match exists."
    }},
    {{
      "name": "Duplicates",
      "input": "([3, 3], 6)",
      "expected_output": "[0, 1]",
      "rationale": "Ensures duplicate values are handled correctly."
    }}
  ]
}}

Example 2
Input
- Title: Valid Parentheses
- Difficulty: medium
- Acceptance: return true/false only
- Frequency: high
- Preferred language: python
Output
{{
  "test_cases": [
    {{
      "name": "Standard valid case",
      "input": "'()[]{{}}'",
      "expected_output": "True",
      "rationale": "Validates standard happy path."
    }},
    {{
      "name": "Broken order",
      "input": "'([)]'",
      "expected_output": "False",
      "rationale": "Catches improper nesting behavior."
    }},
    {{
      "name": "Long valid pattern",
      "input": "'({{[()[]]{{}}})'",
      "expected_output": "True",
      "rationale": "Stresses nested mixes of all bracket types."
    }}
  ]
}}

Create 6-8 test cases.
"""


CODE_TEMPLATE_PROMPT = """
You are a senior engineering coach generating a practical starter template for interview-style coding problems.

Return ONLY valid JSON in this exact shape:
{{
  "language": "python",
  "signature": "def solve(self, nums, target)",
  "template": "class Solution:\\n    def solve(self, nums, target):\\n        # TODO: implement\\n        raise NotImplementedError\\n",
  "notes": "Brief suggestion about where to place parsing/return logic.",
  "problem_prompt": "A LeetCode-style problem statement in 3-5 concise lines."
}}

Important:
- Keep the template realistic for a competitive-coding platform style answer.
- Provide a class-based signature where possible (e.g., `class Solution` + a single solution method).
- The `template` must be valid Python code and directly runnable.

Examples:
Example 1
Input
- Title: Two Sum
- Difficulty: easy
- Acceptance: return list of two indices
- Language: python
Output
{{
  "language": "python",
  "signature": "def solve(self, nums, target)",
  "template": "class Solution:\\n    def solve(self, nums, target):\\n        # Return indices of two numbers that sum to target.\\n        seen = {{}}\\n        for index, value in enumerate(nums):\\n            need = target - value\\n            if need in seen:\\n                return [seen[need], index]\\n            seen[value] = index\\n        return []\\n",
  "notes": "LeetCode-style scaffold for Two Sum with fast hash-map lookup.",
  "problem_prompt": "Find two indices in nums whose values sum to target."
}}

Example 2
Input
- Title: Valid Parentheses
- Difficulty: medium
- Acceptance: return bool for balanced input
- Language: python
Output
{{
  "language": "python",
  "signature": "def solve(self, s)",
  "template": "class Solution:\\n    def solve(self, s):\\n        # Return True if parentheses are valid, else False.\\n        pairs = {{')': '(', ']': '[', '}': '{{'}}\\n        stack = []\\n        for char in s:\\n            if char in '{{([[':\\n                stack.append(char)\\n                continue\\n            if char not in pairs:\\n                continue\\n            if not stack or stack[-1] != pairs[char]:\\n                return False\\n            stack.pop()\\n        return not stack\\n",
  "notes": "Template uses a stack so users can complete logic quickly.",
  "problem_prompt": "Given a string s, return whether parentheses are valid."
}}

Input
- Title: {title}
- Difficulty: {difficulty}
- Acceptance: {acceptance}
- Frequency: {frequency}
- Language: {language}
Output
{{}}
"""


async def simulate_interviewer_chat(
    *,
    title: str,
    url: str,
    company: str,
    difficulty: str | None,
    acceptance: str | None,
    frequency: str | None,
    message: str,
    language: str | None,
    conversation: list[dict[str, str]] | None = None,
    solution_text: str | None = None,
    db: Session | None = None,
    job_id: int | None = None,
) -> str:
    import litellm

    model, kwargs = _get_litellm_model()
    safe_message = message.strip()
    if not safe_message:
        return "Please share a specific question about the problem or your approach."

    document_snippets: list[str] = []
    if db is not None:
        try:
            query_vector = await generate_embedding(safe_message)
            snippets = fetch_context_from_interview_documents(
                db=db,
                query_vector=query_vector,
                job_id=job_id,
            )
            for snippet in snippets[:3]:
                filename = snippet.get("filename", "interview-doc")
                owner_type = snippet.get("owner_type", "global")
                text = snippet.get("snippet", "").strip()
                if not text:
                    continue
                document_snippets.append(
                    f"- [{owner_type}] {filename}: {text}"
                )
        except Exception as exc:
            logger.debug("Interview doc retrieval failed: %s", exc)

    context_appendix = ""
    if document_snippets:
        context_appendix = (
            "\n\nRelevant interview document snippets:\n"
            + "\n".join(document_snippets)
        )

    prompt = INTERVIEW_CHAT_PROMPT.format(
        title=title,
        url=url,
        company=company,
        difficulty=difficulty or "unknown",
        acceptance=acceptance or "unknown",
        frequency=frequency or "unknown",
        message=safe_message,
        draft_solution=_coerce_non_empty_string(solution_text) or "No draft attached yet.",
        conversation=_history_lines(conversation or []),
    ) + context_appendix

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            **kwargs,
        )
        return response.choices[0].message.content.strip() or "Tell me which part feels unclear, and we can narrow it down."
    except Exception:
        return "Tell me your approach, and I can challenge one assumption behind it."


async def generate_adaptive_interview_question(
    *,
    role: str,
    company: str,
    category: str,
    performance_signal: str,
    dev_focus: bool,
    asked_questions: list[str],
    job_requirements: list[str],
    cv_signals: list[str],
    memory_facts: list[str],
    context_snippets: list[str] | None = None,
) -> dict[str, str]:
    try:
        import litellm
    except Exception as exc:
        logger.warning("Adaptive interview question generation unavailable (litellm import failed): %s", exc)
        return {"question": "", "rationale": ""}

    model, kwargs = _get_litellm_model()
    prompt = ADAPTIVE_INTERVIEW_QUESTION_PROMPT.format(
        role=role or "this role",
        company=company or "this company",
        category=category or "technical",
        performance_signal=performance_signal or "unknown",
        dev_focus="yes" if dev_focus else "no",
        asked_questions=json.dumps(asked_questions[-40:], ensure_ascii=True),
        job_requirements=json.dumps(job_requirements[:20], ensure_ascii=True),
        cv_signals=json.dumps(cv_signals[:20], ensure_ascii=True),
        memory_facts=json.dumps(memory_facts[-20:], ensure_ascii=True),
        context_snippets=json.dumps((context_snippets or [])[:8], ensure_ascii=True),
    )
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=320,
            **kwargs,
        )
        raw = _coerce_non_empty_string(response.choices[0].message.content)
    except Exception as exc:
        logger.warning("Adaptive interview question generation failed: %s", exc)
        return {"question": "", "rationale": ""}

    if not raw:
        return {"question": "", "rationale": ""}

    match = _extract_first_json_object(raw)
    candidate = match or raw
    parsed: dict[str, object] = {}
    try:
        parsed = json.loads(candidate)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(candidate))
        except Exception:
            return {"question": "", "rationale": ""}

    question = _coerce_non_empty_string(parsed.get("question"))
    rationale = _coerce_non_empty_string(parsed.get("rationale"))
    if not question:
        return {"question": "", "rationale": ""}
    if len(question) < 18:
        return {"question": "", "rationale": ""}
    return {"question": question, "rationale": rationale}


async def review_solution_with_variant(
    *,
    title: str,
    url: str,
    company: str,
    difficulty: str | None,
    acceptance: str | None,
    frequency: str | None,
    solution_text: str,
    language: str | None,
    difficulty_delta: int | None = None,
) -> dict[str, object]:
    import litellm

    model, kwargs = _get_litellm_model()
    settings = get_settings()
    safe_solution = solution_text.strip()
    if not safe_solution:
        return {
            "review_summary": "No solution was provided. Provide a solution draft before review.",
            "strengths": [],
            "concerns": ["Solution text was empty."],
            "follow_up_prompt": "Add one realistic scale constraint and edge-case checklist to the base question.",
            "follow_up_reason": "Review was requested before a complete submission.",
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }

    prompt = REVIEW_AND_VARIANT_PROMPT.format(
        title=title,
        url=url,
        company=company,
        difficulty=difficulty or "any",
        acceptance=acceptance or "unknown",
        frequency=frequency or "unknown",
        language=language or "any",
        solution=safe_solution,
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1400,
            **kwargs,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        fallback = {
            "review_summary": (
                f"Could not complete automated review for '{title}'. Please resubmit your solution."
            ),
            "strengths": ["A solution was submitted for review."],
            "concerns": ["Review service returned an error."],
            "follow_up_prompt": (
                "Keep the same core problem and add one production-scale or edge-case constraint. "
                "Explain how this changes your implementation decisions."
            ),
            "follow_up_reason": f"Fallback due to review service error: {exc}",
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }
        return fallback

    if not raw:
        fallback = {
            "review_summary": f"Could not produce a review for '{title}'.",
            "strengths": ["A solution draft was supplied."],
            "concerns": ["No review content was returned."],
            "follow_up_prompt": (
                "Add a high-volume edge case to the original problem and ask for optimized behavior."
            ),
            "follow_up_reason": "LLM returned empty output.",
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }
        return fallback

    match = _extract_first_json_object(raw)
    if match:
        raw = match

    parsed: dict[str, object]
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            parsed = {
                "review_summary": _coerce_non_empty_string(raw) or f"Review for '{title}' was not parseable.",
                "strengths": [],
                "concerns": ["Could not parse structured review output."],
                "follow_up_prompt": (
                    "Take the same problem and add one missing edge case in input range, then explain handling."
                ),
                "follow_up_reason": "Parser fallback due to non-JSON model output.",
                "llm_model": model,
                "llm_provider": settings.active_llm_provider,
            }

    review_result = _coerce_review_payload(parsed)

    if difficulty_delta is not None:
        variant_hint = _build_followup_twist({"difficulty_delta": difficulty_delta}, difficulty_delta)
        if variant_hint not in str(review_result["follow_up_prompt"]):
            review_result["follow_up_prompt"] = (
                f"{str(review_result['follow_up_prompt']).rstrip('. ')}.{variant_hint}"
            ).strip()

    review_result["llm_model"] = model
    review_result["llm_provider"] = settings.active_llm_provider
    return review_result


async def generate_constrained_followup(
    *,
    title: str,
    url: str,
    company: str,
    difficulty: str | None,
    acceptance: str | None,
    frequency: str | None,
    constraints: dict[str, object],
) -> dict[str, object]:
    import litellm

    model, kwargs = _get_litellm_model()
    settings = get_settings()
    requested_language = _coerce_non_empty_string(constraints.get("language"))
    language_policy = (
        f"language: {requested_language}"
        if requested_language
        else "language: not specified (use language-agnostic prompt)"
    )
    constraint_block = "\n".join(
        [
            f"{key}: {value}"
            for key, value in constraints.items()
            if value is not None and f"{value}".strip()
        ]
    ) or "none"

    prompt = FOLLOWUP_PROMPT.format(
        title=title,
        url=url,
        company=company,
        difficulty=difficulty or "any",
        acceptance=acceptance or "unknown",
        frequency=frequency or "unknown",
        constraints=constraint_block,
        language_policy=language_policy,
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
            **kwargs,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        fallback = {
            "base_question_link": url,
            "transformed_prompt": (
                f"Take '{title}' and apply these constraints: {constraint_block}. "
                "Then solve a similar problem with the same idea, adapted complexity, and a fresh edge case."
            ),
            "constraint_metadata": constraints,
            "reason": f"Fallback due to LLM error: {exc}",
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }
        return fallback

    if not raw:
        return {
            "base_question_link": url,
            "transformed_prompt": (
                f"Rewrite '{title}' with constraints: {constraint_block}, keeping the core idea."
            ),
            "constraint_metadata": constraints,
            "reason": "No LLM response content was returned.",
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }

    match = _extract_first_json_object(raw)
    if match:
        raw = match
    parsed: dict[str, object]
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(raw))
        except Exception:
            coerced_prompt = _coerce_transformed_prompt(raw)
            reason_message = (
                "Output used a nested payload; extracted the transformed prompt."
                if coerced_prompt != raw
                else "Generated from base question with constraints."
            )
            parsed = {
                "base_question_link": url,
                "transformed_prompt": coerced_prompt,
                "constraint_metadata": constraints,
                "reason": reason_message,
                "llm_model": model,
                "llm_provider": settings.active_llm_provider,
            }
            return parsed

    if not isinstance(parsed.get("constraint_metadata"), dict):
        parsed["constraint_metadata"] = _normalize_followup_result(parsed.get("constraint_metadata")) or constraints
    parsed["transformed_prompt"] = _coerce_transformed_prompt(parsed.get("transformed_prompt")) or str(parsed.get("transformed_prompt", ""))
    difficulty_delta = (
        int(parsed.get("constraint_metadata", {}).get("difficulty_delta"))
        if isinstance(parsed.get("constraint_metadata"), dict) and parsed.get("constraint_metadata").get("difficulty_delta") is not None
        else None
    )
    if not _contains_followup_twist(parsed["transformed_prompt"], constraints):
        parsed["transformed_prompt"] = (
            f"{parsed['transformed_prompt'].strip()} {_build_followup_twist(constraints, difficulty_delta)}".strip()
        )

    parsed["base_question_link"] = parsed.get("base_question_link") or url
    parsed["llm_model"] = model
    parsed["llm_provider"] = settings.active_llm_provider
    if not isinstance(parsed.get("transformed_prompt"), str) or not parsed.get("transformed_prompt"):
        parsed["transformed_prompt"] = (
            f"Rewrite '{title}' into a constrained follow-up and enforce all constraints."
        )
    if not requested_language:
        parsed["transformed_prompt"] = re.sub(
            r"\b(?:javascript|python(?:\s*3)?|typescript|java(?!script)|c\+\+|c#|go|rust|ruby|scala|swift|kotlin|dart|php|perl)\b",
            "a language",
            str(parsed["transformed_prompt"]),
            flags=re.IGNORECASE,
        )
    if not isinstance(parsed.get("reason"), str) or not parsed.get("reason"):
        parsed["reason"] = "Generated from base question with provided constraints."
    return parsed


async def generate_practice_test_cases(
    *,
    title: str,
    url: str,
    company: str,
    difficulty: str | None,
    acceptance: str | None,
    frequency: str | None,
    language: str | None = None,
    count: int = 8,
) -> dict[str, object]:
    import litellm

    model, kwargs = _get_litellm_model()
    settings = get_settings()
    safe_count = max(6, min(10, count))
    safe_language = language or "python"
    safe_title = title.strip() if isinstance(title, str) else "Untitled problem"

    prompt = (
        TEST_CASE_PROMPT.replace("{title}", safe_title)
        .replace("{url}", url or "unknown")
        .replace("{company}", company)
        .replace("{difficulty}", difficulty or "unknown")
        .replace("{acceptance}", acceptance or "unknown")
        .replace("{frequency}", frequency or "unknown")
        .replace("{language}", safe_language)
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1400,
            **kwargs,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        return {
            "test_cases": [],
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
            "error": str(exc),
        }

    if not raw:
        return {
            "test_cases": [],
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
            "error": "Empty LLM response.",
        }

    extracted = _extract_first_json_object(raw) or raw
    parsed: dict[str, object]
    try:
        parsed = json.loads(extracted)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(extracted))
        except Exception:
            return {
                "test_cases": [],
                "llm_model": model,
                "llm_provider": settings.active_llm_provider,
                "error": "Could not parse test case JSON.",
            }

    candidate = parsed.get("test_cases") if isinstance(parsed, dict) else parsed
    if candidate is None:
        candidate = []
    cases = _coerce_test_case_items(candidate, max_cases=safe_count)
    return {
        "test_cases": cases,
        "llm_model": model,
        "llm_provider": settings.active_llm_provider,
    }


async def generate_solution_template(
    *,
    title: str,
    url: str,
    company: str,
    difficulty: str | None,
    acceptance: str | None,
    frequency: str | None,
    language: str | None = None,
    prompt: str | None = None,
) -> dict[str, object]:
    import litellm

    model, kwargs = _get_litellm_model()
    settings = get_settings()
    safe_language = language or "python3"
    safe_title = title.strip() if isinstance(title, str) else "Untitled problem"
    safe_prompt = prompt.strip() if isinstance(prompt, str) else ""
    resolved_prompt = safe_prompt
    if safe_language.lower() == "python3":
        safe_language = "python"

    if not resolved_prompt:
        fetched_prompt = await _fetch_leetcode_problem_statement(url)
        if fetched_prompt:
            resolved_prompt = fetched_prompt

    default_template = (
        "class Solution:\\n"
        "    def solve(self):\\n"
        "        # TODO: implement\\n"
        "        raise NotImplementedError\\n"
    )

    prompt_text = (
        CODE_TEMPLATE_PROMPT.replace("{title}", safe_title)
        .replace("{difficulty}", difficulty or "unknown")
        .replace("{acceptance}", acceptance or "unknown")
        .replace("{frequency}", frequency or "unknown")
        .replace("{language}", safe_language)
    )
    if resolved_prompt:
        prompt_text += f"\\nProblem statement/context:\\n{resolved_prompt}\\n"
    else:
        prompt_text += (
            "\\nProblem context:\\n"
            "No full statement is available. Generate a concise LeetCode-style problem statement from the title and metadata, "
            "then provide a practical Python scaffold.\\n"
        )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1,
            max_tokens=700,
            **kwargs,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        return {
            "language": safe_language,
            "signature": None,
            "template": default_template,
            "notes": str(exc),
            "problem_prompt": resolved_prompt or None,
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }

    if not raw:
        return {
            "language": safe_language,
            "signature": None,
            "template": default_template,
            "notes": "No LLM response content returned.",
            "problem_prompt": resolved_prompt or None,
            "llm_model": model,
            "llm_provider": settings.active_llm_provider,
        }

    parsed_candidate = _extract_first_json_object(raw) or raw
    parsed: dict[str, object]
    try:
        parsed = json.loads(parsed_candidate)
    except Exception:
        try:
            parsed = json.loads(_sanitize_json_token_stream(parsed_candidate))
        except Exception:
            return {
                "language": safe_language,
                "signature": None,
                "template": default_template,
                "notes": "Could not parse structured template output.",
                "problem_prompt": resolved_prompt or None,
                "llm_model": model,
                "llm_provider": settings.active_llm_provider,
            }

    template = _coerce_non_empty_string(parsed.get("template"))
    signature = _coerce_non_empty_string(parsed.get("signature")) or None
    notes = _coerce_non_empty_string(parsed.get("notes")) or None
    generated_language = _coerce_non_empty_string(parsed.get("language")) or safe_language
    emitted_problem_prompt = _coerce_non_empty_string(parsed.get("problem_prompt"))
    normalized_language = generated_language.lower()
    if normalized_language == "python3":
        normalized_language = "python"

    if not template:
        template = default_template

    return {
        "language": normalized_language,
        "signature": signature,
        "template": template,
        "notes": notes,
        "problem_prompt": emitted_problem_prompt or resolved_prompt or None,
        "llm_model": model,
        "llm_provider": settings.active_llm_provider,
    }
