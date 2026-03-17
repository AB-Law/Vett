from __future__ import annotations

from typing import Any
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..services.local_context_tooling import (
    ToolAuthorizationError,
    ToolDispatchError,
    ToolExecutionContext,
    ToolInvocationResult,
    ToolSchemaError,
    execute_from_prompt,
    execute_tool,
)


router = APIRouter(prefix="/local-context", tags=["local-context"])


class LocalContextInvocation(BaseModel):
    request_uuid: str | None = None
    model_id: str = Field(default="vett-local-context-v1", max_length=120)
    prompt: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    actor_role: str | None = None
    request_source: str = "api"
    session_id: str | None = None
    environment: str | None = None

    @model_validator(mode="after")
    def _must_supply_tool_or_prompt(self) -> "LocalContextInvocation":
        has_prompt = bool((self.prompt or "").strip())
        has_tool = bool((self.tool_name or "").strip())
        if not has_prompt and not has_tool:
            raise ValueError("Either prompt or tool_name must be provided.")
        if has_tool and self.tool_arguments is None:
            raise ValueError("tool_arguments is required when tool_name is provided.")
        if has_prompt and self.tool_arguments is not None:
            raise ValueError("tool_arguments can only be used with tool_name mode.")
        if has_tool and not isinstance(self.tool_arguments, dict):
            raise ValueError("tool_arguments must be a mapping.")
        return self


class LocalContextResponse(BaseModel):
    request_uuid: str
    tool: str
    status: str
    latency_ms: int
    decision_rationale: str
    input_hash: str
    result_hash: str
    audit_id: int | None = None
    output: dict[str, Any]


class _ErrorResponse(BaseModel):
    code: str
    message: str
    recovery_hint: str | None = None


def _build_context(
    *,
    request: LocalContextInvocation,
    header_actor_role: str | None,
    header_user_id: str | None,
    header_request_source: str | None,
) -> ToolExecutionContext:
    if not request.request_uuid:
        request_uuid = str(uuid.uuid4())
    else:
        request_uuid = request.request_uuid

    environment = (
        request.environment
        or get_settings().local_context_environment
    ).strip().lower()
    actor_role = (request.actor_role or header_actor_role or "agent").strip().lower()
    request_source = (
        request.request_source
        or header_request_source
        or "api"
    ).strip().lower()
    user_id = header_user_id or None

    return ToolExecutionContext(
        request_uuid=request_uuid,
        model_id=request.model_id.strip(),
        actor_role=actor_role,
        request_source=request_source,
        environment=environment,
        session_id=request.session_id,
        user_id=user_id,
    )


def _to_error(detail: ToolDispatchError) -> _ErrorResponse:
    return _ErrorResponse(
        code=detail.code,
        message=detail.message,
        recovery_hint=detail.recovery_hint,
    )


def _tool_args(request: LocalContextInvocation) -> dict[str, Any]:
    if not request.tool_arguments:
        return {}
    return request.tool_arguments


@router.post("/invoke", response_model=LocalContextResponse)
def invoke_tool(
    payload: LocalContextInvocation,
    db: Session = Depends(get_db),
    x_actor_role: str | None = Header(default=None, alias="x-actor-role"),
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    x_request_source: str | None = Header(default=None, alias="x-request-source"),
) -> LocalContextResponse:
    context = _build_context(
        request=payload,
        header_actor_role=x_actor_role,
        header_user_id=x_user_id,
        header_request_source=x_request_source,
    )

    try:
        if payload.prompt and payload.prompt.strip():
            result: ToolInvocationResult = execute_from_prompt(
                db=db,
                context=context,
                prompt=payload.prompt,
            )
        elif payload.tool_name:
            result = execute_tool(
                db=db,
                context=context,
                tool_name=payload.tool_name,
                arguments=_tool_args(payload),
            )
        else:
            raise ToolSchemaError(
                "local_context.invalid_request",
                "Either prompt or tool_name must be supplied.",
                "Send a non-empty `prompt` or a supported `tool_name` with arguments.",
            )
    except ToolSchemaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_to_error(exc).model_dump())
    except ToolAuthorizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_to_error(exc).model_dump())
    except ToolDispatchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_to_error(exc).model_dump())

    return LocalContextResponse(
        request_uuid=context.request_uuid,
        tool=result.tool_name,
        status="ok",
        latency_ms=result.latency_ms,
        decision_rationale=result.decision_rationale,
        input_hash=result.input_hash,
        result_hash=result.result_hash,
        audit_id=result.audit_id,
        output=result.output,
    )

