"""Practice question retrieval and follow-up generation APIs."""

from __future__ import annotations

from datetime import datetime
import json
import subprocess
import textwrap
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.practice import PracticeGeneration, PracticeQuestion, PracticeSession, PracticeSessionQuestion
from ..models.score import Job
from ..services import llm as llm_service
from ..services.practice_vector import pick_best_candidate
from ..services.practice_sync import (
    get_questions_for_company,
    resolve_company_slug,
    normalize_company_slug,
    sync_all,
)

router = APIRouter(prefix="/practice", tags=["practice"])


class PracticeSyncRequest(BaseModel):
    company_slug: str | None = None
    preferred_window: str | None = Field(default=None, description="Prefer this window filename")


class PracticeSyncResponse(BaseModel):
    commit: str
    inserted: int
    updated: int
    retired: int
    companies: list[dict[str, int | str]]


class PracticeQuestionResponse(BaseModel):
    id: int
    title: str
    url: str | None = None
    difficulty: str | None = None
    acceptance: str | None = None
    frequency: str | None = None
    source_file: str | None = None
    source_window: str | None = None
    prompt: str | None = None
    is_ai_generated: bool = False
    is_solved: bool = False

    class Config:
        from_attributes = True


class PracticeQuestionsResponse(BaseModel):
    session_id: str
    company_slug: str
    questions: list[PracticeQuestionResponse]


class MarkSolvedRequest(BaseModel):
    session_id: str
    question_id: int


class UnmarkSolvedRequest(BaseModel):
    session_id: str
    question_id: int


class MarkSolvedResponse(BaseModel):
    session_id: str
    question_id: int
    status: str


class DiscardQuestionRequest(BaseModel):
    session_id: str
    question_id: int


class DiscardQuestionResponse(BaseModel):
    session_id: str
    question_id: int
    status: str


class PracticeNextRequest(BaseModel):
    session_id: str
    solved_question_id: int
    difficulty_delta: int | None = Field(default=None, ge=-2, le=2)
    language: str | None = None
    technique: str | None = None
    complexity: str | None = None
    time_pressure_minutes: int | None = None
    pattern: str | None = None


class ConstraintMetadata(BaseModel):
    difficulty_delta: int | None = None
    language: str | None = None
    technique: str | None = None
    complexity: str | None = None
    time_pressure_minutes: int | None = None
    pattern: str | None = None


class PracticeNextResponse(BaseModel):
    base_question_id: int
    base_question_link: str
    transformed_prompt: str
    constraint_metadata: ConstraintMetadata
    reason: str
    next_question: PracticeQuestionResponse | None = None


class PracticeChatMessage(BaseModel):
    role: str
    content: str


class PracticeInterviewChatRequest(BaseModel):
    session_id: str
    question_id: int
    message: str
    language: str | None = None
    interview_history: list[PracticeChatMessage] = Field(default_factory=list)
    solution_text: str | None = None


class PracticeInterviewChatResponse(BaseModel):
    session_id: str
    question_id: int
    interviewer_reply: str


class PracticeSolutionReviewRequest(BaseModel):
    session_id: str
    question_id: int
    solution_text: str
    language: str | None = None
    followup_difficulty_delta: int | None = Field(default=None, ge=-2, le=2)


class PracticeSolutionReviewResponse(BaseModel):
    session_id: str
    question_id: int
    review_summary: str
    strengths: list[str]
    concerns: list[str]
    follow_up_prompt: str
    follow_up_reason: str


class PracticeTestCaseItem(BaseModel):
    name: str
    input: str
    expected_output: str
    rationale: str | None = None


class PracticeTestCaseRequest(BaseModel):
    session_id: str
    question_id: int
    language: str | None = None
    count: int | None = Field(default=8, ge=2, le=12)


class PracticeTestCaseResponse(BaseModel):
    session_id: str
    question_id: int
    language: str
    test_cases: list[PracticeTestCaseItem]
    llm_provider: str | None = None
    llm_model: str | None = None


class PracticeSolutionTemplateRequest(BaseModel):
    session_id: str
    question_id: int
    language: str | None = None
    question_prompt: str | None = None


class PracticeSolutionTemplateResponse(BaseModel):
    session_id: str
    question_id: int
    language: str
    template: str
    problem_prompt: str | None = None
    signature: str | None = None
    notes: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None


class PracticeRunTestCase(BaseModel):
    name: str
    input: str
    expected_output: str
    rationale: str | None = None


class PracticeRunRequest(BaseModel):
    session_id: str
    question_id: int
    code: str
    language: str | None = None
    tests: list[PracticeRunTestCase] = Field(default_factory=list)


class PracticeRunCaseResult(BaseModel):
    name: str
    input: str
    expected_output: str
    actual_output: str | None = None
    passed: bool | None = None
    error: str | None = None


class PracticeRunResponse(BaseModel):
    session_id: str
    question_id: int
    status: str
    summary: str
    passed: int
    total: int
    results: list[PracticeRunCaseResult]
    output: list[str]
    llm_provider: str | None = None
    llm_model: str | None = None


class PracticeSessionQuestionResponse(BaseModel):
    session_id: str
    company_slug: str
    question_ids: list[int]
    status: str


def _get_or_create_session(db: Session, job_id: int | None, company_slug: str, session_id: str | None) -> PracticeSession:
    if session_id:
        session = (
            db.query(PracticeSession)
            .filter(PracticeSession.session_id == session_id, PracticeSession.is_active.is_(True))
            .first()
        )
        if session:
            if session.company_slug != company_slug:
                session.is_active = False
            else:
                return session

    resolved_slug = normalize_company_slug(company_slug)
    session = PracticeSession(
        session_id=uuid4().hex,
        job_id=job_id,
        company_slug=resolved_slug,
        is_active=True,
        last_constraint=None,
    )
    db.add(session)
    db.flush()
    return session


def _build_followup_question(source_question: PracticeQuestion) -> PracticeQuestion:
    q = PracticeQuestion(
        title=source_question.title,
        url=source_question.url,
        difficulty=source_question.difficulty,
        acceptance=source_question.acceptance,
        is_active=True,
        source_commit=getattr(source_question, 'source_commit', None),
    )
    q.frequency = None
    q.source_window = "generated"
    return q


def _normalize_interview_messages(messages: list[PracticeChatMessage] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not messages:
        return normalized
    for message in messages[-12:]:
        role = message.role.strip().lower()
        if role not in {"user", "assistant", "interviewer"}:
            role = "user"
        content = message.content.strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _coerce_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _coerce_non_empty_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _build_python_runner_script() -> str:
    return textwrap.dedent(
        '''
        import json
        import ast
        import io
        import sys
        import traceback

        payload = json.loads(r"""__PAYLOAD_PLACEHOLDER__""")

        code = payload.get("code", "")
        tests = payload.get("tests", [])
        output_lines = []
        results = []
        namespace = {}

        def _coerce_value(raw_value):
            if raw_value is None:
                return None
            if isinstance(raw_value, (int, float, bool, list, dict, tuple, str)):
                return raw_value
            if not isinstance(raw_value, str):
                return raw_value
            text = raw_value.strip()
            if not text:
                return ""
            try:
                value = json.loads(text)
            except Exception:
                try:
                    value = ast.literal_eval(text)
                except Exception:
                    return text
            if isinstance(value, str) and len(value) >= 2:
                first = value[0]
                last = value[-1]
                if first == last and first in {'"', "'"}:
                    return value[1:-1]
            return value

        def _normalize_for_compare(value):
            if isinstance(value, (set, tuple)):
                return list(value)
            return value

        def _call_solution(solution_fn, call_args, call_kwargs):
            if not callable(solution_fn):
                raise TypeError("No executable solution found.")

            def _is_signature_error(exc):
                lowered = str(exc).lower()
                return (
                    "positional argument" in lowered
                    or "required positional argument" in lowered
                    or "takes" in lowered
                    or "too many positional" in lowered
                    or "missing" in lowered
                    or "unexpected keyword" in lowered
                    or "got multiple values" in lowered
                    or "object of type '" in lowered
                )

            def _looks_like_instance_method(candidate_callable):
                if not callable(candidate_callable):
                    return False
                underlying_callable = getattr(candidate_callable, "__func__", candidate_callable)
                if not callable(underlying_callable):
                    return False
                try:
                    code = underlying_callable.__code__
                    if not code.co_argcount:
                        return False
                    first_arg = code.co_varnames[0]
                    return first_arg in {"self", "cls"}
                except Exception:
                    return False

            def _add_candidate(candidates, seen, callable_obj, args):
                if not callable(callable_obj):
                    return
                key = (id(callable_obj), repr(args))
                if key in seen:
                    return
                candidates.append((callable_obj, list(args)))
                seen.add(key)

            bound_self = getattr(solution_fn, "__self__", None)
            underlying = getattr(solution_fn, "__func__", None)
            solution_name = getattr(solution_fn, "__name__", "")
            qualname = getattr(solution_fn, "__qualname__", "")
            seen_candidates: set[tuple[int, str]] = set()
            candidates: list[tuple[object, list[object]]] = []

            if bound_self is not None and underlying is not None:
                _add_candidate(candidates, seen_candidates, underlying, list(call_args))
                _add_candidate(candidates, seen_candidates, underlying, [bound_self] + list(call_args))

            if "." in qualname:
                qual_parts = [part for part in qualname.split(".") if part]
                class_names = qual_parts[:-1]
                cls_obj = None
                for class_name in class_names:
                    candidate_class = namespace.get(class_name)
                    if isinstance(candidate_class, type):
                        cls_obj = candidate_class
                        break
                if cls_obj is not None:
                    try:
                        cls_instance = cls_obj()
                    except Exception:
                        cls_instance = None
                    candidate_names = (
                        solution_name,
                        "solve",
                        "solution",
                        "main",
                        "two_sum",
                        "twoSum",
                        "add_two_numbers",
                        "addTwoNumbers",
                        "add_binary",
                        "addBinary",
                        "f",
                    )
                    for candidate_name in candidate_names:
                        if not candidate_name:
                            continue
                        class_method = getattr(cls_obj, candidate_name, None)
                        class_attr = getattr(cls_obj, candidate_name, None)
                        _add_candidate(candidates, seen_candidates, class_method, list(call_args))
                        if cls_instance is not None:
                            class_method_bound = getattr(cls_instance, candidate_name, None)
                            _add_candidate(candidates, seen_candidates, class_method_bound, list(call_args))
                            _add_candidate(candidates, seen_candidates, class_attr, [cls_instance] + list(call_args))
                    if not _looks_like_instance_method(solution_fn) and cls_instance is not None and solution_name:
                        unbound_default = getattr(cls_obj, solution_name, None)
                        _add_candidate(candidates, seen_candidates, unbound_default, list(call_args))

            _add_candidate(candidates, seen_candidates, solution_fn, list(call_args))

            candidates.sort(key=lambda item: (1 if _looks_like_instance_method(item[0]) else 0))

            last_error = None
            for index, (candidate, candidate_args) in enumerate(candidates):
                try:
                    return candidate(*candidate_args, **call_kwargs)
                except TypeError as exc:
                    last_error = exc
                    if not _is_signature_error(exc):
                        raise
                    if index == 0 and bound_self is not None:
                        # Retry even if first candidate already looked bound;
                        # this catches static-like calls where self is omitted.
                        continue
                    # If all variants look signature-related, keep trying the next signature form.
                    if index < len(candidates) - 1:
                        continue
                    raise

            if last_error is None:
                raise TypeError("No executable solution found.")
            raise last_error

        try:
            exec(code, namespace, namespace)
        except Exception:
            results.append(
                {
                    "name": "compile",
                    "input": "",
                    "expected_output": "",
                    "actual_output": None,
                    "passed": False,
                    "error": traceback.format_exc(),
                }
            )
            print(json.dumps({"status": "compile_error", "summary": "Code compilation failed.", "results": results, "output": output_lines}))
            raise SystemExit(0)

        solution = None
        if "Solution" in namespace and isinstance(namespace["Solution"], type):
            try:
                solution_instance = namespace["Solution"]()
                solution_class = namespace["Solution"]
                def _uses_instance_first_param(callable_obj):
                    if not callable(callable_obj):
                        return False
                    underlying_callable = getattr(callable_obj, "__func__", callable_obj)
                    if not callable(underlying_callable):
                        return False
                    try:
                        code = underlying_callable.__code__
                        if not code.co_argcount:
                            return False
                        first_arg = code.co_varnames[0]
                        return first_arg in {"self", "cls"}
                    except Exception:
                        return False

                if hasattr(solution_instance, "solve"):
                    solution = solution_instance.solve
                if solution is None:
                    for name in (
                        "solve",
                        "solution",
                        "main",
                        "two_sum",
                        "twoSum",
                        "add_two_numbers",
                        "addTwoNumbers",
                        "add_binary",
                        "addBinary",
                        "f",
                    ):
                        if hasattr(solution_instance, name):
                            class_method = getattr(solution_class, name, None)
                            instance_method = getattr(solution_instance, name)
                            if callable(class_method) and not _uses_instance_first_param(class_method):
                                solution = class_method
                            else:
                                solution = instance_method
                            break
                if solution is None:
                    public_methods = [
                        name
                        for name in dir(solution_instance)
                        if callable(getattr(solution_instance, name))
                        and not name.startswith("_")
                    ]
                    if public_methods:
                        solution = getattr(solution_instance, public_methods[0])
            except Exception:
                solution = None
        if solution is None:
            for name in ("solve", "solution", "main", "two_sum", "f"):
                candidate = namespace.get(name)
                if callable(candidate):
                    solution = candidate
                    break
        if solution is None:
            output_lines.append("No callable solution function found (tried Solution.solve, solve, solution, main, two_sum, f).")

        for index, case in enumerate(tests, start=1):
            name = case.get("name") or f"Case {index}"
            raw_input = case.get("input", "")
            expected_raw = case.get("expected_output", "")
            input_value = _coerce_value(raw_input)
            expected_value = _coerce_value(expected_raw)
            if expected_value is None:
                expected_value = ""

            if isinstance(input_value, dict) and {"args", "kwargs"} <= set(input_value.keys()):
                call_args = input_value.get("args", [])
                call_kwargs = input_value.get("kwargs", {})
            else:
                if isinstance(input_value, tuple):
                    call_args = list(input_value)
                    call_kwargs = {}
                elif isinstance(input_value, list):
                    call_args = list(input_value)
                    call_kwargs = {}
                else:
                    call_args = [input_value]
                    call_kwargs = {}

            if not isinstance(call_args, list):
                call_args = [call_args]
            if call_kwargs is None or not isinstance(call_kwargs, dict):
                call_kwargs = {}

            if solution is None:
                results.append(
                    {
                        "name": name,
                        "input": str(raw_input),
                        "expected_output": str(expected_raw),
                        "actual_output": None,
                        "passed": False,
                        "error": "No executable solution found.",
                    }
                )
                continue

            buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buffer
            try:
                actual = _call_solution(solution, call_args, call_kwargs)
                sys.stdout = old_stdout
                captured = buffer.getvalue().strip()
                if captured:
                    output_lines.append(captured)
                if _normalize_for_compare(actual) == _normalize_for_compare(expected_value):
                    passed = True
                    error = None
                else:
                    passed = False
                    error = "Output mismatch."
                results.append(
                    {
                        "name": name,
                        "input": str(raw_input),
                        "expected_output": str(expected_raw),
                        "actual_output": repr(_normalize_for_compare(actual)),
                        "passed": passed,
                        "error": error,
                    }
                )
            except Exception:
                sys.stdout = old_stdout
                results.append(
                    {
                        "name": name,
                        "input": str(raw_input),
                        "expected_output": str(expected_raw),
                        "actual_output": None,
                        "passed": False,
                        "error": traceback.format_exc(),
                    }
                )

        print(json.dumps({"status": "complete", "summary": "Run completed", "results": results, "output": output_lines}))
        '''
    )


def _serialize_run_payload(code: str, tests: list[PracticeRunTestCase]) -> str:
    payload = {
        "code": code,
        "tests": [
            {
                "name": case.name,
                "input": case.input,
                "expected_output": case.expected_output,
                "rationale": case.rationale,
            }
            for case in tests
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


@router.post("/sync", response_model=PracticeSyncResponse)
def sync_questions(payload: PracticeSyncRequest, db: Session = Depends(get_db)) -> PracticeSyncResponse:
    try:
        result = sync_all(db, payload.company_slug, payload.preferred_window)
        db.commit()
        return PracticeSyncResponse(
            commit=str(result["commit"]),
            inserted=int(result["inserted"]),
            updated=int(result["updated"]),
            retired=int(result["retired"]),
            companies=result["companies"],  # type: ignore[arg-type]
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/company/{company_slug}/questions", response_model=PracticeQuestionsResponse)
def list_company_questions(
    company_slug: str,
    job_id: int,
    db: Session = Depends(get_db),
    session_id: str | None = None,
    limit: int = 8,
    difficulty: str | None = None,
    source_window: str | None = "all",
    include_solved: bool = False,
) -> PracticeQuestionsResponse:
    resolved_company = resolve_company_slug(company_slug, db=db)
    session = _get_or_create_session(db, job_id=job_id, company_slug=resolved_company, session_id=session_id)

    limit_value = max(1, int(limit))
    exclude_statuses = ["discarded"] if include_solved else ["solved", "discarded"]

    try:
        # If no questions loaded yet, sync this company once.
        questions = get_questions_for_company(
            db=db,
            company_slug=resolved_company,
            job_session_id=session.id,
            exclude_statuses=exclude_statuses,
            limit=limit_value,
            difficulty=difficulty,
            source_window=source_window,
            recent_window_minutes=get_settings().practice_restrict_dup_window_minutes,
        )
        if not questions:
            try:
                sync_all(db, company=resolved_company, preferred_window=source_window)
            except RuntimeError:
                # Missing company folders / renamed repos should not fail the whole page.
                # Surface an empty set instead so callers can show a graceful fallback.
                db.rollback()
                return PracticeQuestionsResponse(
                    session_id=session.session_id,
                    company_slug=resolved_company,
                    questions=[],
                )
            questions = get_questions_for_company(
                db=db,
                company_slug=resolved_company,
                job_session_id=session.id,
                exclude_statuses=exclude_statuses,
                limit=limit_value,
                difficulty=difficulty,
                source_window=source_window,
                recent_window_minutes=get_settings().practice_restrict_dup_window_minutes,
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))

    session_question_rows = {
        row.question_id: row.status
        for row in db.query(PracticeSessionQuestion.question_id, PracticeSessionQuestion.status)
        .filter(PracticeSessionQuestion.practice_session_id == session.id)
        .all()
    }

    return PracticeQuestionsResponse(
        session_id=session.session_id,
        company_slug=resolved_company,
        questions=[
            PracticeQuestionResponse(
                id=q.id,
                title=q.title,
                url=q.url,
                difficulty=q.difficulty,
                acceptance=q.acceptance,
                frequency=q.frequency,
                source_file=q.source_file,
                source_window=q.source_window,
                is_ai_generated=session_question_rows.get(q.id) == "ai-generated",
                is_solved=session_question_rows.get(q.id) == "solved",
                prompt=None,
            )
            for q in questions
        ],
    )


@router.post("/mark-solved", response_model=MarkSolvedResponse)
def mark_solved(payload: MarkSolvedRequest, db: Session = Depends(get_db)) -> MarkSolvedResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == payload.question_id,
            )
        )
        .first()
    )
    if row is None:
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=question.id,
                status="solved",
                asked_at=datetime.utcnow(),
                solved_at=datetime.utcnow(),
            )
        )
    else:
        row.status = "solved"
        row.asked_at = datetime.utcnow()
        row.solved_at = datetime.utcnow()

    session.last_constraint = None
    session.updated_at = datetime.utcnow()
    db.commit()
    return MarkSolvedResponse(session_id=session.session_id, question_id=question.id, status="solved")


@router.post("/unmark-solved", response_model=MarkSolvedResponse)
def unmark_solved(payload: UnmarkSolvedRequest, db: Session = Depends(get_db)) -> MarkSolvedResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == payload.question_id,
            )
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=400, detail="Question is not marked as solved in this session")
    if row.status == "discarded":
        raise HTTPException(status_code=400, detail="Discarded questions cannot be restored here")

    row.status = "seen"
    row.solved_at = None
    row.asked_at = datetime.utcnow()
    session.updated_at = datetime.utcnow()
    db.commit()
    return MarkSolvedResponse(session_id=session.session_id, question_id=question.id, status="unsolved")


@router.post("/discard", response_model=DiscardQuestionResponse)
def discard_question(payload: DiscardQuestionRequest, db: Session = Depends(get_db)) -> DiscardQuestionResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == question.id,
            )
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=400, detail="Only generated questions can be discarded from this session")
    if row.status == "solved":
        raise HTTPException(status_code=400, detail="Solved questions cannot be discarded")
    if row.status != "ai-generated":
        raise HTTPException(status_code=400, detail="Only AI-generated questions can be discarded from this session")

    row.status = "discarded"
    row.asked_at = datetime.utcnow()

    db.commit()
    return DiscardQuestionResponse(
        session_id=session.session_id,
        question_id=question.id,
        status="discarded",
    )

@router.post("/next", response_model=PracticeNextResponse)
@router.post("/next/", response_model=PracticeNextResponse)
async def next_follow_up(payload: PracticeNextRequest, db: Session = Depends(get_db)) -> PracticeNextResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    base_question = (
        db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.solved_question_id).first()
    )
    if not base_question:
        raise HTTPException(status_code=404, detail="Solved question not found")
    if base_question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Solved question does not match session company")

    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == base_question.id,
            )
        )
        .first()
    )
    if row is None:
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=base_question.id,
                status="seen",
                asked_at=datetime.utcnow(),
            )
        )
    else:
        if row.status != "solved":
            row.status = "seen"
        row.asked_at = datetime.utcnow()
        if row.status == "solved":
            row.solved_at = datetime.utcnow()

    payload_constraints = {
        "difficulty_delta": payload.difficulty_delta,
        "language": payload.language,
        "technique": payload.technique,
        "complexity": payload.complexity,
        "time_pressure_minutes": payload.time_pressure_minutes,
        "pattern": payload.pattern,
    }

    settings = get_settings()
    candidate_limit = max(settings.practice_default_limit, 40)
    pool = get_questions_for_company(
        db=db,
        company_slug=session.company_slug,
        job_session_id=session.id,
        exclude_statuses=["solved", "discarded", "ai-generated", "seen"],
        limit=candidate_limit,
        difficulty=None,
        source_window=None,
        recent_window_minutes=settings.practice_restrict_dup_window_minutes,
    )
    candidate_list = [question for question in pool if question.id != base_question.id]
    if not candidate_list:
        # Try one final retrieval without the duplicate window to avoid dead-ends.
        fallback_pool = get_questions_for_company(
            db=db,
            company_slug=session.company_slug,
            job_session_id=None,
            exclude_statuses=["solved", "discarded", "ai-generated", "seen"],
            limit=max(settings.practice_default_limit, 80),
            difficulty=None,
            source_window=None,
            recent_window_minutes=0,
        )
        candidate_list = [question for question in fallback_pool if question.id != base_question.id]

    if not candidate_list:
        llm_result = await llm_service.generate_constrained_followup(
            title=base_question.title,
            url=base_question.url or "",
            company=session.company_slug,
            difficulty=base_question.difficulty,
            acceptance=base_question.acceptance,
            frequency=base_question.frequency,
            constraints=payload_constraints,
        )
        llm_model = str(llm_result.get("llm_model") or "unknown")
        llm_provider = str(llm_result.get("llm_provider") or get_settings().active_llm_provider)
        transformed_prompt = _coerce_non_empty_string(llm_result.get("transformed_prompt"))
        if not transformed_prompt:
            transformed_prompt = (
                "Take this base problem and apply the requested constraints to design a follow-up variant with similar skills."
            )
        base_link = str(llm_result.get("base_question_link") or base_question.url or "")
        reason = _coerce_non_empty_string(llm_result.get("reason")) or "No additional questions available for this session."

        generated_question = _build_followup_question(source_question=base_question)
        generated_question.title = f"{base_question.title} — Follow-up"
        db.add(generated_question)
        db.flush()
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=generated_question.id,
                status="ai-generated",
                asked_at=datetime.utcnow(),
            )
        )
        db.add(
            PracticeGeneration(
                practice_session_id=session.id,
                source_question_id=base_question.id,
                generated_text=transformed_prompt,
                constraint_type="custom",
                applied_constraints=payload_constraints,
                reason=reason,
                base_question_link=base_question.url,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        )
        session.last_constraint = payload_constraints
        session.updated_at = datetime.utcnow()
        db.commit()

        return PracticeNextResponse(
            base_question_id=base_question.id,
            base_question_link=base_link,
            transformed_prompt=transformed_prompt,
            constraint_metadata=ConstraintMetadata(**payload_constraints),
            reason=reason,
            next_question=PracticeQuestionResponse(
                id=generated_question.id,
                title=generated_question.title,
                url=generated_question.url,
                difficulty=generated_question.difficulty,
                acceptance=generated_question.acceptance,
                frequency=generated_question.frequency,
                source_file=generated_question.source_file,
                source_window=generated_question.source_window,
                prompt=transformed_prompt,
                is_ai_generated=True,
            ),
        )

    job_description: str | None = None
    if session.job_id:
        job = db.query(Job).filter(Job.id == session.job_id).first()
        if job and job.description:
            job_description = job.description

    candidate, _ = await pick_best_candidate(
        db=db,
        base_question=base_question,
        candidate_questions=candidate_list,
        difficulty_delta=payload.difficulty_delta,
        job_description=job_description,
        language=payload.language,
        technique=payload.technique,
        complexity=payload.complexity,
        time_pressure_minutes=payload.time_pressure_minutes,
        pattern=payload.pattern,
    )
    if candidate is None:
        candidate = candidate_list[0]

    generated_question = _build_followup_question(source_question=candidate)
    db.add(generated_question)
    db.flush()
    db.add(
        PracticeSessionQuestion(
            practice_session_id=session.id,
            question_id=generated_question.id,
            status="ai-generated",
            asked_at=datetime.utcnow(),
        )
    )
    llm_result = await llm_service.generate_constrained_followup(
        title=base_question.title,
        url=base_question.url or "",
        company=session.company_slug,
        difficulty=base_question.difficulty,
        acceptance=base_question.acceptance,
        frequency=base_question.frequency,
        constraints=payload_constraints,
    )
    llm_model = str(llm_result.get("llm_model") or "unknown")
    llm_provider = str(llm_result.get("llm_provider") or get_settings().active_llm_provider)

    db.add(
        PracticeGeneration(
            practice_session_id=session.id,
            source_question_id=base_question.id,
            generated_text=llm_result["transformed_prompt"],
            constraint_type="custom",
            applied_constraints=payload_constraints,
            reason=llm_result.get("reason"),
            base_question_link=base_question.url,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    )
    session.last_constraint = payload_constraints
    session.updated_at = datetime.utcnow()
    db.commit()

    return PracticeNextResponse(
        base_question_id=base_question.id,
        base_question_link=str(llm_result.get("base_question_link") or base_question.url or ""),
        transformed_prompt=str(llm_result.get("transformed_prompt", "")),
        constraint_metadata=ConstraintMetadata(**payload_constraints),
        reason=str(llm_result.get("reason", "Generated from solved question")),
        next_question=PracticeQuestionResponse(
            id=generated_question.id,
            title=generated_question.title,
            url=generated_question.url,
            difficulty=generated_question.difficulty,
            acceptance=generated_question.acceptance,
            frequency=generated_question.frequency,
            source_file=generated_question.source_file,
            source_window=generated_question.source_window,
            prompt=str(llm_result.get("transformed_prompt", "")) or None,
            is_ai_generated=True,
        ),
    )


@router.post("/interview-chat", response_model=PracticeInterviewChatResponse)
async def interview_chat(payload: PracticeInterviewChatRequest, db: Session = Depends(get_db)) -> PracticeInterviewChatResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Question does not match session company")

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = _normalize_interview_messages(payload.interview_history)
    reply = await llm_service.simulate_interviewer_chat(
        title=question.title,
        url=question.url or "",
        company=session.company_slug,
        difficulty=question.difficulty,
        acceptance=question.acceptance,
        frequency=question.frequency,
        language=payload.language,
        message=message,
        conversation=history,
        solution_text=payload.solution_text,
    )

    return PracticeInterviewChatResponse(session_id=session.session_id, question_id=question.id, interviewer_reply=reply)


@router.post("/review-solution", response_model=PracticeSolutionReviewResponse)
async def review_solution(payload: PracticeSolutionReviewRequest, db: Session = Depends(get_db)) -> PracticeSolutionReviewResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Question does not match session company")

    solution_text = payload.solution_text.strip()
    if not solution_text:
        raise HTTPException(status_code=400, detail="solution_text cannot be empty")

    now = datetime.utcnow()
    row = (
        db.query(PracticeSessionQuestion)
        .filter(
            and_(
                PracticeSessionQuestion.practice_session_id == session.id,
                PracticeSessionQuestion.question_id == question.id,
            )
        )
        .first()
    )
    if row is None:
        db.add(
            PracticeSessionQuestion(
                practice_session_id=session.id,
                question_id=question.id,
                status="solved",
                asked_at=now,
                solved_at=now,
            )
        )
    else:
        row.status = "solved"
        row.asked_at = now
        row.solved_at = now

    settings = get_settings()
    llm_result = await llm_service.review_solution_with_variant(
        title=question.title,
        url=question.url or "",
        company=session.company_slug,
        difficulty=question.difficulty,
        acceptance=question.acceptance,
        frequency=question.frequency,
        solution_text=solution_text,
        language=payload.language,
        difficulty_delta=payload.followup_difficulty_delta,
    )

    db.add(
        PracticeGeneration(
            practice_session_id=session.id,
            source_question_id=question.id,
            generated_text=str(llm_result.get("review_summary", "")),
            constraint_type="solution-review",
            applied_constraints={
                "language": payload.language,
                "followup_difficulty_delta": payload.followup_difficulty_delta,
            },
            reason=_coerce_non_empty_string(llm_result.get("follow_up_reason")),
            base_question_link=question.url,
            llm_provider=str(llm_result.get("llm_provider") or settings.active_llm_provider),
            llm_model=str(llm_result.get("llm_model") or settings.openai_model),
        )
    )
    session.last_constraint = {"type": "review", "difficulty_delta": payload.followup_difficulty_delta}
    session.updated_at = now
    db.commit()

    return PracticeSolutionReviewResponse(
        session_id=session.session_id,
        question_id=question.id,
        review_summary=_coerce_non_empty_string(llm_result.get("review_summary")),
        strengths=_coerce_string_list(llm_result.get("strengths")),
        concerns=_coerce_string_list(llm_result.get("concerns")),
        follow_up_prompt=_coerce_non_empty_string(llm_result.get("follow_up_prompt")),
        follow_up_reason=_coerce_non_empty_string(llm_result.get("follow_up_reason")),
    )
 
 
 
@router.post("/solution-template", response_model=PracticeSolutionTemplateResponse)
async def generate_solution_template(
    payload: PracticeSolutionTemplateRequest,
    db: Session = Depends(get_db),
) -> PracticeSolutionTemplateResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Question does not match session company")

    template_result = await llm_service.generate_solution_template(
        title=question.title,
        url=question.url or "",
        company=session.company_slug,
        difficulty=question.difficulty,
        acceptance=question.acceptance,
        frequency=question.frequency,
        language=payload.language,
        prompt=payload.question_prompt,
    )

    return PracticeSolutionTemplateResponse(
        session_id=session.session_id,
        question_id=question.id,
        language=_coerce_non_empty_string(template_result.get("language")) or (payload.language or "python"),
        template=_coerce_non_empty_string(template_result.get("template"))
        or "class Solution:\\n    def solve(self, nums, target):\\n        # TODO: implement\\n        raise NotImplementedError\\n",
        problem_prompt=_coerce_non_empty_string(template_result.get("problem_prompt")) or None,
        signature=(template_result.get("signature") if isinstance(template_result.get("signature"), str) else None),
        notes=(template_result.get("notes") if isinstance(template_result.get("notes"), str) else None),
        llm_provider=str(template_result.get("llm_provider")) if template_result.get("llm_provider") else None,
        llm_model=str(template_result.get("llm_model")) if template_result.get("llm_model") else None,
    )


@router.post("/run", response_model=PracticeRunResponse)
async def run_practice_code(payload: PracticeRunRequest, db: Session = Depends(get_db)) -> PracticeRunResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Question does not match session company")

    requested_test_count = len(payload.tests)
    runner_script = _build_python_runner_script().replace(
        "__PAYLOAD_PLACEHOLDER__", _serialize_run_payload(payload.code, payload.tests)
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        runner_file = Path(temp_dir) / "runner.py"
        runner_file.write_text(runner_script, encoding="utf-8")
        try:
            completed = subprocess.run(
                ["python3", str(runner_file)],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return PracticeRunResponse(
                session_id=session.session_id,
                question_id=question.id,
                status="timeout",
                summary="Code execution exceeded the time limit.",
                passed=0,
                total=requested_test_count,
                results=[],
                output=["Execution timed out after 8 seconds."],
            )
        except Exception as exc:
            return PracticeRunResponse(
                session_id=session.session_id,
                question_id=question.id,
                status="error",
                summary=f"Runner failed before execution: {exc}",
                passed=0,
                total=requested_test_count,
                results=[],
                output=[str(exc)],
            )

    if not completed.stdout.strip():
        output = [line for line in (completed.stderr or "").splitlines() if line.strip()]
        return PracticeRunResponse(
            session_id=session.session_id,
            question_id=question.id,
            status="error",
            summary="No execution output was produced.",
            passed=0,
                total=requested_test_count,
            results=[],
            output=output,
        )

    try:
        parsed_output = json.loads(completed.stdout.strip())
    except Exception:
        return PracticeRunResponse(
            session_id=session.session_id,
            question_id=question.id,
            status="error",
            summary="Could not parse execution output.",
            passed=0,
                total=requested_test_count,
            results=[],
            output=[completed.stdout.strip()] + [line for line in (completed.stderr or "").splitlines() if line.strip()],
        )

    parsed_results = parsed_output.get("results", [])
    if not isinstance(parsed_results, list):
        parsed_results = []

    run_results = [
        PracticeRunCaseResult(
            name=_coerce_non_empty_string(item.get("name")) or f"Case {index + 1}",
            input=_coerce_non_empty_string(item.get("input")),
            expected_output=_coerce_non_empty_string(item.get("expected_output")),
            actual_output=item.get("actual_output") if isinstance(item.get("actual_output"), str) else None,
            passed=item.get("passed") if isinstance(item.get("passed"), bool) else None,
            error=_coerce_non_empty_string(item.get("error")) or None,
        )
        for index, item in enumerate(parsed_results)
        if isinstance(item, dict)
    ]

    output_lines = parsed_output.get("output") or []
    if not isinstance(output_lines, list):
        output_lines = [str(output_lines)]
    output_lines = [line for line in (str(line) for line in output_lines) if line.strip()]
    if not output_lines:
        output_lines = ["Execution completed."]

    passed_count = sum(1 for result in run_results if result.passed is True)
    if run_results:
        status = "ok" if passed_count == len(run_results) else "failed"
        summary = f"{passed_count}/{len(run_results)} cases passed."
    else:
        status = str(parsed_output.get("status") or "ok")
        summary = _coerce_non_empty_string(parsed_output.get("summary") or "Execution completed.")

    return PracticeRunResponse(
        session_id=session.session_id,
        question_id=question.id,
        status=status,
        summary=summary,
        passed=passed_count,
            total=max(len(run_results), requested_test_count),
        results=run_results,
        output=output_lines,
        llm_provider=str(parsed_output.get("llm_provider")) if parsed_output.get("llm_provider") else None,
        llm_model=str(parsed_output.get("llm_model")) if parsed_output.get("llm_model") else None,
    )


@router.post("/test-cases", response_model=PracticeTestCaseResponse)
async def generate_test_cases(payload: PracticeTestCaseRequest, db: Session = Depends(get_db)) -> PracticeTestCaseResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == payload.session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question = db.query(PracticeQuestion).filter(PracticeQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.company_slug != session.company_slug:
        raise HTTPException(status_code=400, detail="Question does not match session company")

    response = await llm_service.generate_practice_test_cases(
        title=question.title,
        url=question.url or "",
        company=session.company_slug,
        difficulty=question.difficulty,
        acceptance=question.acceptance,
        frequency=question.frequency,
        language=payload.language or "python3",
        count=payload.count or 8,
    )

    raw_cases = response.get("test_cases")
    if not isinstance(raw_cases, list):
        raw_cases = []

    test_cases = [
        PracticeTestCaseItem(
            name=str(item.get("name", f"Case {idx + 1}")).strip() or f"Case {idx + 1}",
            input=str(item.get("input", "")).strip(),
            expected_output=str(item.get("expected_output", "")).strip(),
            rationale=str(item.get("rationale", "")).strip() or None,
        )
        for idx, item in enumerate(raw_cases)
        if isinstance(item, dict)
        and str(item.get("input", "")).strip()
        and str(item.get("expected_output", "")).strip()
    ]

    if not test_cases:
        test_cases.append(
            PracticeTestCaseItem(
                name="Manual baseline",
                input="Build one minimal and one edge case manually.",
                expected_output="Validate both outputs carefully against your implementation.",
                rationale="Fallback generated locally when the model returned no usable case list.",
            )
        )

    session.last_constraint = {
        "language": payload.language or "python3",
        "case_count": payload.count or len(test_cases),
    }
    session.updated_at = datetime.utcnow()
    db.commit()

    return PracticeTestCaseResponse(
        session_id=session.session_id,
        question_id=question.id,
        language=(payload.language or "python3").strip().lower(),
        test_cases=test_cases,
        llm_provider=str(response.get("llm_provider")) if response.get("llm_provider") else None,
        llm_model=str(response.get("llm_model")) if response.get("llm_model") else None,
    )


@router.get("/sessions/{session_id}", response_model=PracticeSessionQuestionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)) -> PracticeSessionQuestionResponse:
    session = (
        db.query(PracticeSession)
        .filter(PracticeSession.session_id == session_id, PracticeSession.is_active.is_(True))
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    question_ids = [
        qid
        for (qid,) in db.query(PracticeSessionQuestion.question_id)
        .filter(PracticeSessionQuestion.practice_session_id == session.id)
        .all()
    ]
    return PracticeSessionQuestionResponse(
        session_id=session.session_id,
        company_slug=session.company_slug,
        question_ids=question_ids,
        status="active",
    )
