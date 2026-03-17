"""LiteLLM-backed LLM service with multi-provider support."""
import json
import re
from html import unescape
from urllib.parse import urlparse
from typing import Any
from ..config import get_settings


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
  "fit_score": <integer 0-100>,
  "matched_keywords": [<list of strings>],
  "missing_keywords": [<list of strings>],
  "gap_analysis": "<2-4 sentence narrative>",
  "rewrite_suggestions": [<list of 3-5 specific suggestion strings>]
}}

CV:
{cv_text}

Job Description:
{jd_text}
"""


async def score_cv_against_jd(cv_text: str, jd_text: str) -> dict:
    import litellm

    model, kwargs = _get_litellm_model()
    prompt = SCORE_PROMPT.format(cv_text=cv_text[:6000], jd_text=jd_text[:4000])

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

    data = json.loads(raw)

    # Normalise
    data["fit_score"] = max(0, min(100, int(data.get("fit_score", 0))))
    data["matched_keywords"] = data.get("matched_keywords") or []
    data["missing_keywords"] = data.get("missing_keywords") or []
    data["gap_analysis"] = data.get("gap_analysis") or ""
    data["rewrite_suggestions"] = data.get("rewrite_suggestions") or []

    return data


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
) -> str:
    import litellm

    model, kwargs = _get_litellm_model()
    safe_message = message.strip()
    if not safe_message:
        return "Please share a specific question about the problem or your approach."

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
    )

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
