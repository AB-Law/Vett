"""CV parsing service – supports PDF, DOCX, and Markdown."""
import io
import re
from pathlib import Path

_WRAPPED_LINE_MAX_CHARS = 180
_WRAPPED_LINE_MAX_WORDS = 16
_CONNECTOR_LINES = {"&", "/", "–", "—", "+"}
_CONTINUATION_START_WORDS = {
    "and",
    "for",
    "with",
    "to",
    "of",
    "in",
    "on",
    "at",
    "from",
    "into",
    "using",
    "including",
    "through",
    "by",
    "as",
    "or",
    "while",
    "and",
    "that",
    "which",
}


def parse_cv(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_bytes)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(file_bytes)
    elif suffix in (".md", ".markdown", ".txt"):
        return _parse_markdown(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(data: bytes) -> str:
    parsers = (
        _parse_pdf_with_pymupdf,
        _parse_pdf_with_pdfplumber_layout,
        _parse_pdf_with_pdfplumber,
        _parse_pdf_with_pypdf2,
    )
    parsed_candidates: list[tuple[str, int, str]] = []
    errors: list[str] = []

    for parser_fn in parsers:
        try:
            extracted = parser_fn(data)
        except ImportError:
            # Optional parser dependencies are environment dependent.
            continue
        except Exception as e:
            errors.append(f"{parser_fn.__name__}: {e}")
            continue

        parsed_text = _normalise_extracted_text(extracted)
        if parsed_text:
            score = _score_parsed_text(parsed_text)
            parsed_candidates.append((parsed_text, score, parser_fn.__name__))

    if not parsed_candidates:
        if errors:
            raise ValueError(f"Failed to parse PDF: {'; '.join(errors)}")
        raise ValueError("Failed to parse PDF: no text extracted")

    # Use the cleaned result with the fewest malformed line fragments.
    return max(parsed_candidates, key=lambda item: item[1])[0]


def _parse_pdf_with_pymupdf(data: bytes) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is not installed") from exc

    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    parsed = "\n\n".join(pages).strip()

    if not parsed:
        raise ValueError("PyMuPDF produced no text")
    return parsed


def _parse_pdf_with_pdfplumber(data: bytes) -> str:
    return _parse_pdf_with_pdfplumber_impl(data, layout=False)


def _parse_pdf_with_pdfplumber_layout(data: bytes) -> str:
    return _parse_pdf_with_pdfplumber_impl(data, layout=True)


def _parse_pdf_with_pdfplumber_impl(data: bytes, layout: bool) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is not installed") from exc

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = []
        for page in pdf.pages:
            text = (
                page.extract_text(x_tolerance=2.0, y_tolerance=2.0, layout=layout)
                or ""
            )
            if text.strip():
                pages.append(text)
            else:
                # Some PDFs expose text content but not layout text in the first pass.
                pages.append("")
        parsed = "\n\n".join(pages).strip()

    if not parsed:
        raise ValueError("pdfplumber produced no text")
    return parsed


def _parse_pdf_with_pypdf2(data: bytes) -> str:
    try:
        import PyPDF2
    except ImportError as exc:
        raise ImportError("PyPDF2 is not installed") from exc

    reader = PyPDF2.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    parsed = "\n\n".join(pages).strip()

    if not parsed:
        raise ValueError("PyPDF2 produced no text")
    return parsed


def _parse_docx(data: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs).strip()
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {e}")


def _parse_markdown(data: bytes) -> str:
    try:
        text = data.decode("utf-8", errors="replace")
        # Strip markdown syntax for plain-text extraction
        # Remove headings markers, bold, italic, links, code blocks
        text = re.sub(r"#{1,6}\s+", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text, flags=re.DOTALL)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = _normalise_extracted_text(text)
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to parse Markdown: {e}")


def _normalise_extracted_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    # Repair common line-break hyphenation from PDF text extraction.
    text = re.sub(r"-\n\s*", "", text)
    text = _restore_section_breaks(text)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = _merge_fragmented_lines(lines)

    # Remove duplicated consecutive lines (common when PDFs have hidden headers/footers).
    deduped_lines: list[str] = []
    previous = None
    for line in lines:
        if line and line == previous:
            continue
        deduped_lines.append(line)
        previous = line

    text = "\n".join(deduped_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def _restore_section_breaks(text: str) -> str:
    # Handle cases where the parser loses layout and concatenates contact lines with
    # a section heading that starts in all-caps form.
    text = re.sub(
        r"(?<!\n)(\b\S+\.(?:com|org|net|io|in|edu|gov|co|ai|dev)(?:/\S+)?)(\s+)([A-Z][A-Z0-9&/().-]*(?:\s+[A-Z][A-Z0-9&/().-]*){1,})",
        r"\1\n\n\3",
        text,
    )
    return text


def _merge_fragmented_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    pending_blank = False

    for line in lines:
        if not line:
            if merged:
                pending_blank = True
            continue

        if pending_blank and merged:
            if _should_merge_lines(merged[-1], line):
                merged[-1] = f"{merged[-1]} {line}".strip()
                pending_blank = False
                continue
            merged.append("")
            pending_blank = False

        if line in _CONNECTOR_LINES:
            if merged and merged[-1] != "":
                # Keep connector tokens in the same sentence as neighboring text.
                merged[-1] = f"{merged[-1]} {line}"
            continue

        if line == "|":
            if merged and merged[-1] not in ("", "|"):
                merged[-1] = f"{merged[-1]} |"
            continue

        if merged and _should_merge_lines(merged[-1], line):
            merged[-1] = f"{merged[-1]} {line}".strip()
            continue

        merged.append(line)

    return merged


def _should_merge_lines(previous: str, current: str) -> bool:
    if not previous or not current:
        return False

    if _is_block_boundary_line(previous) or _is_block_boundary_line(current):
        return False
    if _is_section_header(previous) or _is_section_header(current):
        return False
    if _is_link_like(previous) or _is_link_like(current):
        return False

    # Preserve sentence boundaries, but keep continuations that were split after
    # abbreviations such as U.S.
    if re.search(r"[.!?]$", previous):
        if not _is_sentence_continuation(previous, current):
            return False

    if _looks_like_continuation(previous, current):
        return True

    if not _is_wrap_fragment(previous) or not _is_wrap_fragment(current):
        return False

    # Keep URLs, emails, and domains as standalone tokens.
    if (
        previous.startswith(("http://", "https://", "www."))
        or current.startswith(("http://", "https://", "www."))
        or "@" in previous
        or "@" in current
    ):
        return False

    return True


def _is_sentence_continuation(previous: str, current: str) -> bool:
    # Continuation is likely when the previous line ends with an abbreviation
    # and the next line continues the phrase.
    if _ends_with_abbreviation(previous):
        return True

    # If the next token continues with a lowercase word, it's often a wrapped
    # phrase in PDFs that should be merged.
    if current and current[0].islower():
        return True

    if _is_numeric_fragment(current):
        return True

    # If the current token is a short numeric-like fragment (e.g. 2.5+),
    # it is usually a wrapped mid-sentence split.
    if _is_numeric_fragment(previous):
        return True

    # Avoid merging after clear end-of-sentence markers.
    return False


def _looks_like_continuation(previous: str, current: str) -> bool:
    # Connector or numeric tokens often get split in PDF text streams.
    if current in _CONNECTOR_LINES or _is_numeric_fragment(current):
        return True

    # Wrapped fragments often start with a lowercase continuation word.
    if current and current[0].islower():
        return True

    # Common short continuation starters split by line breaks.
    if _is_continuation_word(current):
        return True

    # Preserve very short wrapped tails (e.g. headings, bullets, or short fragments).
    if len(current.split()) <= 2 and _is_wrap_fragment(current):
        return True

    # Avoid merging across punctuation-heavy sentence-ending context.
    if previous.rstrip().endswith((".", "!", "?")):
        return False

    return False


def _is_continuation_word(value: str) -> bool:
    return value.strip().lower() in _CONTINUATION_START_WORDS


def _is_numeric_fragment(line: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?\+?", line.strip()))


def _ends_with_abbreviation(line: str) -> bool:
    # Common abbreviation-like endings that should not force a hard break.
    if re.match(r"^\s*(?:[A-Za-z]\.){2,}$", line):
        return True
    if re.match(r"^.*\b(?:U\.S\.|U\.K\.|p\.m\.|a\.m\.|e\.g\.|i\.e\.)$", line, flags=re.IGNORECASE):
        return True
    return False


def _is_link_like(line: str) -> bool:
    line = line.strip().lower()
    if " " in line:
        return False

    if line.startswith(("http://", "https://", "www.")):
        return True

    if "@" in line:
        return True

    if "." in line and re.match(r"^[a-z0-9][a-z0-9.+#/_:-]*\.[a-z0-9][a-z0-9.+#/_:-]*$", line):
        return True

    return False


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or " " not in stripped:
        return False
    if len(stripped) > 80:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9 &/().-]*$", stripped):
        return False
    if not re.match(r"^[A-Z0-9 &/().-]+$", stripped):
        return False
    return True


def _is_block_boundary_line(line: str) -> bool:
    return bool(
        re.match(
            r"^(?:[•●◦▪︎▶]\s+|[\-\*]\s+|\d+[.)]\s+|[A-Za-z]\)\s+|[A-Za-z]\.\s+)",
            line,
        )
    ) or line.strip().endswith(":")


def _is_wrap_fragment(line: str) -> bool:
    if line in _CONNECTOR_LINES:
        return True

    if _is_link_like(line) or _is_section_header(line):
        return False

    if len(line) > _WRAPPED_LINE_MAX_CHARS:
        return False
    if len(line.split()) > _WRAPPED_LINE_MAX_WORDS:
        return False
    return bool(re.search(r"[A-Za-z0-9]", line))


def _score_parsed_text(text: str) -> int:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 0

    # Prefer fewer fragmented lines (single / very short tokens)
    short_fragments = 0
    single_word_lines = 0
    for line in lines:
        words = line.split()
        if len(words) <= 2 and line not in _CONNECTOR_LINES and not _is_abbreviation_like(line):
            short_fragments += 1
        if len(words) == 1:
            single_word_lines += 1

    # Slight reward for punctuation that likely indicates real sentence structure.
    sentence_ends = sum(1 for line in lines if re.search(r"[.!?]$", line))

    return (
        len(text)
        - (short_fragments * 35)
        - (single_word_lines * 15)
        + (sentence_ends * 5)
    )


def _is_abbreviation_like(line: str) -> bool:
    return bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", line.strip()))
