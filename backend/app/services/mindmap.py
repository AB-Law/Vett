from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - test/runtime parity
    litellm = None  # type: ignore[assignment]
from sqlalchemy.orm import Session

from ..models.interview import InterviewKnowledgeDocument
from ..models.score import Job
from ..models.study import MindMap
from . import interview_docs, llm as llm_service

logger = logging.getLogger(__name__)

MAX_NODES = 20
MAX_EDGES = 40
MAX_CONTEXT_CHUNKS = 12
MIN_NODE_LABEL_LENGTH = 2
MIN_NODE_TOKEN_LENGTH = 3

MINDMAP_PROMPT = """You are an interview study assistant.

Extract a concise concept graph from the provided context snippets.
Return exactly one JSON object with shape:
{
  "nodes": [{"id":"string","label":"string","group":"string"}],
  "edges": [{"source":"string","target":"string","label":"string"}]
}

Rules:
- Maximum 20 nodes.
- Keep labels short and specific.
- Use stable lowercase ids with underscores (example: "distributed_systems").
- Edges must reference existing node ids.
- Prefer clear relationship labels like "depends_on", "enables", "compares_with".
- Return JSON only (no markdown).
"""


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_json_payload(raw: str) -> str | None:
    matched = llm_service._extract_first_json_object(raw)  # type: ignore[attr-defined]
    if matched:
        return matched
    fallback_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not fallback_match:
        return None
    return fallback_match.group(0)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not slug:
        slug = "concept"
    return slug[:64]


def _graph_from_payload(raw_payload: object) -> dict[str, list[dict[str, str]]]:
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")

    nodes: list[dict[str, str]] = []
    node_by_id: dict[str, dict[str, str]] = {}
    seen_node_ids: set[str] = set()
    for item in raw_nodes if isinstance(raw_nodes, list) else []:
        if not isinstance(item, dict):
            continue
        label = _normalize_text(item.get("label"))
        if len(label) < MIN_NODE_LABEL_LENGTH:
            continue
        node_id_candidate = _normalize_text(item.get("id")) or _safe_slug(label)
        node_id = _safe_slug(node_id_candidate)
        if node_id in seen_node_ids:
            continue
        node = {
            "id": node_id,
            "label": label[:120],
            "group": _normalize_text(item.get("group"))[:80] or "general",
        }
        seen_node_ids.add(node_id)
        node_by_id[node_id] = node
        nodes.append(node)
        if len(nodes) >= MAX_NODES:
            break

    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for item in raw_edges if isinstance(raw_edges, list) else []:
        if not isinstance(item, dict):
            continue
        source = _safe_slug(_normalize_text(item.get("source")))
        target = _safe_slug(_normalize_text(item.get("target")))
        if source not in node_by_id or target not in node_by_id or source == target:
            continue
        label = _normalize_text(item.get("label"))[:80] or "related_to"
        key = (source, target, label.lower())
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "label": label,
            }
        )
        if len(edges) >= MAX_EDGES:
            break

    return {"nodes": nodes, "edges": edges}


def _build_context_block(contexts: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for index, context in enumerate(contexts, start=1):
        filename = _normalize_text(context.get("filename")) or f"document-{index}"
        snippet = _normalize_text(context.get("snippet")) or "No snippet available."
        rendered.append(f"{index}. {filename}: {snippet}")
    return "\n".join(rendered)


def _choose_source_chunks(graph: dict[str, list[dict[str, str]]], contexts: list[dict[str, str]]) -> dict[str, str]:
    if not contexts:
        return {}
    context_snippets = [ctx.get("snippet", "") for ctx in contexts]
    output: dict[str, str] = {}
    for node in graph.get("nodes", []):
        node_id = _normalize_text(node.get("id"))
        label = _normalize_text(node.get("label"))
        if not node_id or not label:
            continue
        tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", label.lower()) if len(token) >= MIN_NODE_TOKEN_LENGTH]
        matched = ""
        for snippet in context_snippets:
            snippet_lower = snippet.lower()
            if any(token in snippet_lower for token in tokens):
                matched = snippet
                break
        output[node_id] = matched or context_snippets[0]
    return output


async def _generate_graph_from_contexts(contexts: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    if litellm is None:
        import litellm as runtime_litellm

        completion_client = runtime_litellm
    else:
        completion_client = litellm

    model, kwargs = llm_service._get_litellm_model()  # type: ignore[attr-defined]
    prompt = "\n\n".join(
        [
            MINDMAP_PROMPT,
            "context_snippets:",
            _build_context_block(contexts),
        ]
    )
    response = await completion_client.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=4096,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    logger.debug("MINDMAP RAW RESPONSE: %r", content[:500])

    payload_text = _extract_json_payload(content) or ""
    if not payload_text.strip():
        return {"nodes": [], "edges": []}

    try:
        parsed = json.loads(payload_text)
    except Exception:
        try:
            parsed = json.loads(llm_service._sanitize_json_token_stream(payload_text))  # type: ignore[attr-defined]
        except Exception:
            parsed = {}
    return _graph_from_payload(parsed)


def _eligible_job_documents(db: Session, *, job_id: int, doc_id: int | None) -> list[InterviewKnowledgeDocument]:
    if doc_id is not None:
        docs = (
            db.query(InterviewKnowledgeDocument)
            .filter(InterviewKnowledgeDocument.id == int(doc_id))
            .all()
        )
        return docs

    # Scope to global + job-specific documents as retrieval corpus for the selected job.
    return (
        db.query(InterviewKnowledgeDocument)
        .filter(
            InterviewKnowledgeDocument.status != "failed",
            InterviewKnowledgeDocument.parsed_text.isnot(None),
            InterviewKnowledgeDocument.parsed_text != "",
            (
                (InterviewKnowledgeDocument.owner_type == "global")
                | (
                    (InterviewKnowledgeDocument.owner_type == "job")
                    & (InterviewKnowledgeDocument.job_id == int(job_id))
                )
            ),
        )
        .order_by(InterviewKnowledgeDocument.id.desc())
        .all()
    )


def _collect_contexts_for_hash(db: Session, *, job: Job, doc_id: int | None) -> tuple[list[dict[str, str]], str]:
    selected_docs = _eligible_job_documents(db, job_id=int(job.id), doc_id=doc_id)
    if not selected_docs:
        raise ValueError("No interview documents available for this selection.")

    contexts: list[dict[str, str]] = []
    if doc_id is not None:
        document = selected_docs[0]
        for index, snippet in enumerate(interview_docs.chunk_interview_text(document.parsed_text)):
            contexts.append(
                {
                    "filename": document.source_filename,
                    "snippet": snippet,
                    "chunk_id": f"{document.id}:chunk:{index}",
                }
            )
            if len(contexts) >= MAX_CONTEXT_CHUNKS:
                break
    else:
        # Pragmatic top-k approximation across job corpus: take early chunks from newest documents.
        for document in selected_docs:
            for index, snippet in enumerate(interview_docs.chunk_interview_text(document.parsed_text)[:2]):
                contexts.append(
                    {
                        "filename": document.source_filename,
                        "snippet": snippet,
                        "chunk_id": f"{document.id}:chunk:{index}",
                    }
                )
                if len(contexts) >= MAX_CONTEXT_CHUNKS:
                    break
            if len(contexts) >= MAX_CONTEXT_CHUNKS:
                break

    if not contexts:
        raise ValueError("No interview document content available for mind map extraction.")

    hash_source = "\n".join(
        [
            f"job:{job.id}",
            f"doc:{doc_id if doc_id is not None else 'all'}",
            *[f"{ctx['chunk_id']}::{ctx['snippet']}" for ctx in contexts],
        ]
    )
    content_hash = hashlib.sha256(hash_source.encode("utf-8", errors="replace")).hexdigest()
    return contexts, content_hash


def _serialize_payload(
    row: MindMap,
    *,
    graph: dict[str, list[dict[str, str]]],
    node_sources: dict[str, str],
) -> dict[str, object]:
    return {
        "id": int(row.id),
        "job_id": int(row.job_id),
        "doc_id": int(row.doc_id) if row.doc_id is not None else None,
        "content_hash": row.content_hash,
        "graph": graph,
        "node_sources": node_sources,
        "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None,
    }


async def generate_or_get_mind_map(
    db: Session,
    *,
    job_id: int,
    doc_id: int | None = None,
) -> tuple[dict[str, object], bool]:
    job = db.query(Job).filter(Job.id == int(job_id)).first()
    if not job:
        raise ValueError("Job not found")

    contexts, content_hash = _collect_contexts_for_hash(db, job=job, doc_id=doc_id)
    cached = (
        db.query(MindMap)
        .filter(
            MindMap.job_id == int(job_id),
            MindMap.doc_id == (int(doc_id) if doc_id is not None else None),
            MindMap.content_hash == content_hash,
        )
        .order_by(MindMap.id.desc())
        .first()
    )
    if cached:
        graph = _graph_from_payload(cached.graph_json)
        node_sources = _choose_source_chunks(graph, contexts)
        return _serialize_payload(cached, graph=graph, node_sources=node_sources), True

    graph = await _generate_graph_from_contexts(contexts)
    if not graph["nodes"]:
        raise ValueError("Mind map generation returned no concepts.")

    row = MindMap(
        job_id=int(job_id),
        doc_id=int(doc_id) if doc_id is not None else None,
        content_hash=content_hash,
        graph_json=graph,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    node_sources = _choose_source_chunks(graph, contexts)
    return _serialize_payload(row, graph=graph, node_sources=node_sources), False


def get_cached_mind_map(
    db: Session,
    *,
    job_id: int,
    doc_id: int | None = None,
) -> dict[str, object]:
    job = db.query(Job).filter(Job.id == int(job_id)).first()
    if not job:
        raise ValueError("Job not found")
    contexts, content_hash = _collect_contexts_for_hash(db, job=job, doc_id=doc_id)
    cached = (
        db.query(MindMap)
        .filter(
            MindMap.job_id == int(job_id),
            MindMap.doc_id == (int(doc_id) if doc_id is not None else None),
            MindMap.content_hash == content_hash,
        )
        .order_by(MindMap.id.desc())
        .first()
    )
    if not cached:
        raise ValueError("Mind map not found for current document content.")
    graph = _graph_from_payload(cached.graph_json)
    node_sources = _choose_source_chunks(graph, contexts)
    return _serialize_payload(cached, graph=graph, node_sources=node_sources)
