from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - test/runtime parity
    litellm = None  # type: ignore[assignment]

from ..database import SessionLocal
from ..models.interview import InterviewKnowledgeDocument
from ..models.practice import PracticeQuestion
from ..models.score import Job
from ..models.study import MindMap, MindMapJob, MindMapNodeInfo
from . import interview_docs, llm as llm_service

logger = logging.getLogger(__name__)

MAX_NODES = 40
MAX_EDGES = 80
MAX_CONTEXT_CHUNKS = 12
BATCH_SIZE = 10
MIN_NODE_LABEL_LENGTH = 2
MIN_NODE_TOKEN_LENGTH = 3

MINDMAP_PROMPT = """You are an interview study assistant building a hierarchical mind map.

Extract a concept graph from the provided context snippets.
Return exactly one JSON object with shape:
{
  "nodes": [{"id":"string","label":"string","group":"string"}],
  "edges": [{"source":"string","target":"string","label":"string"}]
}

Structure rules (CRITICAL):
- The FIRST node must be the central root topic — the single overarching theme of the content.
  Give it id "root" and group "root".
- Then create 3–6 main category nodes (group "main"), each connected FROM root with an edge.
- Each main category should have 2–4 specific subtopic/detail nodes (group "detail") branching from it.
- Every non-root node must have at least one incoming edge from its parent.
- Do NOT create cycles or edges that skip levels (root→detail is allowed only if no suitable main exists).
- Maximum 15 nodes total (1 root + 4–5 main + remaining as detail).
- Keep labels short (2–5 words).
- Use stable lowercase ids with underscores (example: "distributed_systems").
- Prefer relationship labels like "covers", "includes", "requires", "leads_to".
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
    lowered_snippets = [s.lower() for s in context_snippets]
    output: dict[str, str] = {}
    for node in graph.get("nodes", []):
        node_id = _normalize_text(node.get("id"))
        label = _normalize_text(node.get("label"))
        if not node_id or not label:
            continue
        tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", label.lower()) if len(token) >= MIN_NODE_TOKEN_LENGTH]
        matched = ""
        for snippet, snippet_lower in zip(context_snippets, lowered_snippets):
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


def _eligible_job_documents(db: Session, *, job_id: int | None, doc_id: int | None) -> list[InterviewKnowledgeDocument]:
    if doc_id is not None:
        docs = (
            db.query(InterviewKnowledgeDocument)
            .filter(InterviewKnowledgeDocument.id == int(doc_id))
            .all()
        )
        return docs

    base_filter = [
        InterviewKnowledgeDocument.status != "failed",
        InterviewKnowledgeDocument.parsed_text.isnot(None),
        InterviewKnowledgeDocument.parsed_text != "",
    ]

    if job_id is not None:
        # Global docs + job-specific docs
        scope_filter = (
            (InterviewKnowledgeDocument.owner_type == "global")
            | (
                (InterviewKnowledgeDocument.owner_type == "job")
                & (InterviewKnowledgeDocument.job_id == int(job_id))
            )
        )
    else:
        # Global docs only
        scope_filter = (InterviewKnowledgeDocument.owner_type == "global")

    return (
        db.query(InterviewKnowledgeDocument)
        .filter(*base_filter, scope_filter)
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
        node_sources = cached.graph_json.get("node_sources", {}) if isinstance(cached.graph_json, dict) else {}
        return _serialize_payload(cached, graph=graph, node_sources=node_sources), True

    graph = await _generate_graph_from_contexts(contexts)
    if not graph["nodes"]:
        raise ValueError("Mind map generation returned no concepts.")

    node_sources = _choose_source_chunks(graph, contexts)
    stored_json = {**graph, "node_sources": node_sources}
    row = MindMap(
        job_id=int(job_id),
        doc_id=int(doc_id) if doc_id is not None else None,
        content_hash=content_hash,
        graph_json=stored_json,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # Another concurrent request already inserted the same (job_id, doc_id, content_hash).
        row = (
            db.query(MindMap)
            .filter(
                MindMap.job_id == int(job_id),
                MindMap.doc_id == (int(doc_id) if doc_id is not None else None),
                MindMap.content_hash == content_hash,
            )
            .order_by(MindMap.id.desc())
            .first()
        )
        if not row:
            raise ValueError("Mind map could not be saved.")
        graph = _graph_from_payload(row.graph_json)
        node_sources = row.graph_json.get("node_sources", {}) if isinstance(row.graph_json, dict) else {}
        return _serialize_payload(row, graph=graph, node_sources=node_sources), True
    db.refresh(row)
    return _serialize_payload(row, graph=graph, node_sources=node_sources), False


def get_cached_mind_map(
    db: Session,
    *,
    job_id: int,
    doc_id: int | None = None,
) -> dict[str, object]:
    if not db.query(Job).filter(Job.id == int(job_id)).first():
        raise ValueError("Job not found")
    cached = (
        db.query(MindMap)
        .filter(
            MindMap.job_id == int(job_id),
            MindMap.doc_id == (int(doc_id) if doc_id is not None else None),
        )
        .order_by(MindMap.id.desc())
        .first()
    )
    if not cached:
        raise ValueError("Mind map not found for current document content.")
    graph = _graph_from_payload(cached.graph_json)
    node_sources = cached.graph_json.get("node_sources", {}) if isinstance(cached.graph_json, dict) else {}
    payload = _serialize_payload(cached, graph=graph, node_sources=node_sources)
    # Look up the latest completed MindMapJob so the frontend can restore task_id for chat
    latest_job = (
        db.query(MindMapJob)
        .filter(
            MindMapJob.job_id == int(job_id),
            MindMapJob.doc_id == (int(doc_id) if doc_id is not None else None),
            MindMapJob.status == "done",
        )
        .order_by(MindMapJob.id.desc())
        .first()
    )
    payload["task_id"] = latest_job.task_id if latest_job else None
    return payload


# ── New async job-based mind map functions ────────────────────────────────────


def _build_chunk_page_map(db: Session, document_id: int) -> dict[int, int | None]:
    """Return {chunk_index: page_number} for a document's embedded chunks."""
    rows = (
        db.query(PracticeQuestion.source_window, PracticeQuestion.chunk_page)
        .filter(
            PracticeQuestion.source_table == interview_docs.DOC_TABLE_NAME,
            PracticeQuestion.source_id == document_id,
        )
        .all()
    )
    result: dict[int, int | None] = {}
    for sw, cp in rows:
        idx = interview_docs.chunk_index_from_source_window(sw)
        if idx is not None:
            result[idx] = cp
    return result


def _collect_all_contexts(db: Session, *, job_id: int, doc_id: int | None) -> list[dict[str, str]]:
    selected_docs = _eligible_job_documents(db, job_id=job_id, doc_id=doc_id)
    if not selected_docs:
        raise ValueError("No interview documents available for this selection.")
    contexts: list[dict[str, str]] = []
    docs_to_process = [selected_docs[0]] if doc_id is not None else selected_docs
    for document in docs_to_process:
        chunk_page_map = _build_chunk_page_map(db, document.id)
        for index, snippet in enumerate(interview_docs.chunk_interview_text(document.parsed_text)):
            page_num = chunk_page_map.get(index)
            ctx: dict[str, str] = {
                "filename": document.source_filename or "",
                "snippet": snippet,
                "chunk_id": f"{document.id}:chunk:{index}",
                "doc_id": str(document.id),
                "page_number": str(page_num) if page_num is not None else "",
            }
            contexts.append(ctx)
    if not contexts:
        raise ValueError("No interview document content available for mind map extraction.")
    return contexts


def _merge_partial_graphs(partial_graphs: list[dict[str, list[dict[str, str]]]]) -> dict[str, list[dict[str, str]]]:
    seen_node_ids: set[str] = set()
    merged_nodes: list[dict[str, str]] = []
    for graph in partial_graphs:
        for node in graph.get("nodes", []):
            node_id = _normalize_text(node.get("id"))
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            merged_nodes.append(node)
            if len(merged_nodes) >= MAX_NODES:
                break
        if len(merged_nodes) >= MAX_NODES:
            break

    seen_edges: set[tuple[str, str, str]] = set()
    merged_edges: list[dict[str, str]] = []
    for graph in partial_graphs:
        for edge in graph.get("edges", []):
            source = _normalize_text(edge.get("source"))
            target = _normalize_text(edge.get("target"))
            label = _normalize_text(edge.get("label"))
            if not source or not target:
                continue
            key = (source, target, label.lower())
            if key in seen_edges:
                continue
            seen_edges.add(key)
            merged_edges.append(edge)
            if len(merged_edges) >= MAX_EDGES:
                break
        if len(merged_edges) >= MAX_EDGES:
            break

    # Re-filter edges to only reference surviving node ids
    surviving_ids = {node["id"] for node in merged_nodes}
    merged_edges = [
        edge for edge in merged_edges
        if edge.get("source") in surviving_ids and edge.get("target") in surviving_ids
    ]

    return {"nodes": merged_nodes, "edges": merged_edges}


async def _run_merge_pass(merged_graph: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    if litellm is None:
        import litellm as runtime_litellm
        completion_client = runtime_litellm
    else:
        completion_client = litellm

    model, kwargs = llm_service._get_litellm_model()  # type: ignore[attr-defined]
    nodes_list = [{"id": node["id"], "label": node.get("label", "")} for node in merged_graph.get("nodes", [])]
    existing_edges = merged_graph.get("edges", [])

    prompt = "\n\n".join([
        "You are given a list of concept nodes extracted from a document.",
        "Identify important relationships BETWEEN nodes that are not already captured.",
        'Return exactly one JSON object: {"edges": [{"source": "node_id", "target": "node_id", "label": "relationship"}]}',
        "Only return edges. Do not add new nodes. Node ids must be from the provided list.",
        "Return JSON only.",
        f"nodes: {json.dumps(nodes_list)}",
        f"existing_edges: {json.dumps(existing_edges)}",
    ])

    response = await completion_client.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    payload_text = _extract_json_payload(content) or ""
    if not payload_text.strip():
        return []
    try:
        parsed = json.loads(payload_text)
    except Exception:
        try:
            parsed = json.loads(llm_service._sanitize_json_token_stream(payload_text))  # type: ignore[attr-defined]
        except Exception:
            return []
    if not isinstance(parsed, dict):
        return []
    raw_edges = parsed.get("edges")
    if not isinstance(raw_edges, list):
        return []
    return [edge for edge in raw_edges if isinstance(edge, dict)]


async def run_mindmap_job(task_id: str, *, job_id: int | None, doc_id: int | None) -> None:
    db = SessionLocal()
    try:
        row = db.query(MindMapJob).filter(MindMapJob.task_id == task_id).first()
        if row is None:
            logger.error("run_mindmap_job: task_id=%s not found in DB", task_id)
            return
        row.status = "processing"
        row.updated_at = datetime.now(timezone.utc)
        db.commit()

        contexts = _collect_all_contexts(db, job_id=job_id, doc_id=doc_id)

        # Split into batches
        batches: list[list[dict[str, str]]] = []
        for start in range(0, len(contexts), BATCH_SIZE):
            batches.append(contexts[start:start + BATCH_SIZE])

        partial_graphs: list[dict[str, list[dict[str, str]]]] = []
        merged_graph: dict[str, list[dict[str, str]]] = {"nodes": [], "edges": []}
        for batch_contexts in batches:
            batch_graph = await _generate_graph_from_contexts(batch_contexts)
            partial_graphs.append(batch_graph)
            merged_graph = _merge_partial_graphs(partial_graphs)
            row.graph_json = merged_graph
            row.updated_at = datetime.now(timezone.utc)
            db.commit()

        # Merge pass for cross-batch edges
        additional_edges = await _run_merge_pass(merged_graph)
        if additional_edges:
            merged_graph = _merge_partial_graphs([merged_graph, {"nodes": [], "edges": additional_edges}])

        row.status = "done"
        row.graph_json = merged_graph
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        logger.exception("run_mindmap_job failed for task_id=%s: %s", task_id, exc)
        try:
            row = db.query(MindMapJob).filter(MindMapJob.task_id == task_id).first()
            if row is not None:
                row.status = "failed"
                row.error = str(exc)
                row.updated_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            logger.exception("run_mindmap_job: could not persist failure for task_id=%s", task_id)
    finally:
        db.close()


def create_mindmap_job(db: Session, *, job_id: int | None, doc_id: int | None) -> str:
    if job_id is not None and not db.query(Job).filter(Job.id == int(job_id)).first():
        raise ValueError("Job not found")
    task_id = str(uuid.uuid4())
    row = MindMapJob(
        task_id=task_id,
        job_id=int(job_id) if job_id is not None else None,
        doc_id=int(doc_id) if doc_id is not None else None,
        status="pending",
        graph_json={},
    )
    db.add(row)
    db.commit()
    return task_id


def get_latest_mindmap_job(db: Session, *, job_id: int | None, doc_id: int | None) -> dict[str, object] | None:
    """Return the most recent completed MindMapJob matching the given filters."""
    query = db.query(MindMapJob).filter(MindMapJob.status == "done")
    if job_id is not None:
        query = query.filter(MindMapJob.job_id == int(job_id))
    if doc_id is not None:
        query = query.filter(MindMapJob.doc_id == int(doc_id))
    row = query.order_by(MindMapJob.id.desc()).first()
    if row is None:
        return None
    graph: dict | None = None
    if isinstance(row.graph_json, dict) and row.graph_json.get("nodes"):
        graph = row.graph_json
    return {
        "task_id": row.task_id,
        "status": row.status,
        "error": row.error,
        "graph": graph,
    }


def get_mindmap_job_status(db: Session, *, task_id: str) -> dict[str, object]:
    row = db.query(MindMapJob).filter(MindMapJob.task_id == task_id).first()
    if row is None:
        raise ValueError("Mind map job not found")
    graph: dict | None = None
    if isinstance(row.graph_json, dict) and row.graph_json.get("nodes"):
        graph = row.graph_json
    return {
        "task_id": row.task_id,
        "status": row.status,
        "error": row.error,
        "graph": graph,
    }


def _score_contexts_for_query(contexts: list[dict[str, str]], query: str, top_k: int = 6) -> list[dict[str, str]]:
    """Return top_k contexts most relevant to the query by keyword overlap."""
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) >= MIN_NODE_TOKEN_LENGTH]
    if not tokens:
        return contexts[:top_k]
    scored: list[tuple[int, dict[str, str]]] = []
    for ctx in contexts:
        snippet_lower = ctx.get("snippet", "").lower()
        score = sum(1 for t in tokens if t in snippet_lower)
        scored.append((score, ctx))
    scored.sort(key=lambda x: -x[0])
    top = [ctx for _, ctx in scored[:top_k]]
    # If none matched, fall back to first top_k
    if all(s == 0 for s, _ in scored[:top_k]):
        return contexts[:top_k]
    return top


async def get_or_generate_node_info(db: Session, *, task_id: str, node_id: str) -> dict[str, object]:
    row = db.query(MindMapJob).filter(MindMapJob.task_id == task_id).first()
    if row is None:
        raise ValueError("Mind map job not found")

    cached_info = (
        db.query(MindMapNodeInfo)
        .filter(
            MindMapNodeInfo.mind_map_job_id == row.id,
            MindMapNodeInfo.node_id == node_id,
        )
        .first()
    )
    if cached_info is not None:
        sources = cached_info.sources_json if isinstance(cached_info.sources_json, list) else []
        return {"encyclopedic": cached_info.encyclopedic, "interview_prep": cached_info.interview_prep, "sources": sources}

    # Find node label
    graph = row.graph_json if isinstance(row.graph_json, dict) else {}
    node_label = ""
    for node in graph.get("nodes", []):
        if node.get("id") == node_id:
            node_label = node.get("label", node_id)
            break
    if not node_label:
        node_label = node_id

    contexts: list[dict[str, str]] = []
    try:
        contexts = _collect_all_contexts(db, job_id=row.job_id, doc_id=int(row.doc_id) if row.doc_id is not None else None)
    except ValueError:
        pass

    source_contexts = _score_contexts_for_query(contexts, node_label, top_k=4) if contexts else []

    result = await _generate_node_info_llm(node_label, source_contexts)

    node_info = MindMapNodeInfo(
        mind_map_job_id=row.id,
        node_id=node_id,
        encyclopedic=result["encyclopedic"],
        interview_prep=result["interview_prep"],
        sources_json=result["sources"],
    )
    db.add(node_info)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        cached_info = (
            db.query(MindMapNodeInfo)
            .filter(
                MindMapNodeInfo.mind_map_job_id == row.id,
                MindMapNodeInfo.node_id == node_id,
            )
            .first()
        )
        if cached_info is not None:
            sources = cached_info.sources_json if isinstance(cached_info.sources_json, list) else []
            return {"encyclopedic": cached_info.encyclopedic, "interview_prep": cached_info.interview_prep, "sources": sources}

    return result


async def _generate_node_info_llm(node_label: str, source_contexts: list[dict[str, str]]) -> dict[str, object]:
    if litellm is None:
        import litellm as runtime_litellm
        completion_client = runtime_litellm
    else:
        completion_client = litellm

    model, kwargs = llm_service._get_litellm_model()  # type: ignore[attr-defined]

    sources: list[dict[str, object]] = [
        {
            "index": i,
            "filename": ctx.get("filename", f"document-{i}"),
            "snippet": ctx.get("snippet", "")[:400],
            "doc_id": int(ctx["doc_id"]) if ctx.get("doc_id") else None,
            "page_number": int(ctx["page_number"]) if ctx.get("page_number") else None,
        }
        for i, ctx in enumerate(source_contexts, start=1)
    ]

    excerpts_block = "\n\n".join(
        f"[{src['index']}] {src['filename']}:\n{src['snippet']}" for src in sources
    ) or "No document excerpts available."

    prompt = "\n\n".join([
        "You are an expert technical interview coach.",
        "Given a concept and relevant document excerpts, produce two things:",
        "1. encyclopedic: A clear, thorough explanation of what this concept is, how it works, and where it is used. "
        "Ground it in the provided excerpts and include inline citations like [1] or [2] where applicable.",
        "2. interview_prep: How to answer questions about this concept in a technical interview. "
        "Include key talking points, common follow-up questions, and tradeoffs. Use bullet points (• prefix).",
        'Return exactly one JSON object:\n{"encyclopedic": "...", "interview_prep": "..."}',
        "Return JSON only (no markdown).",
        f"concept: {node_label}",
        f"document_excerpts:\n{excerpts_block}",
    ])

    response = await completion_client.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    payload_text = _extract_json_payload(content) or ""
    if not payload_text.strip():
        return {"encyclopedic": "", "interview_prep": "", "sources": sources}
    try:
        parsed = json.loads(payload_text)
    except Exception:
        try:
            parsed = json.loads(llm_service._sanitize_json_token_stream(payload_text))  # type: ignore[attr-defined]
        except Exception:
            parsed = {}
    if not isinstance(parsed, dict):
        return {"encyclopedic": "", "interview_prep": "", "sources": sources}
    return {
        "encyclopedic": str(parsed.get("encyclopedic") or ""),
        "interview_prep": str(parsed.get("interview_prep") or ""),
        "sources": sources,
    }


async def chat_with_mindmap(db: Session, *, task_id: str, message: str) -> dict[str, object]:
    row = db.query(MindMapJob).filter(MindMapJob.task_id == task_id).first()
    if row is None:
        raise ValueError("Mind map job not found")

    contexts: list[dict[str, str]] = []
    try:
        contexts = _collect_all_contexts(db, job_id=row.job_id, doc_id=int(row.doc_id) if row.doc_id is not None else None)
    except ValueError:
        pass

    if not contexts:
        return {"answer": "No document content is available for this mind map.", "sources": []}

    top_contexts = _score_contexts_for_query(contexts, message, top_k=6)
    return await _generate_chat_answer(message, top_contexts)


async def _generate_chat_answer(message: str, contexts: list[dict[str, str]]) -> dict[str, object]:
    if litellm is None:
        import litellm as runtime_litellm
        completion_client = runtime_litellm
    else:
        completion_client = litellm

    model, kwargs = llm_service._get_litellm_model()  # type: ignore[attr-defined]

    sources: list[dict[str, object]] = [
        {
            "index": i,
            "filename": ctx.get("filename", f"document-{i}"),
            "snippet": ctx.get("snippet", "")[:400],
            "doc_id": int(ctx["doc_id"]) if ctx.get("doc_id") else None,
            "page_number": int(ctx["page_number"]) if ctx.get("page_number") else None,
        }
        for i, ctx in enumerate(contexts, start=1)
    ]

    excerpts_block = "\n\n".join(
        f"[{src['index']}] {src['filename']}:\n{src['snippet']}" for src in sources
    )

    prompt = "\n\n".join([
        "You are an expert technical interview coach helping a candidate study for interviews.",
        "Answer the question using the provided document excerpts.",
        "Include inline citations like [1] or [2] wherever you use information from a source.",
        "Be thorough but concise. Use bullet points (• prefix) where appropriate.",
        f"question: {message}",
        f"document excerpts:\n{excerpts_block}",
    ])

    response = await completion_client.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
        **kwargs,
    )
    answer = response.choices[0].message.content or ""
    return {"answer": answer.strip(), "sources": sources}
