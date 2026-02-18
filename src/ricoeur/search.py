"""Search layer — FTS5, semantic, and hybrid search."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SearchResult:
    conv_id: str
    title: str
    platform: str
    model: Optional[str]
    created_at: Optional[str]
    language: Optional[str]
    topic_id: Optional[int]
    score: float
    snippet: Optional[str] = None


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    platform: Optional[str] = None,
    lang: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    topic: Optional[int] = None,
    role: Optional[str] = None,
    code: bool = False,
    limit: int = 20,
) -> list[SearchResult]:
    """Full-text search using FTS5."""
    if code:
        return _code_search(conn, query, limit=limit)

    # Build FTS query
    sql = """
        SELECT
            c.id, c.title, c.platform, c.model, c.created_at,
            c.language, c.topic_id,
            rank AS score,
            snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet
        FROM messages_fts
        JOIN conversations c ON messages_fts.conv_id = c.id
        WHERE messages_fts MATCH ?
    """
    params: list = [query]

    if platform:
        sql += " AND c.platform = ?"
        params.append(platform)
    if lang:
        sql += " AND c.language = ?"
        params.append(lang)
    if since:
        sql += " AND c.created_at >= ?"
        params.append(since)
    if until:
        sql += " AND c.created_at <= ?"
        params.append(until)
    if model:
        sql += " AND c.model = ?"
        params.append(model)
    if topic is not None:
        sql += " AND c.topic_id = ?"
        params.append(topic)
    if role:
        sql += " AND messages_fts.role = ?"
        params.append(role)

    # No GROUP BY — FTS5 auxiliary functions (snippet, highlight) cannot be
    # used with GROUP BY.  Fetch extra rows and deduplicate in Python,
    # keeping the best-ranked match per conversation.
    sql += """
        ORDER BY rank
        LIMIT ?
    """
    params.append(limit * 5)

    rows = conn.execute(sql, params).fetchall()
    seen: set[str] = set()
    results: list[SearchResult] = []
    for r in rows:
        cid = r["id"]
        if cid in seen:
            continue
        seen.add(cid)
        results.append(
            SearchResult(
                conv_id=cid,
                title=r["title"],
                platform=r["platform"],
                model=r["model"],
                created_at=r["created_at"],
                language=r["language"],
                topic_id=r["topic_id"],
                score=abs(r["score"]),
                snippet=r["snippet"],
            )
        )
        if len(results) >= limit:
            break
    return results


def _code_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Search within extracted code blocks."""
    sql = """
        SELECT DISTINCT
            c.id, c.title, c.platform, c.model, c.created_at,
            c.language, c.topic_id,
            1.0 AS score
        FROM code_blocks cb
        JOIN messages m ON cb.msg_id = m.id
        JOIN conversations c ON m.conv_id = c.id
        WHERE cb.code LIKE ?
        LIMIT ?
    """
    rows = conn.execute(sql, (f"%{query}%", limit)).fetchall()
    return [
        SearchResult(
            conv_id=r["id"],
            title=r["title"],
            platform=r["platform"],
            model=r["model"],
            created_at=r["created_at"],
            language=r["language"],
            topic_id=r["topic_id"],
            score=r["score"],
        )
        for r in rows
    ]


# ── Embedding helpers ────────────────────────────────────────────────────

# Module-level cache for sentence-transformer models (avoids reloading per query)
_model_cache: dict[str, Any] = {}


def _load_embeddings(home: Path) -> tuple[Any, dict] | tuple[None, None]:
    """Load pre-computed embeddings and metadata.

    Returns (embeddings_ndarray, meta_dict) or (None, None) if unavailable.
    """
    import numpy as np

    embed_dir = home / "embeddings"
    npy_path = embed_dir / "embeddings.npy"
    meta_path = embed_dir / "meta.json"

    if not npy_path.exists() or not meta_path.exists():
        return None, None

    embeddings = np.load(str(npy_path))
    with open(meta_path) as f:
        meta = json.load(f)

    if len(meta.get("conv_ids", [])) != embeddings.shape[0]:
        return None, None

    return embeddings, meta


def _encode_query(query: str, model_spec: str, device: str = "auto") -> Any:
    """Encode a single query string into a normalized embedding vector."""
    import numpy as np
    from .indexer import _parse_model_spec

    backend, model_name = _parse_model_spec(model_spec)

    if backend == "ollama":
        import ollama

        response = ollama.embed(model=model_name, input=[f"search_query: {query}"])
        vec = np.array(response["embeddings"][0], dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer

        if model_spec not in _model_cache:
            _model_cache[model_spec] = SentenceTransformer(
                model_name, device=device if device != "auto" else None
            )
        model = _model_cache[model_spec]
        vec = model.encode(query, normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32)

    # Safety-normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ── Semantic search ──────────────────────────────────────────────────────


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    home: Path,
    *,
    model_spec: str,
    platform: Optional[str] = None,
    lang: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    topic: Optional[int] = None,
    limit: int = 20,
    device: str = "auto",
) -> list[SearchResult]:
    """Search by cosine similarity against pre-computed embeddings."""
    import numpy as np

    embeddings, meta = _load_embeddings(home)
    if embeddings is None or meta is None:
        raise RuntimeError(
            "No embeddings found. Run 'ricoeur index --embeddings' first."
        )

    query_vec = _encode_query(query, model_spec, device)

    # Cosine similarity (dot product of L2-normalized vectors)
    scores = embeddings @ query_vec
    candidate_count = min(limit * 5, len(scores))
    if candidate_count >= len(scores):
        # All scores fit — just argsort
        top_indices = np.argsort(-scores)[:candidate_count]
    else:
        top_indices = np.argpartition(-scores, candidate_count)[:candidate_count]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

    conv_ids = meta["conv_ids"]
    candidates = [(conv_ids[i], float(scores[i])) for i in top_indices]

    # Fetch metadata from SQLite for all candidates in one query
    id_list = [c[0] for c in candidates]
    placeholders = ",".join("?" * len(id_list))

    sql = f"""
        SELECT id, title, platform, model, created_at, language, topic_id
        FROM conversations
        WHERE id IN ({placeholders})
    """
    rows = conn.execute(sql, id_list).fetchall()
    row_map = {r["id"]: r for r in rows}

    # Filter and build results (preserve score-sorted order)
    results: list[SearchResult] = []
    for cid, score in candidates:
        r = row_map.get(cid)
        if r is None:
            continue
        if platform and r["platform"] != platform:
            continue
        if lang and r["language"] != lang:
            continue
        if since and (r["created_at"] or "") < since:
            continue
        if until and (r["created_at"] or "") > until:
            continue
        if model and r["model"] != model:
            continue
        if topic is not None and r["topic_id"] != topic:
            continue
        results.append(
            SearchResult(
                conv_id=cid,
                title=r["title"],
                platform=r["platform"],
                model=r["model"],
                created_at=r["created_at"],
                language=r["language"],
                topic_id=r["topic_id"],
                score=score,
            )
        )
        if len(results) >= limit:
            break

    return results


# ── Hybrid search (RRF) ─────────────────────────────────────────────────


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    home: Path,
    *,
    model_spec: str,
    platform: Optional[str] = None,
    lang: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    topic: Optional[int] = None,
    role: Optional[str] = None,
    code: bool = False,
    limit: int = 20,
    device: str = "auto",
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of FTS5 + semantic search (k=60)."""
    k = 60
    overfetch = limit * 5

    # FTS leg — always available
    fts_results = fts_search(
        conn,
        query,
        platform=platform,
        lang=lang,
        since=since,
        until=until,
        model=model,
        topic=topic,
        role=role,
        code=code,
        limit=overfetch,
    )

    # Semantic leg — graceful degradation
    try:
        sem_results = semantic_search(
            conn,
            query,
            home,
            model_spec=model_spec,
            platform=platform,
            lang=lang,
            since=since,
            until=until,
            model=model,
            topic=topic,
            limit=overfetch,
            device=device,
        )
    except RuntimeError:
        # No embeddings — fall back to FTS-only
        return fts_results[:limit]

    # Build rank lookup dicts
    fts_rank = {r.conv_id: rank for rank, r in enumerate(fts_results)}
    sem_rank = {r.conv_id: rank for rank, r in enumerate(sem_results)}

    # Collect all candidate results (prefer FTS object for snippet)
    result_map: dict[str, SearchResult] = {}
    for r in fts_results:
        result_map[r.conv_id] = r
    for r in sem_results:
        if r.conv_id not in result_map:
            result_map[r.conv_id] = r

    # Compute RRF score
    rrf_scores: dict[str, float] = {}
    all_ids = set(fts_rank) | set(sem_rank)
    for cid in all_ids:
        score = 0.0
        if cid in fts_rank:
            score += 1.0 / (k + fts_rank[cid])
        if cid in sem_rank:
            score += 1.0 / (k + sem_rank[cid])
        rrf_scores[cid] = score

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results: list[SearchResult] = []
    for cid in sorted_ids[:limit]:
        r = result_map[cid]
        results.append(
            SearchResult(
                conv_id=r.conv_id,
                title=r.title,
                platform=r.platform,
                model=r.model,
                created_at=r.created_at,
                language=r.language,
                topic_id=r.topic_id,
                score=rrf_scores[cid],
                snippet=r.snippet,
            )
        )
    return results


# ── Dispatch ─────────────────────────────────────────────────────────────


def search_dispatch(
    conn: sqlite3.Connection,
    query: str,
    home: Path,
    *,
    semantic: bool = False,
    keyword: bool = False,
    model_spec: str = "",
    platform: Optional[str] = None,
    lang: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    topic: Optional[int] = None,
    role: Optional[str] = None,
    code: bool = False,
    limit: int = 20,
    device: str = "auto",
) -> tuple[list[SearchResult], str]:
    """Route to the appropriate search backend.

    Returns (results, effective_mode) where mode is 'keyword', 'semantic', or 'hybrid'.
    """
    # FTS-only flags force keyword mode
    if keyword or code or role:
        results = fts_search(
            conn,
            query,
            platform=platform,
            lang=lang,
            since=since,
            until=until,
            model=model,
            topic=topic,
            role=role,
            code=code,
            limit=limit,
        )
        return results, "keyword"

    if semantic:
        results = semantic_search(
            conn,
            query,
            home,
            model_spec=model_spec,
            platform=platform,
            lang=lang,
            since=since,
            until=until,
            model=model,
            topic=topic,
            limit=limit,
            device=device,
        )
        return results, "semantic"

    # Default: hybrid if embeddings exist, else keyword
    embeddings, _ = _load_embeddings(home)
    if embeddings is not None:
        results = hybrid_search(
            conn,
            query,
            home,
            model_spec=model_spec,
            platform=platform,
            lang=lang,
            since=since,
            until=until,
            model=model,
            topic=topic,
            role=role,
            code=code,
            limit=limit,
            device=device,
        )
        return results, "hybrid"

    results = fts_search(
        conn,
        query,
        platform=platform,
        lang=lang,
        since=since,
        until=until,
        model=model,
        topic=topic,
        role=role,
        code=code,
        limit=limit,
    )
    return results, "keyword"
