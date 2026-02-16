"""Search layer — FTS5 full-text search."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional


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

    sql += """
        GROUP BY c.id
        ORDER BY rank
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(
            conv_id=r["id"],
            title=r["title"],
            platform=r["platform"],
            model=r["model"],
            created_at=r["created_at"],
            language=r["language"],
            topic_id=r["topic_id"],
            score=abs(r["score"]),
            snippet=r["snippet"],
        )
        for r in rows
    ]


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
