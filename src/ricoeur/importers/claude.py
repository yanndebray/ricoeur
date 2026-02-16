"""Claude conversation importer.

Handles the JSON export from Claude (Settings > Export my data).
Claude exports are flat arrays of conversations with messages.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Optional

from rich.progress import Progress

from .base import (
    ImportStats,
    insert_conversation,
    insert_message,
    make_message_id,
    update_conversation_counts,
)


def import_claude(
    conn: sqlite3.Connection,
    path: Path,
    *,
    update: bool = False,
    since: Optional[str] = None,
    dry_run: bool = False,
    progress: Optional[Progress] = None,
) -> ImportStats:
    """Import conversations from a Claude export."""
    stats = ImportStats()
    data = _load_data(path)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of conversations")

    stats.parsed = len(data)
    task = None
    if progress:
        task = progress.add_task("Importing Claude...", total=len(data))

    for conv_raw in data:
        _import_one(conn, conv_raw, stats, update=update, since=since, dry_run=dry_run)
        if progress and task is not None:
            progress.advance(task)

    if not dry_run:
        update_conversation_counts(conn)
        conn.commit()

    return stats


def _load_data(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith("conversations.json")]
            if not names:
                raise FileNotFoundError("No conversations.json found in zip")
            with zf.open(names[0]) as f:
                return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def _import_one(
    conn: sqlite3.Connection,
    conv: dict[str, Any],
    stats: ImportStats,
    *,
    update: bool,
    since: Optional[str],
    dry_run: bool,
) -> None:
    conv_id = conv.get("uuid", conv.get("id", ""))
    title = conv.get("name", conv.get("title", "Untitled"))
    created_at = conv.get("created_at")
    updated_at = conv.get("updated_at")

    if since and created_at and created_at < since:
        stats.skipped += 1
        return

    model = conv.get("model")

    if dry_run:
        stats.new += 1
        return

    inserted = insert_conversation(
        conn,
        id=conv_id,
        title=title,
        platform="claude",
        model=model,
        created_at=created_at,
        updated_at=updated_at,
        update=update,
    )

    if inserted:
        stats.new += 1
    else:
        stats.skipped += 1
        return

    # Claude exports have a flat "chat_messages" array
    messages = conv.get("chat_messages", conv.get("messages", []))
    for i, msg in enumerate(messages):
        msg_id = msg.get("uuid", msg.get("id")) or make_message_id(conv_id, i)
        role = msg.get("sender", msg.get("role", "unknown"))
        # Normalize Claude's "human"/"assistant" roles
        if role == "human":
            role = "user"
        content = _extract_content(msg)
        timestamp = msg.get("created_at", msg.get("timestamp"))

        if content:
            insert_message(
                conn,
                id=msg_id,
                conv_id=conv_id,
                role=role,
                content=content,
                timestamp=timestamp,
            )
            stats.messages += 1


def _extract_content(msg: dict[str, Any]) -> str:
    """Extract text from a Claude message."""
    # Claude messages can have "text" directly or "content" as a list of blocks
    text = msg.get("text")
    if text:
        return text

    content = msg.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)

    return ""
