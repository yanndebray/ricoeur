"""ChatGPT conversation importer.

Handles the JSON export from ChatGPT (Settings > Data Controls > Export Data).
The export contains a conversations.json with a tree structure of messages.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import datetime, timezone
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


def import_chatgpt(
    conn: sqlite3.Connection,
    path: Path,
    *,
    update: bool = False,
    since: Optional[str] = None,
    dry_run: bool = False,
    progress: Optional[Progress] = None,
) -> ImportStats:
    """Import conversations from a ChatGPT export."""
    stats = ImportStats()
    data = _load_data(path)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of conversations")

    stats.parsed = len(data)
    task = None
    if progress:
        task = progress.add_task("Importing ChatGPT...", total=len(data))

    for conv_raw in data:
        _import_one(conn, conv_raw, stats, update=update, since=since, dry_run=dry_run)
        if progress and task is not None:
            progress.advance(task)

    if not dry_run:
        update_conversation_counts(conn)
        conn.commit()

    return stats


def _load_data(path: Path) -> list[dict[str, Any]]:
    """Load from .json or .zip file."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            # Look for conversations.json inside the zip
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
    conv_id = conv.get("id", conv.get("conversation_id", ""))
    title = conv.get("title", "Untitled")
    create_time = conv.get("create_time")
    update_time = conv.get("update_time")

    created_at = _ts_to_iso(create_time) if create_time else None
    updated_at = _ts_to_iso(update_time) if update_time else None

    if since and created_at and created_at < since:
        stats.skipped += 1
        return

    # Walk the message tree to find the default model
    model = _extract_model(conv)

    if dry_run:
        stats.new += 1
        return

    inserted = insert_conversation(
        conn,
        id=conv_id,
        title=title,
        platform="chatgpt",
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

    # Walk message tree and insert messages
    messages = _walk_messages(conv)
    for i, msg in enumerate(messages):
        msg_id = msg.get("id") or make_message_id(conv_id, i)
        role = msg.get("role", "unknown")
        content = _extract_content(msg)
        timestamp = _ts_to_iso(msg.get("create_time")) if msg.get("create_time") else created_at

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


def _walk_messages(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the ChatGPT message tree to extract messages in order."""
    mapping = conv.get("mapping", {})
    if not mapping:
        return []

    # Build parent->children map and find root
    children_map: dict[str, list[str]] = {}
    root = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None:
            root = node_id
        else:
            children_map.setdefault(parent, []).append(node_id)

    if root is None:
        return []

    # Walk the tree depth-first, preferring the last child (default branch)
    messages = []
    stack = [root]
    while stack:
        node_id = stack.pop()
        node = mapping.get(node_id, {})
        msg = node.get("message")
        if msg and msg.get("content"):
            author = msg.get("author", {})
            role = author.get("role", "unknown")
            if role in ("user", "assistant", "system", "tool"):
                messages.append({**msg, "role": role})

        # Add children to stack (reversed so first child is processed first)
        kids = children_map.get(node_id, [])
        # Pick the default (last) branch
        if kids:
            stack.append(kids[-1])

    return messages


def _extract_content(msg: dict[str, Any]) -> str:
    """Extract text content from a ChatGPT message."""
    content = msg.get("content", {})
    if isinstance(content, str):
        return content
    parts = content.get("parts", [])
    text_parts = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict) and part.get("content_type") == "text":
            text_parts.append(part.get("text", ""))
    return "\n".join(text_parts)


def _extract_model(conv: dict[str, Any]) -> Optional[str]:
    """Try to extract the model slug from the conversation."""
    # Check top-level
    if "default_model_slug" in conv:
        return conv["default_model_slug"]
    # Walk messages looking for model_slug in metadata
    mapping = conv.get("mapping", {})
    for node in mapping.values():
        msg = node.get("message", {})
        meta = msg.get("metadata", {})
        slug = meta.get("model_slug")
        if slug:
            return slug
    return None


def _ts_to_iso(ts: float | int | None) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
