"""Claude conversation importer.

Handles the JSON export from Claude (Settings > Export my data).
Claude exports are flat arrays of conversations with messages.
"""

from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Optional

from rich.progress import Progress

from .base import (
    ImportStats,
    delete_conversation_messages,
    insert_conversation,
    insert_message,
    make_message_id,
    render_thinking,
    render_tool_block,
    update_conversation_counts,
)

# Claude's export flattens content blocks it can't render (artifacts, the
# code-execution / analysis tool, …) into this literal placeholder inside the
# message's ``text`` field. When we see it, the flattened text is unreliable
# and we reconstruct from the structured ``content`` blocks instead.
UNSUPPORTED_BLOCK = "This block is not supported on your current device yet."

# A fenced block whose only body is the placeholder — strip the whole fence.
_PLACEHOLDER_FENCE_RE = re.compile(
    r"```[^\n]*\n[ \t]*" + re.escape(UNSUPPORTED_BLOCK) + r"[ \t]*\n```\n?"
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

    existed = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()

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

    # On --update, clear the old messages so the re-extracted content
    # (e.g. recovered artifact/code blocks) actually replaces the stale rows.
    if existed and update:
        delete_conversation_messages(conn, conv_id)

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
    """Extract text from a Claude message.

    Claude's export gives each message a flattened ``text`` field, but it mashes
    reasoning, response, and tool activity into one undifferentiated string. We
    rebuild from the structured ``content`` blocks whenever they carry thinking
    or tool activity — so those can be labelled distinctly — or when the
    flattened text is contaminated by the ``UNSUPPORTED_BLOCK`` placeholder.
    Otherwise the clean flattened text is authoritative. As a last resort we
    salvage the flattened text with placeholder noise stripped out.
    """
    text = msg.get("text")
    has_text = bool(text and text.strip())
    content = msg.get("content", "")

    contaminated = has_text and UNSUPPORTED_BLOCK in text
    if _has_rich_blocks(content) or contaminated or not has_text:
        rebuilt = _from_blocks(content)
        if rebuilt.strip():
            return rebuilt

    if has_text and not contaminated:
        return text
    if has_text:
        return _strip_placeholder(text)
    return ""


def _has_rich_blocks(content: Any) -> bool:
    """True if the structured content has thinking or tool blocks worth
    labelling distinctly (so we should rebuild from blocks rather than trust
    the flattened text)."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") in ("thinking", "tool_use")
        for b in content
    )


def _from_blocks(content: Any) -> str:
    """Reconstruct message text from Claude's structured ``content`` blocks."""
    if isinstance(content, str):
        return _strip_placeholder(content)
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(_render_block(block))
    return "\n\n".join(p for p in parts if p and p.strip())


def _render_block(block: dict[str, Any]) -> str:
    btype = block.get("type")
    if btype == "text":
        return _strip_placeholder(block.get("text", ""))
    if btype == "thinking":
        return render_thinking(block.get("thinking", ""))
    if btype == "tool_use":
        return _render_tool_use(block)
    # tool_result and other interface blocks are noise — skip them.
    return ""


def _render_tool_use(block: dict[str, Any]) -> str:
    """Render a tool-use block that carries real content (artifacts, the
    code-execution tool) as a labelled, fenced code block. Interface-only tools
    such as web search carry no code/content and render to nothing.
    """
    inp = block.get("input")
    if not isinstance(inp, dict):
        return ""
    body = inp.get("code") or inp.get("content")
    if not isinstance(body, str) or not body.strip():
        return ""
    name = block.get("name") or "tool"
    lang = inp.get("language") or inp.get("lang") or ""
    title = inp.get("title")
    return render_tool_block(
        name, body, lang=lang, title=title if isinstance(title, str) and title else None
    )


def _strip_placeholder(text: str) -> str:
    """Remove ``UNSUPPORTED_BLOCK`` placeholders (and the empty fences that
    wrap them) from flattened export text."""
    if not text or UNSUPPORTED_BLOCK not in text:
        return text
    text = _PLACEHOLDER_FENCE_RE.sub("", text)
    text = text.replace(UNSUPPORTED_BLOCK, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
