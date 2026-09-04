"""Base importer utilities shared across platforms."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImportStats:
    parsed: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    messages: int = 0
    code_blocks: int = 0
    attachments: int = 0
    malformed: int = 0
    languages: dict[str, int] = field(default_factory=dict)


CODE_BLOCK_RE = re.compile(
    r"```(\w*)\n(.*?)```", re.DOTALL
)


def extract_code_blocks(content: str) -> list[tuple[str, str]]:
    """Extract fenced code blocks from markdown content.

    Returns list of (language, code) tuples.
    """
    blocks = []
    for match in CODE_BLOCK_RE.finditer(content):
        lang = match.group(1) or "unknown"
        code = match.group(2).strip()
        if code:
            blocks.append((lang, code))
    return blocks


def make_message_id(conv_id: str, index: int) -> str:
    """Generate a deterministic message ID."""
    return hashlib.sha256(f"{conv_id}:{index}".encode()).hexdigest()[:16]


def insert_conversation(
    conn: sqlite3.Connection,
    *,
    id: str,
    title: Optional[str],
    platform: str,
    model: Optional[str],
    created_at: Optional[str],
    updated_at: Optional[str] = None,
    language: Optional[str] = None,
    update: bool = False,
) -> bool:
    """Insert or skip a conversation. Returns True if inserted/updated."""
    existing = conn.execute(
        "SELECT id FROM conversations WHERE id = ?", (id,)
    ).fetchone()

    if existing:
        if update:
            conn.execute(
                """UPDATE conversations
                   SET title=?, model=?, updated_at=?, language=?
                   WHERE id=?""",
                (title, model, updated_at, language, id),
            )
            return True
        return False

    conn.execute(
        """INSERT INTO conversations(id, title, platform, model, created_at, updated_at, language)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (id, title, platform, model, created_at, updated_at, language),
    )
    return True


def upsert_conversation(
    conn: sqlite3.Connection,
    *,
    id: str,
    title: Optional[str],
    platform: str,
    model: Optional[str],
    created_at: Optional[str],
    updated_at: Optional[str] = None,
    project: Optional[str] = None,
    source_path: Optional[str] = None,
) -> str:
    """Insert a conversation, or refresh its mutable metadata if it exists.

    For append-only sources — a Claude Code session log grows as the session
    runs — re-import must *add* messages, never replace them. So unlike
    :func:`insert_conversation` with ``update=True``, this never touches the
    message rows: it only moves ``updated_at`` forward and refreshes the
    title/model that a longer session may have changed.

    Returns ``"new"`` or ``"updated"``.
    """
    existing = conn.execute(
        "SELECT id FROM conversations WHERE id = ?", (id,)
    ).fetchone()

    if existing:
        conn.execute(
            """UPDATE conversations
               SET title=?, model=?, updated_at=?, project=?, source_path=?
               WHERE id=?""",
            (title, model, updated_at, project, source_path, id),
        )
        return "updated"

    conn.execute(
        """INSERT INTO conversations(
               id, title, platform, model, created_at, updated_at, project, source_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (id, title, platform, model, created_at, updated_at, project, source_path),
    )
    return "new"


def insert_attachment(
    conn: sqlite3.Connection,
    *,
    conv_id: str,
    msg_id: Optional[str],
    type: str,
    filename: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """Record an attachment (an image block, an uploaded file)."""
    conn.execute(
        """INSERT INTO attachments(conv_id, msg_id, type, filename, path)
           VALUES (?, ?, ?, ?, ?)""",
        (conv_id, msg_id, type, filename, path),
    )


# ── Rich-block rendering ─────────────────────────────────────────────────────
#
# Shared by both Claude importers so a web-export transcript and a Claude Code
# transcript read the same way in ``ricoeur show`` and the TUI.


def render_thinking(text: str) -> str:
    """Render a thinking block as a Markdown blockquote so it reads as a
    de-emphasized aside (Rich draws a dim left bar) clearly separated from
    Claude's actual response."""
    text = text.strip()
    if not text:
        return ""
    quoted = "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())
    return f"> 💭 *Thinking…*\n>\n{quoted}"


def render_tool_block(
    name: str,
    body: str,
    *,
    lang: str = "",
    title: Optional[str] = None,
) -> str:
    """Render tool activity that carries real content as a labelled, fenced
    code block — fenced so :func:`extract_code_blocks` picks it up and it
    becomes searchable via ``ricoeur search --code``."""
    if not body or not body.strip():
        return ""
    label = f"🛠️ **Tool · {name}**"
    if title:
        label += f" — {title}"
    return f"{label}\n\n```{lang}\n{body.strip()}\n```"


def insert_message(
    conn: sqlite3.Connection,
    *,
    id: str,
    conv_id: str,
    role: str,
    content: str,
    timestamp: Optional[str] = None,
    content_type: str = "text",
    token_count: Optional[int] = None,
) -> tuple[bool, int]:
    """Insert a message and extract code blocks.

    Returns ``(inserted, code_blocks)``. Existing message IDs are ignored
    rather than replaced, which is what makes re-importing an append-only
    source (a live session log) idempotent — and why callers need to know
    whether a row was actually added.
    """
    cur = conn.execute(
        """INSERT OR IGNORE INTO messages(id, conv_id, role, content, timestamp, content_type, token_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (id, conv_id, role, content, timestamp, content_type, token_count),
    )
    if not cur.rowcount:
        # Already archived — don't duplicate its code blocks.
        return False, 0

    # Extract and insert code blocks
    blocks = 0
    if content:
        for lang, code in extract_code_blocks(content):
            conn.execute(
                "INSERT INTO code_blocks(msg_id, language, code) VALUES (?, ?, ?)",
                (id, lang, code),
            )
            blocks += 1
    return True, blocks


def delete_conversation_messages(conn: sqlite3.Connection, conv_id: str) -> None:
    """Remove a conversation's messages (and their code blocks) for a clean
    re-import. The FTS index stays in sync via the ``messages`` delete trigger.
    """
    msg_ids = [
        r[0] for r in conn.execute(
            "SELECT id FROM messages WHERE conv_id = ?", (conv_id,)
        )
    ]
    if msg_ids:
        conn.executemany(
            "DELETE FROM code_blocks WHERE msg_id = ?", [(m,) for m in msg_ids]
        )
    conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))


def update_conversation_counts(conn: sqlite3.Connection) -> None:
    """Update message_count on all conversations."""
    conn.execute(
        """UPDATE conversations SET message_count = (
               SELECT COUNT(*) FROM messages WHERE messages.conv_id = conversations.id
           )"""
    )
