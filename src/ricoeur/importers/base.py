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


def insert_message(
    conn: sqlite3.Connection,
    *,
    id: str,
    conv_id: str,
    role: str,
    content: str,
    timestamp: Optional[str] = None,
    content_type: str = "text",
) -> None:
    """Insert a message and extract code blocks."""
    conn.execute(
        """INSERT OR IGNORE INTO messages(id, conv_id, role, content, timestamp, content_type)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (id, conv_id, role, content, timestamp, content_type),
    )

    # Extract and insert code blocks
    if content:
        for lang, code in extract_code_blocks(content):
            conn.execute(
                "INSERT INTO code_blocks(msg_id, language, code) VALUES (?, ?, ?)",
                (id, lang, code),
            )


def update_conversation_counts(conn: sqlite3.Connection) -> None:
    """Update message_count on all conversations."""
    conn.execute(
        """UPDATE conversations SET message_count = (
               SELECT COUNT(*) FROM messages WHERE messages.conv_id = conversations.id
           )"""
    )
