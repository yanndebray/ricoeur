"""SQLite database layer for ricoeur."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_home

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    platform TEXT NOT NULL,
    model TEXT,
    created_at TEXT,
    updated_at TEXT,
    language TEXT,
    topic_id INTEGER,
    message_count INTEGER DEFAULT 0,
    project TEXT,
    source_path TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT,
    timestamp TEXT,
    content_type TEXT DEFAULT 'text',
    token_count INTEGER
);

CREATE TABLE IF NOT EXISTS code_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_id TEXT NOT NULL REFERENCES messages(id),
    language TEXT,
    code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id TEXT NOT NULL REFERENCES conversations(id),
    msg_id TEXT REFERENCES messages(id),
    type TEXT,
    filename TEXT,
    path TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    conv_id TEXT PRIMARY KEY REFERENCES conversations(id),
    summary TEXT NOT NULL,
    model_used TEXT
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    label TEXT,
    keywords TEXT,
    count INTEGER DEFAULT 0
);

-- Per-file import bookkeeping for append-only local sources (Claude Code
-- session logs). Purely a speed cache: message IDs make re-import idempotent,
-- so a stale or missing row here costs time, never correctness.
CREATE TABLE IF NOT EXISTS import_sources (
    path TEXT PRIMARY KEY,
    conv_id TEXT,
    mtime REAL,
    size INTEGER,
    imported_at TEXT
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Full-text search index on messages
-- Column names must match the content table ('messages') for content-sync to work.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    conv_id UNINDEXED,
    id UNINDEXED,
    role UNINDEXED,
    content='messages',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, conv_id, id, role)
    VALUES (new.rowid, new.content, new.conv_id, new.id, new.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, conv_id, id, role)
    VALUES ('delete', old.rowid, old.content, old.conv_id, old.id, old.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, conv_id, id, role)
    VALUES ('delete', old.rowid, old.content, old.conv_id, old.id, old.role);
    INSERT INTO messages_fts(rowid, content, conv_id, id, role)
    VALUES (new.rowid, new.content, new.conv_id, new.id, new.role);
END;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_conversations_platform ON conversations(platform);
CREATE INDEX IF NOT EXISTS idx_conversations_language ON conversations(language);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_topic ON conversations(topic_id);
CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project);
CREATE INDEX IF NOT EXISTS idx_code_blocks_msg ON code_blocks(msg_id);
CREATE INDEX IF NOT EXISTS idx_code_blocks_lang ON code_blocks(language);
"""


def db_path(home: Optional[Path] = None) -> Path:
    return (home or get_home()) / "ricoeur.db"


def get_connection(home: Optional[Path] = None) -> sqlite3.Connection:
    """Get a connection to the ricoeur database, migrating it if needed."""
    path = db_path(home)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def init_db(home: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize the database with the schema."""
    conn = get_connection(home)
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    return conn


# ── Migrations ───────────────────────────────────────────────────────────────
#
# ``SCHEMA_SQL`` is all ``CREATE ... IF NOT EXISTS``, so it brings a *fresh*
# database up to the current version but silently leaves an older one behind.
# ``migrate`` closes that gap: every step is idempotent, so it is safe to run
# on every connection.


def schema_version(conn: sqlite3.Connection) -> int:
    """Read the recorded schema version (1 for pre-versioning databases)."""
    if not _table_exists(conn, "schema_meta"):
        return 1
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    if row is None:
        return 1
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 1


def migrate(conn: sqlite3.Connection) -> int:
    """Bring an existing database up to ``SCHEMA_VERSION``.

    Returns the resulting version. A database with no ``conversations`` table
    is untouched — it is either brand new (``init_db`` will build it) or not a
    ricoeur database at all.
    """
    if not _table_exists(conn, "conversations"):
        return SCHEMA_VERSION

    version = schema_version(conn)
    if version >= SCHEMA_VERSION:
        return version

    if version < 2:
        # v2: provenance for local session-log sources (Claude Code).
        _add_column(conn, "conversations", "project", "TEXT")
        _add_column(conn, "conversations", "source_path", "TEXT")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS import_sources (
                path TEXT PRIMARY KEY,
                conv_id TEXT,
                mtime REAL,
                size INTEGER,
                imported_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_project
                ON conversations(project);
            """
        )

    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    return SCHEMA_VERSION


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _add_column(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    """``ALTER TABLE ... ADD COLUMN``, skipped if the column already exists."""
    # index, not name: works whether or not a row factory is installed
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
