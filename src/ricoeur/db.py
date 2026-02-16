"""SQLite database layer for ricoeur."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_home

SCHEMA_VERSION = 1

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
    message_count INTEGER DEFAULT 0
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

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Full-text search index on messages
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    conv_id UNINDEXED,
    msg_id UNINDEXED,
    role UNINDEXED,
    content='messages',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, conv_id, msg_id, role)
    VALUES (new.rowid, new.content, new.conv_id, new.id, new.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, conv_id, msg_id, role)
    VALUES ('delete', old.rowid, old.content, old.conv_id, old.id, old.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, conv_id, msg_id, role)
    VALUES ('delete', old.rowid, old.content, old.conv_id, old.id, old.role);
    INSERT INTO messages_fts(rowid, content, conv_id, msg_id, role)
    VALUES (new.rowid, new.content, new.conv_id, new.id, new.role);
END;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_conversations_platform ON conversations(platform);
CREATE INDEX IF NOT EXISTS idx_conversations_language ON conversations(language);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_topic ON conversations(topic_id);
CREATE INDEX IF NOT EXISTS idx_code_blocks_msg ON code_blocks(msg_id);
CREATE INDEX IF NOT EXISTS idx_code_blocks_lang ON code_blocks(language);
"""


def db_path(home: Optional[Path] = None) -> Path:
    return (home or get_home()) / "ricoeur.db"


def get_connection(home: Optional[Path] = None) -> sqlite3.Connection:
    """Get a connection to the ricoeur database."""
    path = db_path(home)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
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
