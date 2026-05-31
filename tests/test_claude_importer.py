"""Tests for the Claude conversation importer."""

from __future__ import annotations

import json
import sqlite3
import zipfile

import pytest

from ricoeur.db import SCHEMA_SQL
from ricoeur.importers.claude import (
    UNSUPPORTED_BLOCK,
    import_claude,
    _extract_content,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def conn():
    """In-memory SQLite database with the ricoeur schema."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_SQL)
    return c


def _conv(uuid, name, messages, **extra):
    """Build a Claude-export-shaped conversation dict."""
    return {
        "uuid": uuid,
        "name": name,
        "summary": "",
        "created_at": "2025-10-23T01:18:38.477062Z",
        "updated_at": "2025-10-23T01:18:45.100465Z",
        "account": {"uuid": "acct-1"},
        "chat_messages": messages,
        **extra,
    }


def _msg(uuid, sender, text=None, content=None, **extra):
    m = {
        "uuid": uuid,
        "sender": sender,
        "created_at": "2025-10-23T01:18:39.446848Z",
        "updated_at": "2025-10-23T01:18:39.446848Z",
        "attachments": [],
        "files": [],
        "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        **extra,
    }
    if text is not None:
        m["text"] = text
    if content is not None:
        m["content"] = content
    return m


@pytest.fixture
def sample_export():
    return [
        _conv(
            "conv-a",
            "Magic number calculation",
            [
                _msg("m1", "human", text="compute magic of 5"),
                _msg(
                    "m2",
                    "assistant",
                    text="The magic constant of a 5x5 square is 65.\n```python\nprint(65)\n```",
                ),
            ],
        ),
        _conv(
            "conv-b",
            "Strawberry letters",
            [
                _msg("m3", "human", text="count r in strawberry"),
                _msg("m4", "assistant", text="There are 3 r's."),
            ],
        ),
    ]


# ── Basic import ─────────────────────────────────────────────────────────


def test_import_from_file(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))

    stats = import_claude(conn, p)

    assert stats.parsed == 2
    assert stats.new == 2
    assert stats.skipped == 0
    assert stats.messages == 4

    convs = conn.execute(
        "SELECT id, title, platform, model FROM conversations ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in convs] == ["conv-a", "conv-b"]
    assert all(r["platform"] == "claude" for r in convs)
    # Claude exports carry no model field
    assert all(r["model"] is None for r in convs)


def test_role_normalization(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    import_claude(conn, p)

    roles = {
        r["role"]
        for r in conn.execute("SELECT DISTINCT role FROM messages").fetchall()
    }
    # "human" must be normalized to "user"; no raw "human" left
    assert roles == {"user", "assistant"}


def test_message_count_updated(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    import_claude(conn, p)

    row = conn.execute(
        "SELECT message_count FROM conversations WHERE id = 'conv-a'"
    ).fetchone()
    assert row["message_count"] == 2


def test_code_blocks_extracted(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    import_claude(conn, p)

    blocks = conn.execute(
        "SELECT language, code FROM code_blocks"
    ).fetchall()
    assert len(blocks) == 1
    assert blocks[0]["language"] == "python"
    assert "print(65)" in blocks[0]["code"]


def test_timestamps_preserved(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    import_claude(conn, p)

    # Claude timestamps are already ISO strings — stored as-is
    row = conn.execute(
        "SELECT created_at FROM conversations WHERE id = 'conv-a'"
    ).fetchone()
    assert row["created_at"] == "2025-10-23T01:18:38.477062Z"


# ── Options ──────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    stats = import_claude(conn, p, dry_run=True)

    assert stats.new == 2
    assert conn.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] == 0


def test_since_filter(conn, sample_export, tmp_path):
    # First conversation is dated 2025-10-23; filter past it
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    stats = import_claude(conn, p, since="2026-01-01")

    assert stats.new == 0
    assert stats.skipped == 2


def test_reimport_skips_without_update(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    import_claude(conn, p)
    stats = import_claude(conn, p)  # second pass

    assert stats.new == 0
    assert stats.skipped == 2
    assert conn.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"] == 2


def test_update_flag_refreshes_title(conn, sample_export, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(sample_export))
    import_claude(conn, p)

    sample_export[0]["name"] = "Renamed conversation"
    p.write_text(json.dumps(sample_export))
    stats = import_claude(conn, p, update=True)

    assert stats.new == 2  # update counts as inserted in stats
    title = conn.execute(
        "SELECT title FROM conversations WHERE id = 'conv-a'"
    ).fetchone()["title"]
    assert title == "Renamed conversation"


def test_load_from_zip(conn, sample_export, tmp_path):
    zip_path = tmp_path / "claude-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(sample_export))

    stats = import_claude(conn, zip_path)
    assert stats.new == 2


def test_non_list_raises(conn, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="array of conversations"):
        import_claude(conn, p)


# ── Content extraction ─────────────────────────────────────────────────────


def test_extract_content_prefers_text_field():
    msg = {"text": "hello", "content": [{"type": "text", "text": "ignored"}]}
    assert _extract_content(msg) == "hello"


def test_extract_content_falls_back_to_blocks():
    # Empty top-level text → reconstruct from content blocks
    msg = {
        "text": "   ",
        "content": [
            {"type": "thinking", "thinking": "reasoning here"},
            {"type": "tool_use", "name": "web_search"},
            {"type": "tool_result", "content": "noise"},
            {"type": "text", "text": "the answer"},
        ],
    }
    out = _extract_content(msg)
    assert "the answer" in out
    assert "reasoning here" in out
    # tool noise excluded
    assert "web_search" not in out
    assert "noise" not in out


def test_extract_content_string_content():
    assert _extract_content({"content": "plain string"}) == "plain string"


# ── Unsupported-block (placeholder) handling — issue #17 ────────────────────


def test_placeholder_text_recovered_from_blocks():
    """A `text` field contaminated with the unsupported-block placeholder is
    rebuilt from the structured content (here an artifact carrying real code)."""
    msg = {
        "text": (
            "Here's the app:\n```\n" + UNSUPPORTED_BLOCK + "\n```\nDone."
        ),
        "content": [
            {"type": "text", "text": "Here's the app:"},
            {
                "type": "tool_use",
                "name": "artifacts",
                "input": {
                    "title": "Weather Fit",
                    "language": "matlab",
                    "content": "x = polyfit(t, y, 3);",
                },
            },
            {"type": "text", "text": "Done."},
        ],
    }
    out = _extract_content(msg)
    assert UNSUPPORTED_BLOCK not in out
    assert "polyfit(t, y, 3)" in out
    assert "Weather Fit" in out
    assert "```matlab" in out
    assert "Here's the app:" in out and "Done." in out


def test_placeholder_text_stripped_when_no_blocks():
    """When only the flattened `text` survives, the placeholder noise (and the
    empty fence around it) is stripped, keeping the real prose."""
    msg = {
        "text": (
            "I'll count the r's.\n```\n"
            + UNSUPPORTED_BLOCK
            + "\n```\n\nThere are 3 r's."
        )
    }
    out = _extract_content(msg)
    assert UNSUPPORTED_BLOCK not in out
    assert "```" not in out  # the empty placeholder fence is gone
    assert "I'll count the r's." in out
    assert "There are 3 r's." in out


def test_thinking_block_is_labelled_as_blockquote():
    """Thinking renders as a labelled Markdown blockquote, distinct from the
    response text."""
    msg = {
        "text": "",
        "content": [
            {"type": "thinking", "thinking": "Let me reason.\nStep two."},
            {"type": "text", "text": "The final answer is 42."},
        ],
    }
    out = _extract_content(msg)
    assert "💭 *Thinking…*" in out
    assert "> Let me reason." in out  # thinking lines are quoted
    assert "> Step two." in out
    # The response itself is NOT quoted.
    assert "The final answer is 42." in out
    assert "> The final answer" not in out


def test_thinking_labelled_even_when_flattened_text_is_clean():
    """A clean flattened `text` is overridden so thinking can be called out."""
    msg = {
        "text": "The final answer is 42.",
        "content": [
            {"type": "thinking", "thinking": "secret reasoning"},
            {"type": "text", "text": "The final answer is 42."},
        ],
    }
    out = _extract_content(msg)
    assert "💭 *Thinking…*" in out
    assert "secret reasoning" in out


def test_tool_use_is_labelled():
    msg = {
        "text": "",
        "content": [
            {
                "type": "tool_use",
                "name": "repl",
                "input": {"code": "print(65)", "language": "python"},
            }
        ],
    }
    out = _extract_content(msg)
    assert "🛠️ **Tool · repl**" in out
    assert "```python" in out
    assert "print(65)" in out


def test_code_execution_tool_use_recovered():
    msg = {
        "text": UNSUPPORTED_BLOCK,
        "content": [
            {"type": "tool_use", "name": "repl", "input": {"code": "print(65)"}},
        ],
    }
    out = _extract_content(msg)
    assert "print(65)" in out
    assert UNSUPPORTED_BLOCK not in out


def test_interface_only_tool_use_renders_nothing():
    """A tool_use with no code/content (e.g. web search) contributes nothing."""
    msg = {
        "text": "  ",
        "content": [
            {"type": "tool_use", "name": "web_search", "input": {"query": "x"}},
            {"type": "text", "text": "the answer"},
        ],
    }
    out = _extract_content(msg)
    assert out.strip() == "the answer"


def test_update_flag_refreshes_message_content(conn, tmp_path):
    """Re-importing with --update replaces stale message content — the recovery
    path for conversations imported before the placeholder fix."""
    stale_text = "Here's the app:\n```\n" + UNSUPPORTED_BLOCK + "\n```"
    stale = [
        _conv(
            "conv-x",
            "Artifact chat",
            [_msg("mx", "assistant", text=stale_text)],
        )
    ]
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(stale))
    import_claude(conn, p)
    before = conn.execute(
        "SELECT content FROM messages WHERE id = 'mx'"
    ).fetchone()["content"]
    # No structured blocks to recover from → placeholder noise is stripped,
    # leaving just the prose.
    assert UNSUPPORTED_BLOCK not in before
    assert "Here's the app:" in before

    # The real export carries structured blocks; re-import with update.
    fixed = [
        _conv(
            "conv-x",
            "Artifact chat",
            [
                _msg(
                    "mx",
                    "assistant",
                    text=stale_text,
                    content=[
                        {"type": "text", "text": "Here's the app:"},
                        {
                            "type": "tool_use",
                            "name": "artifacts",
                            "input": {"language": "python", "content": "print(65)"},
                        }
                    ],
                ),
            ],
        )
    ]
    p.write_text(json.dumps(fixed))
    import_claude(conn, p, update=True)

    after = conn.execute(
        "SELECT content FROM messages WHERE id = 'mx'"
    ).fetchone()["content"]
    assert UNSUPPORTED_BLOCK not in after
    assert "print(65)" in after
    # No duplicate rows, and the recovered code block is indexed.
    assert conn.execute("SELECT COUNT(*) n FROM messages WHERE id='mx'").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM code_blocks").fetchone()["n"] == 1


def test_extract_content_empty():
    assert _extract_content({"content": []}) == ""
    assert _extract_content({}) == ""


def test_pure_tool_message_skipped(conn, tmp_path):
    """A message with no visible text (only tool/thinking-less blocks) is skipped."""
    export = [
        _conv(
            "conv-c",
            "Tool only",
            [
                _msg("m5", "human", text="do something"),
                _msg(
                    "m6",
                    "assistant",
                    text="",
                    content=[
                        {"type": "tool_use", "name": "bash"},
                        {"type": "tool_result", "content": "result"},
                    ],
                ),
            ],
        )
    ]
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(export))
    stats = import_claude(conn, p)

    # Only the human message has content
    assert stats.messages == 1
    roles = [
        r["role"] for r in conn.execute("SELECT role FROM messages").fetchall()
    ]
    assert roles == ["user"]
