"""Tests for the Claude Code session importer."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ricoeur.db import SCHEMA_SQL
from ricoeur.importers.claude_code import (
    _clean_prompt,
    _render_tool_use,
    import_claude_code,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def conn():
    """In-memory SQLite database with the ricoeur schema."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_SQL)
    return c


SESSION = "11111111-2222-3333-4444-555555555555"


def _rec(
    rtype,
    uuid,
    *,
    content=None,
    ts="2026-08-04T07:55:36.764Z",
    message=None,
    **extra,
):
    """Build a Claude-Code-shaped JSONL record.

    ``message`` overrides fields on the generated message envelope (e.g. to
    set a ``<synthetic>`` model).
    """
    rec = {
        "type": rtype,
        "uuid": uuid,
        "sessionId": SESSION,
        "timestamp": ts,
        "cwd": "/Users/me/Devel/ricoeur",
        "gitBranch": "main",
        "version": "2.0.0",
        "isSidechain": False,
        **extra,
    }
    if content is not None:
        envelope = {
            "role": "user" if rtype == "user" else "assistant",
            "content": content,
        }
        if rtype == "assistant":
            envelope["model"] = "claude-opus-5"
            envelope["usage"] = {"input_tokens": 100, "output_tokens": 42}
        envelope.update(message or {})
        rec["message"] = envelope
    return rec


def _write(path, records):
    """Write records as a JSONL session log named after the session id."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


@pytest.fixture
def session_file(tmp_path):
    """A projects root holding one realistic session."""
    project = tmp_path / "-Users-me-Devel-ricoeur"
    project.mkdir()
    records = [
        # Harness state — must be ignored entirely.
        {"type": "mode", "mode": "default", "sessionId": SESSION},
        {"type": "permission-mode", "permissionMode": "acceptEdits", "sessionId": SESSION},
        _rec("user", "u1", content="add a claude-code importer"),
        _rec(
            "assistant",
            "a1",
            content=[{"type": "text", "text": "On it."}],
        ),
        _rec(
            "assistant",
            "a2",
            content=[
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Write",
                    "input": {
                        "file_path": "/Users/me/Devel/ricoeur/hello.py",
                        "content": "print('hi')",
                    },
                }
            ],
        ),
        # Tool output fed back to the model — not a human turn.
        _rec(
            "user",
            "u2",
            content=[{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        ),
        {"type": "ai-title", "aiTitle": "Add a Claude Code importer", "sessionId": SESSION},
    ]
    _write(project / f"{SESSION}.jsonl", records)
    return tmp_path


# ── Basic import ─────────────────────────────────────────────────────────


def test_import_session(conn, session_file):
    stats = import_claude_code(conn, session_file)

    assert stats.parsed == 1
    assert stats.new == 1
    assert stats.messages == 3  # one prompt, two assistant turns; no tool_result

    conv = conn.execute("SELECT * FROM conversations").fetchone()
    assert conv["id"] == SESSION
    assert conv["platform"] == "claude-code"
    assert conv["model"] == "claude-opus-5"
    assert conv["message_count"] == 3


def test_title_from_ai_title(conn, session_file):
    import_claude_code(conn, session_file)
    conv = conn.execute("SELECT title FROM conversations").fetchone()
    assert conv["title"] == "Add a Claude Code importer"


def test_title_falls_back_to_first_prompt(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec("user", "u1", content="why is the FTS index stale?"),
            _rec("assistant", "a1", content=[{"type": "text", "text": "Because…"}]),
        ],
    )
    import_claude_code(conn, tmp_path)
    conv = conn.execute("SELECT title FROM conversations").fetchone()
    assert conv["title"] == "why is the FTS index stale?"


def test_long_first_prompt_title_is_truncated(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [_rec("user", "u1", content="explain " + "x" * 200)],
    )
    import_claude_code(conn, tmp_path)
    title = conn.execute("SELECT title FROM conversations").fetchone()["title"]
    assert len(title) <= 72
    assert title.endswith("…")


def test_project_and_source_path_recorded(conn, session_file):
    import_claude_code(conn, session_file)
    conv = conn.execute("SELECT project, source_path FROM conversations").fetchone()
    # Taken from the record's cwd, not the slugified directory name
    assert conv["project"] == "ricoeur"
    assert conv["source_path"].endswith(f"{SESSION}.jsonl")


def test_timestamps_span_the_session(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec("user", "u1", content="first", ts="2026-08-04T07:00:00.000Z"),
            _rec(
                "assistant",
                "a1",
                content=[{"type": "text", "text": "last"}],
                ts="2026-08-04T09:30:00.000Z",
            ),
        ],
    )
    import_claude_code(conn, tmp_path)
    conv = conn.execute("SELECT created_at, updated_at FROM conversations").fetchone()
    assert conv["created_at"] == "2026-08-04T07:00:00.000Z"
    assert conv["updated_at"] == "2026-08-04T09:30:00.000Z"


def test_token_count_from_usage(conn, session_file):
    import_claude_code(conn, session_file)
    rows = conn.execute(
        "SELECT role, token_count FROM messages ORDER BY id"
    ).fetchall()
    by_role = {r["role"]: r["token_count"] for r in rows}
    assert by_role["assistant"] == 42
    assert by_role["user"] is None


# ── What counts as a message ─────────────────────────────────────────────


def test_tool_result_is_not_a_user_message(conn, session_file):
    """The single most consequential filter: 6,900-odd tool_result records in a
    real corpus would otherwise masquerade as things the human said."""
    import_claude_code(conn, session_file)
    users = conn.execute("SELECT content FROM messages WHERE role = 'user'").fetchall()
    assert len(users) == 1
    assert users[0]["content"] == "add a claude-code importer"


def test_tool_results_included_on_request(conn, session_file):
    import_claude_code(conn, session_file, include_tool_results=True)
    users = conn.execute(
        "SELECT content FROM messages WHERE role = 'user' ORDER BY timestamp"
    ).fetchall()
    assert len(users) == 2
    assert "ok" in users[1]["content"]


def test_meta_and_error_records_skipped(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec("user", "u1", content="real prompt"),
            _rec("user", "u2", content="<system-reminder>noise</system-reminder>", isMeta=True),
            _rec(
                "assistant",
                "a1",
                content=[{"type": "text", "text": "API Error: overloaded"}],
                isApiErrorMessage=True,
            ),
            _rec(
                "assistant",
                "a2",
                content=[{"type": "text", "text": "[Request interrupted]"}],
                message={"model": "<synthetic>"},
            ),
        ],
    )
    stats = import_claude_code(conn, tmp_path)
    assert stats.messages == 1
    contents = [r["content"] for r in conn.execute("SELECT content FROM messages")]
    assert contents == ["real prompt"]


def test_asking_for_more_content_bypasses_the_cache(conn, session_file):
    """--include-tool-results after a plain import must not be a no-op."""
    import_claude_code(conn, session_file)
    before = conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"]

    stats = import_claude_code(conn, session_file, include_tool_results=True)

    assert stats.skipped == 0
    assert stats.messages == 1  # the tool_result turn, now archived
    assert conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] == before + 1


def test_sidechains_skipped_by_default(conn, tmp_path):
    records = [
        _rec("user", "u1", content="delegate this"),
        _rec(
            "assistant",
            "a1",
            content=[{"type": "text", "text": "subagent reasoning"}],
            isSidechain=True,
        ),
    ]
    _write(tmp_path / f"{SESSION}.jsonl", records)

    import_claude_code(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] == 1

    import_claude_code(conn, tmp_path, include_sidechains=True)
    assert conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] == 2


def test_empty_thinking_block_yields_no_message(conn, tmp_path):
    """Claude Code persists thinking blocks with a signature but no text."""
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec("user", "u1", content="hello"),
            _rec(
                "assistant",
                "a1",
                content=[{"type": "thinking", "thinking": "", "signature": "abc"}],
            ),
        ],
    )
    stats = import_claude_code(conn, tmp_path)
    assert stats.messages == 1


def test_thinking_text_is_labelled_when_present(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec(
                "assistant",
                "a1",
                content=[
                    {"type": "thinking", "thinking": "weighing options", "signature": "s"},
                    {"type": "text", "text": "Here you go."},
                ],
            )
        ],
    )
    import_claude_code(conn, tmp_path)
    content = conn.execute("SELECT content FROM messages").fetchone()["content"]
    assert "> 💭 *Thinking…*" in content
    assert "> weighing options" in content
    assert "Here you go." in content


def test_compact_summary_marked(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [_rec("user", "u1", content="summary of earlier work", isCompactSummary=True)],
    )
    import_claude_code(conn, tmp_path)
    row = conn.execute("SELECT content_type FROM messages").fetchone()
    assert row["content_type"] == "compact_summary"


def test_images_recorded_as_attachments(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec(
                "user",
                "u1",
                content=[
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "data": "…"}},
                ],
            )
        ],
    )
    stats = import_claude_code(conn, tmp_path)
    assert stats.attachments == 1
    row = conn.execute("SELECT conv_id, type FROM attachments").fetchone()
    assert row["conv_id"] == SESSION
    assert row["type"] == "image"


# ── Tool rendering ───────────────────────────────────────────────────────


def test_write_tool_becomes_a_fenced_code_block(conn, session_file):
    import_claude_code(conn, session_file)
    row = conn.execute(
        "SELECT language, code FROM code_blocks"
    ).fetchone()
    assert row["language"] == "python"
    assert row["code"] == "print('hi')"


def test_bash_tool_renders_command():
    out = _render_tool_use(
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "pytest -q", "description": "Run tests"},
        }
    )
    assert "Tool · Bash" in out
    assert "Run tests" in out
    assert "```bash\npytest -q\n```" in out


def test_edit_tool_renders_a_diff():
    out = _render_tool_use(
        {
            "type": "tool_use",
            "name": "Edit",
            "input": {
                "file_path": "app.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
            },
        }
    )
    assert "```diff" in out
    assert "-x = 1" in out
    assert "+x = 2" in out


def test_content_free_tools_render_nothing():
    for name, inp in [
        ("Read", {"file_path": "a.py"}),
        ("Glob", {"pattern": "**/*.py"}),
        ("TodoWrite", {"todos": []}),
    ]:
        assert _render_tool_use({"name": name, "input": inp}) == ""


def test_task_prompt_is_not_fenced():
    """A subagent prompt is prose — fencing it would pollute code_blocks."""
    out = _render_tool_use(
        {
            "name": "Task",
            "input": {"prompt": "research the format", "description": "Research"},
        }
    )
    assert "research the format" in out
    assert "```" not in out


def test_mcp_tool_label_is_readable():
    out = _render_tool_use(
        {"name": "mcp__skore__skore_agent", "input": {"command": "run"}}
    )
    assert "skore · skore_agent" in out


# ── Prompt cleaning ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "<command-message>humanizer</command-message>\n<command-name>/humanizer</command-name>",
            "`/humanizer`",
        ),
        (
            "<command-name>/login</command-name>\n<command-message>login</command-message>\n<command-args></command-args>",
            "`/login`",
        ),
        (
            "<command-name>/plugin</command-name><command-args>install x</command-args>",
            "`/plugin install x`",
        ),
        ("real prompt\n<local-command-stdout>noise</local-command-stdout>", "real prompt"),
        ("<task-notification><summary>done</summary></task-notification>", ""),
        ("plain prompt", "plain prompt"),
    ],
)
def test_clean_prompt(raw, expected):
    assert _clean_prompt(raw) == expected


def test_bash_gesture_becomes_a_code_block():
    out = _clean_prompt("<bash-input>git status</bash-input><bash-stdout>clean</bash-stdout>")
    assert "```bash\ngit status\n```" in out
    assert "clean" not in out


def test_harness_only_message_is_dropped(conn, tmp_path):
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            _rec("user", "u1", content="<task-notification><summary>x</summary></task-notification>"),
            _rec("user", "u2", content="a real question"),
        ],
    )
    stats = import_claude_code(conn, tmp_path)
    assert stats.messages == 1


# ── Append-only re-import ────────────────────────────────────────────────


def test_reimport_is_idempotent(conn, session_file):
    first = import_claude_code(conn, session_file)
    counts = lambda table: conn.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"]
    msgs, blocks = counts("messages"), counts("code_blocks")

    second = import_claude_code(conn, session_file, update=True)

    assert second.new == 0
    assert second.updated == 1
    assert second.messages == 0  # nothing archived that wasn't already
    assert counts("messages") == msgs == first.messages
    assert counts("code_blocks") == blocks


def test_growing_session_appends_only_new_messages(conn, tmp_path):
    path = tmp_path / f"{SESSION}.jsonl"
    _write(path, [_rec("user", "u1", content="first", ts="2026-08-04T07:00:00.000Z")])
    import_claude_code(conn, tmp_path)

    # The session keeps running and the log grows.
    with open(path, "a") as fh:
        fh.write(
            json.dumps(
                _rec(
                    "assistant",
                    "a1",
                    content=[{"type": "text", "text": "second"}],
                    ts="2026-08-04T08:00:00.000Z",
                )
            )
            + "\n"
        )

    stats = import_claude_code(conn, tmp_path, update=True)

    assert stats.messages == 1
    assert stats.updated == 1
    contents = [
        r["content"] for r in conn.execute("SELECT content FROM messages ORDER BY timestamp")
    ]
    assert contents == ["first", "second"]
    conv = conn.execute("SELECT updated_at, message_count FROM conversations").fetchone()
    assert conv["updated_at"] == "2026-08-04T08:00:00.000Z"
    assert conv["message_count"] == 2


def test_unchanged_sessions_are_skipped_without_update(conn, session_file):
    import_claude_code(conn, session_file)
    again = import_claude_code(conn, session_file)
    assert again.skipped == 1
    assert again.new == 0
    assert again.updated == 0


def test_changed_session_is_reread_without_update(conn, tmp_path):
    """The mtime/size cache must not hide a session that actually grew."""
    path = tmp_path / f"{SESSION}.jsonl"
    _write(path, [_rec("user", "u1", content="first")])
    import_claude_code(conn, tmp_path)

    _write(
        path,
        [
            _rec("user", "u1", content="first"),
            _rec("user", "u2", content="a second, longer prompt"),
        ],
    )
    stats = import_claude_code(conn, tmp_path)
    assert stats.messages == 1
    assert conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] == 2


# ── Robustness ───────────────────────────────────────────────────────────


def test_truncated_final_line_tolerated(conn, tmp_path):
    """A session being written right now ends mid-record."""
    path = tmp_path / f"{SESSION}.jsonl"
    good = json.dumps(_rec("user", "u1", content="a question"))
    path.write_text(good + "\n" + '{"type": "assistant", "message": {"cont')

    stats = import_claude_code(conn, tmp_path)

    assert stats.malformed == 1
    assert stats.messages == 1


def test_content_free_session_skipped(conn, tmp_path):
    """4 of 87 real logs hold only harness state. Not an error."""
    _write(
        tmp_path / f"{SESSION}.jsonl",
        [
            {"type": "mode", "mode": "default", "sessionId": SESSION},
            {"type": "file-history-snapshot", "messageId": "m1"},
        ],
    )
    stats = import_claude_code(conn, tmp_path)
    assert stats.parsed == 1
    assert stats.skipped == 1
    assert conn.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"] == 0


def test_empty_file_skipped(conn, tmp_path):
    (tmp_path / f"{SESSION}.jsonl").write_text("")
    stats = import_claude_code(conn, tmp_path)
    assert stats.skipped == 1
    assert stats.malformed == 0


def test_missing_path_raises(conn, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_claude_code(conn, tmp_path / "nope")


# ── Options ──────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(conn, session_file):
    stats = import_claude_code(conn, session_file, dry_run=True)
    assert stats.new == 1
    assert conn.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM import_sources").fetchone()["n"] == 0


def test_since_filter(conn, tmp_path):
    _write(
        tmp_path / "old.jsonl",
        [_rec("user", "u1", content="ancient", ts="2026-01-01T00:00:00.000Z")],
    )
    _write(
        tmp_path / "new.jsonl",
        [
            _rec(
                "user",
                "u2",
                content="recent",
                ts="2026-08-04T00:00:00.000Z",
                sessionId="99999999-2222-3333-4444-555555555555",
            )
        ],
    )
    stats = import_claude_code(conn, tmp_path, since="2026-06-01")
    assert stats.new == 1
    assert stats.skipped == 1


def test_project_filter(conn, tmp_path):
    for slug, session in [
        ("-Users-me-Devel-ricoeur", SESSION),
        ("-Users-me-Devel-other", "99999999-2222-3333-4444-555555555555"),
    ]:
        d = tmp_path / slug
        d.mkdir()
        _write(d / f"{session}.jsonl", [_rec("user", "u1", content="hi", sessionId=session)])

    stats = import_claude_code(conn, tmp_path, project="ricoeur")
    assert stats.parsed == 1
    assert stats.new == 1


def test_single_session_file_accepted(conn, session_file):
    path = next(session_file.glob("*/*.jsonl"))
    stats = import_claude_code(conn, path)
    assert stats.parsed == 1
    assert stats.new == 1
