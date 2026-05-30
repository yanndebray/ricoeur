"""Tests for the Textual TUI.

These exercise the app headlessly via Textual's test pilot. Skipped entirely
when the optional ``textual`` dependency isn't installed.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

pytest.importorskip("textual")

from ricoeur.db import SCHEMA_SQL  # noqa: E402
from ricoeur.tui import RicoeurApp, ConversationScreen  # noqa: E402
from textual.widgets import DataTable, Static  # noqa: E402


@pytest.fixture
def populated_home(tmp_path):
    """A ricoeur home dir with a seeded database."""
    db = tmp_path / "ricoeur.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT INTO conversations (id, title, platform, model, created_at, message_count) "
        "VALUES ('c1', 'MATLAB from Python', 'claude', NULL, '2025-10-26T10:00:00Z', 2)"
    )
    conn.execute(
        "INSERT INTO conversations (id, title, platform, model, created_at, message_count) "
        "VALUES ('c2', 'Docker basics', 'chatgpt', 'gpt-4o', '2025-09-01T10:00:00Z', 2)"
    )
    conn.executemany(
        "INSERT INTO messages (id, conv_id, role, content, timestamp) VALUES (?,?,?,?,?)",
        [
            ("m1", "c1", "user", "how to call matlab from python", "2025-10-26T10:00:01Z"),
            ("m2", "c1", "assistant", "Use the matlabengine package.", "2025-10-26T10:00:02Z"),
            ("m3", "c2", "user", "what is docker", "2025-09-01T10:00:01Z"),
            ("m4", "c2", "assistant", "Docker packages apps in containers.", "2025-09-01T10:00:02Z"),
        ],
    )
    conn.commit()
    conn.close()
    return tmp_path


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_mount_shows_recent(populated_home):
    async def scenario():
        app = RicoeurApp(home=populated_home)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#results", DataTable)
            # Both conversations show, most recent first
            assert table.row_count == 2
            assert app._row_ids[0] == "c1"
            status = str(app.query_one("#status", Static).render())
            assert "2 conversations" in status

    _run(scenario())


def test_keyword_search_filters(populated_home):
    async def scenario():
        app = RicoeurApp(home=populated_home)
        async with app.run_test() as pilot:
            search = app.query_one("#search")
            search.focus()
            search.value = "matlab"
            await pilot.press("enter")
            await pilot.pause()
            table = app.query_one("#results", DataTable)
            assert table.row_count == 1
            assert app._row_ids == ["c1"]

    _run(scenario())


def test_no_match_clears_table(populated_home):
    async def scenario():
        app = RicoeurApp(home=populated_home)
        async with app.run_test() as pilot:
            search = app.query_one("#search")
            search.focus()
            search.value = "nonexistentterm"
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#results", DataTable).row_count == 0
            assert app._row_ids == []

    _run(scenario())


def test_empty_query_returns_to_recent(populated_home):
    async def scenario():
        app = RicoeurApp(home=populated_home)
        async with app.run_test() as pilot:
            search = app.query_one("#search")
            search.focus()
            search.value = "matlab"
            await pilot.press("enter")
            await pilot.pause()
            # Searching moves focus to the results; refocus search to clear it
            search.focus()
            search.value = ""
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#results", DataTable).row_count == 2

    _run(scenario())


def test_open_conversation_renders(populated_home):
    async def scenario():
        app = RicoeurApp(home=populated_home)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._open_row(0)
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            # Back out
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)

    _run(scenario())


def test_transcript_includes_messages(populated_home):
    """The detail screen's transcript carries the conversation's messages."""
    async def scenario():
        app = RicoeurApp(home=populated_home)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = ConversationScreen(app._conn, "c1")
            text = screen._transcript()
            assert "MATLAB from Python" in text
            assert "matlabengine" in text
            assert "how to call matlab from python" in text

    _run(scenario())
