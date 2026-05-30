"""Terminal UI for ricoeur, built with Textual.

Browse, search, and read your conversation archive interactively. The TUI is a
thin shell over the same data layer the CLI uses (``db`` + ``search``), so it
needs no extra state of its own.

Launch with ``ricoeur tui`` (requires ``pip install 'ricoeur[tui]'``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_home, load_config
from .db import get_connection
from .search import search_dispatch

try:
    from rich.markdown import Markdown as RichMarkdown
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import VerticalScroll
    from textual.screen import Screen
    from textual.widgets import (
        DataTable,
        Footer,
        Header,
        Input,
        Static,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via CLI
    raise ModuleNotFoundError(
        "The TUI requires the 'textual' package. Install it with: "
        "pip install 'ricoeur[tui]'  (or: uv sync --extra tui)"
    ) from exc


# ── Data helpers ───────────────────────────────────────────────────────────


def _recent_conversations(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    """Most recently created conversations — the default browse view."""
    return conn.execute(
        """SELECT id, title, platform, model, created_at, message_count
           FROM conversations
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()


def _conversation(conn: sqlite3.Connection, conv_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()


def _messages(conn: sqlite3.Connection, conv_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE conv_id = ? ORDER BY timestamp",
        (conv_id,),
    ).fetchall()


# ── Conversation detail screen ───────────────────────────────────────────────


class ConversationScreen(Screen):
    """Full read view of a single conversation, rendered as Markdown."""

    BINDINGS = [
        Binding("escape,backspace,q", "app.pop_screen", "Back"),
        Binding("j,down", "scroll_down", "Down", show=False),
        Binding("k,up", "scroll_up", "Up", show=False),
    ]

    def __init__(self, conn: sqlite3.Connection, conv_id: str) -> None:
        super().__init__()
        self._conn = conn
        self._conv_id = conv_id

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="conversation-body"):
            yield Static(RichMarkdown(self._transcript()))
        yield Footer()

    def _transcript(self) -> str:
        conv = _conversation(self._conn, self._conv_id)
        if conv is None:
            return "# Conversation not found"

        date = (conv["created_at"] or "?")[:10]
        meta = f"`{conv['platform']}`"
        if conv["model"]:
            meta += f" · `{conv['model']}`"
        meta += f" · {date}"

        lines = [f"# {conv['title'] or 'Untitled'}", "", meta, ""]
        for msg in _messages(self._conn, self._conv_id):
            who = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
            ts = (msg["timestamp"] or "")[:16]
            lines.append("---")
            lines.append(f"### {who}  ·  {ts}")
            lines.append("")
            lines.append(msg["content"] or "*(empty)*")
            lines.append("")
        return "\n".join(lines)

    def action_scroll_down(self) -> None:
        self.query_one("#conversation-body", VerticalScroll).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one("#conversation-body", VerticalScroll).scroll_up()


# ── Main search / browse screen ──────────────────────────────────────────────


class RicoeurApp(App):
    """Browse and search your conversation archive."""

    TITLE = "ricoeur"
    SUB_TITLE = "your conversation archive"

    CSS = """
    #search {
        dock: top;
        margin: 1 1 0 1;
    }
    #status {
        dock: top;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    DataTable {
        height: 1fr;
    }
    #conversation-body {
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("enter", "open_selected", "Open", show=False),
    ]

    def __init__(self, home: Optional[Path] = None) -> None:
        super().__init__()
        self._home = home or get_home()
        self._conn = get_connection(self._home)
        cfg = load_config()
        embed_cfg = cfg.get("embeddings", {})
        self._model_spec = embed_cfg.get("model", "")
        self._device = embed_cfg.get("device", "auto")
        # Maps the visible row index → conversation id
        self._row_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Search your conversations… (Enter to search, empty = recent)", id="search")
        yield Static("", id="status")
        table = DataTable(id="results", cursor_type="row", zebra_stripes=True)
        table.add_columns("Date", "Platform", "Msgs", "Title")
        yield table
        yield Footer()

    def on_mount(self) -> None:
        self._show_recent()

    # ── Populating the table ──────────────────────────────────────────────

    def _table(self) -> DataTable:
        return self.query_one("#results", DataTable)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _show_recent(self) -> None:
        try:
            rows = _recent_conversations(self._conn)
            n = self._conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
        except sqlite3.Error:
            # Database exists but isn't initialized / has no schema yet
            self._populate([], [])
            self._set_status("No conversations yet — import an export with `ricoeur import`.")
            return

        self._populate(
            [
                ((r["created_at"] or "?")[:10], r["platform"], str(r["message_count"] or 0), r["title"] or "Untitled")
                for r in rows
            ],
            [r["id"] for r in rows],
        )
        if n:
            self._set_status(f"{n:,} conversations · showing {len(rows)} most recent")
        else:
            self._set_status("No conversations yet — import an export with `ricoeur import`.")

    def _run_search(self, query: str) -> None:
        try:
            results, mode = search_dispatch(
                self._conn,
                query,
                self._home,
                keyword=True,  # keyword is instant and needs no embedding model
                model_spec=self._model_spec,
                device=self._device,
                limit=100,
            )
        except (RuntimeError, sqlite3.Error) as exc:
            self._set_status(f"[red]Search error:[/red] {exc}")
            return

        self._populate(
            [
                ((r.created_at or "?")[:10], r.platform, "", r.title or "Untitled")
                for r in results
            ],
            [r.conv_id for r in results],
        )
        self._set_status(f'{len(results)} result(s) for "{query}" ({mode})')

    def _populate(self, rows: list[tuple], ids: list[str]) -> None:
        table = self._table()
        table.clear()
        self._row_ids = ids
        for row in rows:
            table.add_row(*row)
        if rows:
            table.focus()
            table.move_cursor(row=0)

    # ── Actions ───────────────────────────────────────────────────────────

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    @on(Input.Submitted, "#search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if query:
            self._run_search(query)
        else:
            self._show_recent()

    @on(DataTable.RowSelected, "#results")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        self._open_row(event.cursor_row)

    def action_open_selected(self) -> None:
        table = self._table()
        if table.has_focus:
            self._open_row(table.cursor_row)

    def _open_row(self, row_index: int) -> None:
        if 0 <= row_index < len(self._row_ids):
            self.push_screen(ConversationScreen(self._conn, self._row_ids[row_index]))

    def on_unmount(self) -> None:
        self._conn.close()


def run(home: Optional[Path] = None) -> None:
    """Launch the ricoeur TUI."""
    RicoeurApp(home=home).run()
