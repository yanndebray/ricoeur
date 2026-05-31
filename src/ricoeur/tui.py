"""Terminal UI for ricoeur, built with Textual.

Browse, search, and read your conversation archive interactively. The TUI is a
thin shell over the same data layer the CLI uses (``db`` + ``search``), so it
needs no extra state of its own.

Launch with ``ricoeur tui`` (requires ``pip install 'ricoeur[tui]'``).
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from .config import get_home, load_config
from .db import get_connection
from .search import search_dispatch

try:
    from rich.markdown import Markdown as RichMarkdown
    from rich.text import Text
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
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


def _recent_conversations(
    conn: sqlite3.Connection, limit: Optional[int] = None
) -> list[sqlite3.Row]:
    """Conversations newest-first — the default browse view.

    With ``limit=None`` (the default) every conversation is returned so the
    table is fully scrollable; pass an integer to cap the result set.
    """
    sql = """SELECT id, title, platform, model, created_at, message_count
             FROM conversations
             ORDER BY created_at DESC"""
    if limit is None:
        return conn.execute(sql).fetchall()
    return conn.execute(sql + "\n           LIMIT ?", (limit,)).fetchall()


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


# ── Startup splash ───────────────────────────────────────────────────────────

# ANSI_Shadow wordmark — painted on, left to right, when the TUI opens.
_WORDMARK = [
    "██████╗ ██╗ ██████╗ ██████╗ ███████╗██╗   ██╗██████╗ ",
    "██╔══██╗██║██╔════╝██╔═══██╗██╔════╝██║   ██║██╔══██╗",
    "██████╔╝██║██║     ██║   ██║█████╗  ██║   ██║██████╔╝",
    "██╔══██╗██║██║     ██║   ██║██╔══╝  ██║   ██║██╔══██╗",
    "██║  ██║██║╚██████╗╚██████╔╝███████╗╚██████╔╝██║  ██║",
    "╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝",
]
_WORDMARK_NARROW = ["r  i  c  o  e  u  r"]
_TAGLINE = "your conversation archive"

# Palette (mirrors the website).
_EMERALD = "#52b788"
_EMERALD_DIM = "#2d6a4f"
_TERRACOTTA = "#d4916f"
_MUTED = "#8a9b93"

# Animation cadence (frames at _FPS). Total ≈ 2.5s including the fade-out.
_FPS = 24
_WIPE_FRAMES = 12     # wordmark painted on by ~0.5s
_TAG_START = 15       # tagline begins typing
_TAG_FRAMES = 22      # …and finishes ~0.9s later
_FOOT_START = 40      # conversation count fades in
_END_FRAME = 54       # ~2.25s, then a 0.3s fade-out


class SplashScreen(Screen):
    """A brief, skippable opening animation shown when the TUI launches."""

    CSS = """
    SplashScreen { align: center middle; background: #0f1c17; }
    #splash-card { width: auto; height: auto; align: center middle; }
    #splash-mark { width: auto; }
    #splash-rule { width: auto; color: #d4916f; opacity: 0; margin-top: 1; }
    #splash-tag  { width: auto; color: #8a9b93; margin-top: 1; }
    #splash-foot { width: auto; color: #52b788; opacity: 0; margin-top: 1; }
    #splash-hint {
        dock: bottom; width: 100%; text-align: center;
        color: #3a4d45; padding-bottom: 1;
    }
    """

    def __init__(self, count: Optional[int] = None) -> None:
        super().__init__()
        self._count = count
        self._frame = 0
        self._settled = False
        self._done = False
        self._timer = None
        self._mark = list(_WORDMARK)
        self._mark_w = max(len(line) for line in self._mark)

    def compose(self) -> ComposeResult:
        with Vertical(id="splash-card"):
            yield Static("", id="splash-mark")
            yield Static("", id="splash-rule")
            yield Static("", id="splash-tag")
            yield Static("", id="splash-foot")
        yield Static("press any key to skip", id="splash-hint")

    def on_mount(self) -> None:
        w, h = self.size.width, self.size.height
        # Tiny terminals: don't bother — go straight to the app.
        if w < 16 or h < 8:
            self._finish()
            return
        if w < len(_WORDMARK[0]) + 4:
            self._mark = list(_WORDMARK_NARROW)
        self._mark_w = max(len(line) for line in self._mark)
        self._mark = [line.ljust(self._mark_w) for line in self._mark]

        self.query_one("#splash-rule", Static).update("─" * min(self._mark_w, 38))
        self._timer = self.set_interval(1 / _FPS, self._tick)

    # ── Frame rendering ───────────────────────────────────────────────────

    def _mark_text(self, reveal: int, settled: bool) -> Text:
        """The wordmark revealed up to ``reveal`` columns. Until settled, the
        leading edge glows terracotta (the 'pen'); the painted body trails in
        dim emerald, then brightens once fully revealed."""
        head = 3
        t = Text(no_wrap=True)
        for idx, line in enumerate(self._mark):
            if idx:
                t.append("\n")
            if settled:
                t.append(line, style=f"bold {_EMERALD}")
                continue
            shown = line[:reveal]
            if reveal > head:
                t.append(shown[: reveal - head], style=f"bold {_EMERALD_DIM}")
                t.append(shown[reveal - head:], style=f"bold {_TERRACOTTA}")
            else:
                t.append(shown, style=f"bold {_TERRACOTTA}")
            t.append(" " * (self._mark_w - reveal))  # keep the block width stable
        return t

    def _tag_text(self, n: int, cursor: bool) -> Text:
        t = Text(style=_MUTED)
        t.append(_TAGLINE[:n])
        if cursor:
            t.append("▌", style=_EMERALD)
        return t

    def _foot_text(self) -> Text:
        if self._count:
            return Text(f"✦  {self._count:,} conversations  ✦", style=_EMERALD)
        return Text("✦  opening your archive  ✦", style=_EMERALD)

    def _tick(self) -> None:
        if self._done:
            return
        self._frame += 1
        f = self._frame

        if f <= _WIPE_FRAMES:
            reveal = round(self._mark_w * f / _WIPE_FRAMES)
            self.query_one("#splash-mark", Static).update(self._mark_text(reveal, False))
        elif not self._settled:
            self._settled = True
            self.query_one("#splash-mark", Static).update(self._mark_text(self._mark_w, True))
            self.query_one("#splash-rule", Static).styles.animate(
                "opacity", value=1.0, duration=0.4
            )

        if f >= _TAG_START:
            n = min(len(_TAGLINE), round((f - _TAG_START) * len(_TAGLINE) / _TAG_FRAMES))
            self.query_one("#splash-tag", Static).update(
                self._tag_text(n, cursor=(f // 6) % 2 == 0)
            )

        if f == _FOOT_START:
            foot = self.query_one("#splash-foot", Static)
            foot.update(self._foot_text())
            foot.styles.animate("opacity", value=1.0, duration=0.5)

        if f >= _END_FRAME:
            self._finish()

    # ── Teardown ──────────────────────────────────────────────────────────

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        if self._timer is not None:
            self._timer.stop()
        try:
            self.styles.animate("opacity", value=0.0, duration=0.3, on_complete=self.dismiss)
        except Exception:  # pragma: no cover - fade is cosmetic
            self.dismiss()

    def on_key(self, event) -> None:
        # Any key skips the intro.
        event.stop()
        self._finish()


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

    def __init__(self, home: Optional[Path] = None, splash: Optional[bool] = None) -> None:
        super().__init__()
        self._home = home or get_home()
        self._conn = get_connection(self._home)
        cfg = load_config()
        embed_cfg = cfg.get("embeddings", {})
        self._model_spec = embed_cfg.get("model", "")
        self._device = embed_cfg.get("device", "auto")
        # Maps the visible row index → conversation id
        self._row_ids: list[str] = []
        # Show the opening animation only for real interactive terminals
        # (so it never disrupts piped output or the headless test pilot),
        # unless explicitly overridden or disabled via RICOEUR_NO_SPLASH.
        if splash is None:
            disabled = os.environ.get("RICOEUR_NO_SPLASH", "").lower() in ("1", "true", "yes")
            splash = not disabled and bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._splash = splash

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
        if self._splash:
            self.push_screen(SplashScreen(self._conversation_count()), self._after_splash)

    def _conversation_count(self) -> Optional[int]:
        try:
            return self._conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
        except sqlite3.Error:
            return None

    def _after_splash(self, _result: object = None) -> None:
        try:
            self._table().focus()
        except Exception:  # pragma: no cover - focus is best-effort
            pass

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
            self._set_status(f"{n:,} conversations · newest first")
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
