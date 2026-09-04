"""Claude Code session importer.

Unlike the ChatGPT and Claude importers, this one reads no export file. Claude
Code writes its own transcript as it works, to::

    ~/.claude/projects/<slugified-cwd>/<sessionId>.jsonl

one JSON record per line, appended live. So these sessions need no export step
— ricoeur can read them straight from disk and stay current — but they are
*append-only and possibly mid-write*, which shapes the whole design:

* records are streamed, never ``json.load``-ed (a month of sessions is ~100 MB)
* a truncated final line is tolerated: that's a session in flight, not corruption
* ``messages.id`` is the record's own UUID, so re-import is idempotent and a
  growing session appends rows instead of replacing them

Only ``user`` and ``assistant`` records carry conversation content; the ~15
other record types are harness/UI state. Thinking blocks are persisted with
their ``signature`` but their text stripped, so they render to nothing — which
is why an assistant record can legitimately produce no message at all. Crucially, most ``user`` records are
*not* human turns — they carry ``tool_result`` blocks feeding tool output back
to the model. Counting those as user messages would poison both
``search --role user`` and every number in ``ricoeur stats``, so they are
dropped unless ``include_tool_results`` asks for the full trace.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from rich.progress import Progress

from .base import (
    ImportStats,
    insert_attachment,
    insert_message,
    make_message_id,
    render_thinking,
    render_tool_block,
    update_conversation_counts,
    upsert_conversation,
)

PLATFORM = "claude-code"

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# The only record types that carry conversation content.
CONTENT_TYPES = ("user", "assistant")

# Claude Code attributes harness-generated messages (interrupts, API errors)
# to this pseudo-model. They are not authored by anyone.
SYNTHETIC_MODEL = "<synthetic>"

# Claude Code wraps several non-prose gestures in XML inside a user message.
# A slash command arrives as command-name / command-message / command-args, in
# either order; the ``!`` gesture as bash-input plus its captured output.
_COMMAND_NAME_RE = re.compile(r"<command-name>(?P<name>[^<]*)</command-name>")
_COMMAND_ARGS_RE = re.compile(r"<command-args>(?P<args>.*?)</command-args>", re.DOTALL)
_COMMAND_MESSAGE_RE = re.compile(
    r"<command-message>.*?</command-message>", re.DOTALL
)
_BASH_INPUT_RE = re.compile(r"<bash-input>(?P<command>.*?)</bash-input>", re.DOTALL)

# Envelopes the harness injects that no human wrote: local command output,
# background-task notifications, captured stdout/stderr.
_HARNESS_NOISE_RE = re.compile(
    r"<(local-command-stdout|task-notification|bash-stdout|bash-stderr)>"
    r".*?</\1>",
    re.DOTALL,
)

# Fence languages for file contents written by tools.
EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript",
    ".tsx": "tsx", ".rs": "rust", ".go": "go", ".rb": "ruby", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "fish",
    ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".rst": "rst", ".tex": "latex", ".r": "r",
    ".m": "matlab", ".lua": "lua", ".pl": "perl", ".php": "php",
    ".ipynb": "json", ".dockerfile": "dockerfile", ".tf": "hcl",
}

TITLE_MAX = 72


# ── Public entry point ───────────────────────────────────────────────────────


def import_claude_code(
    conn: sqlite3.Connection,
    path: Optional[Path] = None,
    *,
    update: bool = False,
    since: Optional[str] = None,
    dry_run: bool = False,
    project: Optional[str] = None,
    include_tool_results: bool = False,
    include_sidechains: bool = False,
    progress: Optional[Progress] = None,
) -> ImportStats:
    """Import Claude Code sessions.

    Args:
        path: A projects root, a single project directory, or one ``.jsonl``
            session file. Defaults to ``~/.claude/projects``.
        update: Re-read every session, even ones whose file is unchanged since
            the last import. Never destructive — sessions only ever gain rows.
            Implied by ``include_tool_results`` and ``include_sidechains``,
            which ask for content a previous import dropped.
        since: Skip sessions that started before this ISO date.
        dry_run: Parse and report without writing.
        project: Only import sessions whose project path matches this substring.
        include_tool_results: Keep tool output in the transcript (~30x larger).
        include_sidechains: Include subagent transcripts.

    ``stats.messages`` counts messages actually archived, so a re-import of
    unchanged sessions reports zero (on a dry run, where nothing is written, it
    counts what would be considered).
    """
    stats = ImportStats()
    root = Path(path) if path is not None else DEFAULT_PROJECTS_DIR

    if not root.exists():
        raise FileNotFoundError(
            f"No Claude Code sessions at {root}. "
            "Claude Code writes them to ~/.claude/projects/ as you use it."
        )

    files = list(_session_files(root, project))
    stats.parsed = len(files)

    # The mtime/size cache keys on the file alone, so it would wrongly skip a
    # session whose *options* changed — asking for tool results or sidechains
    # means asking for content a cached import deliberately dropped.
    use_cache = not (update or dry_run or include_tool_results or include_sidechains)

    task = None
    if progress:
        task = progress.add_task("Importing Claude Code...", total=len(files))

    for file in files:
        _import_session(
            conn,
            file,
            stats,
            since=since,
            dry_run=dry_run,
            use_cache=use_cache,
            include_tool_results=include_tool_results,
            include_sidechains=include_sidechains,
        )
        if progress and task is not None:
            progress.advance(task)

    if not dry_run:
        update_conversation_counts(conn)
        conn.commit()

    return stats


def _session_files(root: Path, project: Optional[str]) -> Iterator[Path]:
    """Resolve ``root`` to session files.

    Accepts a single ``.jsonl`` file, a project directory, or the projects root
    (whose children are per-project directories).
    """
    if root.is_file():
        candidates: Iterable[Path] = [root]
    else:
        own = sorted(root.glob("*.jsonl"))
        nested = sorted(root.glob("*/*.jsonl"))
        candidates = own + nested

    for file in candidates:
        if project and project.lower() not in str(file.parent).lower():
            continue
        yield file


# ── Parsed shapes ────────────────────────────────────────────────────────────


@dataclass
class _Message:
    id: str
    role: str
    content: str
    timestamp: Optional[str]
    content_type: str = "text"
    token_count: Optional[int] = None
    images: int = 0


@dataclass
class _Session:
    """Everything one session log tells us, after filtering."""

    id: Optional[str] = None
    ai_title: Optional[str] = None
    cwd: Optional[str] = None
    models: Counter = field(default_factory=Counter)
    messages: list[_Message] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)

    @property
    def created_at(self) -> Optional[str]:
        return min(self.timestamps) if self.timestamps else None

    @property
    def updated_at(self) -> Optional[str]:
        return max(self.timestamps) if self.timestamps else None

    @property
    def model(self) -> Optional[str]:
        """The session's dominant model — sessions can switch mid-flight."""
        if not self.models:
            return None
        return self.models.most_common(1)[0][0]


# ── Per-session import ───────────────────────────────────────────────────────


def _import_session(
    conn: sqlite3.Connection,
    file: Path,
    stats: ImportStats,
    *,
    since: Optional[str],
    dry_run: bool,
    use_cache: bool,
    include_tool_results: bool,
    include_sidechains: bool,
) -> None:
    if use_cache and _unchanged(conn, file):
        stats.skipped += 1
        return

    session = _parse_session(
        file,
        stats,
        include_tool_results=include_tool_results,
        include_sidechains=include_sidechains,
    )

    if not session.messages or not session.id:
        # Harness-state-only logs — 4 of my 87 files. Not an error.
        stats.skipped += 1
        return

    created_at = session.created_at
    if since and created_at and created_at < since:
        stats.skipped += 1
        return

    if dry_run:
        stats.new += 1
        stats.messages += len(session.messages)
        return

    project = _project_name(session, file)
    outcome = upsert_conversation(
        conn,
        id=session.id,
        title=_resolve_title(session, project),
        platform=PLATFORM,
        model=session.model,
        created_at=created_at,
        updated_at=session.updated_at,
        project=project,
        source_path=str(file),
    )
    if outcome == "new":
        stats.new += 1
    else:
        stats.updated += 1

    for msg in session.messages:
        inserted, blocks = insert_message(
            conn,
            id=msg.id,
            conv_id=session.id,
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            content_type=msg.content_type,
            token_count=msg.token_count,
        )
        stats.code_blocks += blocks
        if not inserted:
            # Already archived on an earlier run — a re-read, not a new turn.
            continue
        stats.messages += 1
        for _ in range(msg.images):
            insert_attachment(
                conn,
                conv_id=session.id,
                msg_id=msg.id,
                type="image",
            )
            stats.attachments += 1

    _record_import(conn, file, session.id)


def _parse_session(
    file: Path,
    stats: ImportStats,
    *,
    include_tool_results: bool,
    include_sidechains: bool,
) -> _Session:
    """Stream one session log, keeping only what is conversation."""
    session = _Session()
    index = 0

    with open(file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A session being written right now, or a partial flush.
                stats.malformed += 1
                continue
            if not isinstance(record, dict):
                stats.malformed += 1
                continue

            if session.id is None and record.get("sessionId"):
                session.id = record["sessionId"]

            rtype = record.get("type")
            if rtype == "ai-title":
                # Later titles supersede earlier ones.
                session.ai_title = record.get("aiTitle") or session.ai_title
                continue
            if rtype not in CONTENT_TYPES:
                continue

            msg = _content_message(
                record,
                rtype,
                index,
                session,
                include_tool_results=include_tool_results,
                include_sidechains=include_sidechains,
            )
            if msg is not None:
                session.messages.append(msg)
                index += 1

    if session.id is None:
        session.id = file.stem or None
    return session


def _content_message(
    record: dict[str, Any],
    rtype: str,
    index: int,
    session: _Session,
    *,
    include_tool_results: bool,
    include_sidechains: bool,
) -> Optional[_Message]:
    """Turn one ``user``/``assistant`` record into a message, or ``None``."""
    if record.get("isSidechain") and not include_sidechains:
        return None
    if record.get("isMeta"):
        # Injected system context (system-reminders), authored by no one.
        return None
    if record.get("isApiErrorMessage"):
        return None

    if session.cwd is None and record.get("cwd"):
        session.cwd = record["cwd"]

    message = record.get("message")
    if not isinstance(message, dict):
        return None

    model = message.get("model")
    if model == SYNTHETIC_MODEL:
        return None
    if isinstance(model, str) and model:
        session.models[model] += 1

    content, images = _extract_content(
        message, include_tool_results=include_tool_results
    )
    if not content:
        return None

    timestamp = record.get("timestamp")
    if timestamp:
        session.timestamps.append(timestamp)

    role = message.get("role") or rtype
    if role == "human":
        role = "user"

    return _Message(
        id=record.get("uuid") or make_message_id(session.id or str(index), index),
        role=role,
        content=content,
        timestamp=timestamp,
        content_type="compact_summary" if record.get("isCompactSummary") else "text",
        token_count=_output_tokens(message),
        images=images,
    )


def _output_tokens(message: dict[str, Any]) -> Optional[int]:
    """Tokens in this message — ``usage.output_tokens`` for assistant turns."""
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("output_tokens")
    return value if isinstance(value, int) else None


# ── Content extraction ───────────────────────────────────────────────────────


def _extract_content(
    message: dict[str, Any], *, include_tool_results: bool
) -> tuple[str, int]:
    """Render a message's content blocks. Returns ``(text, image_count)``."""
    content = message.get("content")

    if isinstance(content, str):
        return _clean_prompt(content), 0
    if not isinstance(content, list):
        return "", 0

    parts: list[str] = []
    images = 0
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue

        btype = block.get("type")
        if btype == "text":
            parts.append(_clean_prompt(block.get("text") or ""))
        elif btype == "thinking":
            parts.append(render_thinking(block.get("thinking") or ""))
        elif btype == "tool_use":
            parts.append(_render_tool_use(block))
        elif btype == "image":
            images += 1
        elif btype == "tool_result":
            images += _result_images(block)
            if include_tool_results:
                parts.append(_render_tool_result(block))
        # Every other block type is interface state.

    return "\n\n".join(p for p in parts if p and p.strip()), images


def _clean_prompt(text: str) -> str:
    """Reduce a user message to what the human actually wrote.

    Claude Code injects envelopes into user records — slash-command XML, the
    ``!`` bash gesture and its output, background-task notifications. Those are
    harness plumbing, not prompts, so they're stripped or rendered as the
    gesture they represent. A message left empty by this is dropped upstream.
    """
    if not text:
        return ""

    text = _HARNESS_NOISE_RE.sub("", text)

    command = _slash_command(text)
    if command is not None:
        text = _COMMAND_MESSAGE_RE.sub("", text)
        text = _COMMAND_NAME_RE.sub("", text)
        text = _COMMAND_ARGS_RE.sub("", text)
        text = f"{command}\n\n{text.strip()}"

    text = _BASH_INPUT_RE.sub(
        lambda m: render_tool_block("!", m.group("command"), lang="bash"), text
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _slash_command(text: str) -> Optional[str]:
    """Render a slash-command invocation as the command line the user typed."""
    name = _COMMAND_NAME_RE.search(text)
    if not name:
        return None
    command = name.group("name").strip()
    if not command:
        return None

    args_match = _COMMAND_ARGS_RE.search(text)
    args = args_match.group("args").strip() if args_match else ""
    if not args:
        return f"`{command}`"
    if "\n" in args:
        return f"`{command}`\n\n{args}"
    return f"`{command} {args}`"


def _result_images(block: dict[str, Any]) -> int:
    content = block.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for b in content if isinstance(b, dict) and b.get("type") == "image"
    )


def _render_tool_result(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, list):
        content = "\n".join(
            b.get("text") or ""
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if not isinstance(content, str) or not content.strip():
        return ""
    return render_tool_block("result", content)


# ── Tool rendering ───────────────────────────────────────────────────────────


def _render_tool_use(block: dict[str, Any]) -> str:
    """Render a tool call that carries real content.

    Claude Code's tool inputs are per-tool and structured (``Bash`` has a
    ``command``, ``Write`` a ``content``, ``Edit`` a string swap), so unlike the
    web export there is no single content key to reach for. Tools that carry no
    content of their own — ``Read``, ``Glob``, ``Grep`` — render to nothing,
    the same way interface-only tools do in the web importer.

    The content that *is* rendered comes out fenced, which is what puts the code
    Claude Code actually wrote into ``code_blocks`` and within reach of
    ``ricoeur search --code``.
    """
    name = block.get("name") or "tool"
    inp = block.get("input")
    if not isinstance(inp, dict):
        return ""

    label = _tool_label(name)
    path = inp.get("file_path") or inp.get("notebook_path")
    lang = _lang_for(path)

    if name == "Bash":
        return render_tool_block(
            label,
            inp.get("command") or "",
            lang="bash",
            title=_str_or_none(inp.get("description")),
        )

    if name in ("Write", "NotebookEdit"):
        body = inp.get("content") or inp.get("new_source") or ""
        return render_tool_block(label, body, lang=lang, title=_str_or_none(path))

    if name == "Edit":
        return render_tool_block(
            label, _as_diff(inp), lang="diff", title=_str_or_none(path)
        )

    if name in ("Task", "Agent"):
        # A subagent prompt is prose — labelled, but deliberately not fenced,
        # so it doesn't land in code_blocks as junk.
        prompt = _str_or_none(inp.get("prompt"))
        if not prompt:
            return ""
        title = _str_or_none(inp.get("description"))
        header = f"🛠️ **Tool · {label}**" + (f" — {title}" if title else "")
        return f"{header}\n\n{prompt.strip()}"

    # Generic fallback: fence whatever content-shaped string the tool carries.
    for key in ("content", "code", "command", "query"):
        body = inp.get(key)
        if isinstance(body, str) and body.strip():
            return render_tool_block(label, body, lang=lang, title=_str_or_none(path))
    return ""


def _tool_label(name: str) -> str:
    """``mcp__skore__skore_agent`` reads better as ``skore · skore_agent``."""
    if name.startswith("mcp__"):
        parts = [p for p in name[len("mcp__"):].split("__") if p]
        if parts:
            return " · ".join(parts)
    return name


def _as_diff(inp: dict[str, Any]) -> str:
    """Render an ``Edit`` as a unified-ish diff, which is how it reads best."""
    old = inp.get("old_string")
    new = inp.get("new_string")
    lines: list[str] = []
    if isinstance(old, str) and old:
        lines += [f"-{line}" for line in old.splitlines()]
    if isinstance(new, str) and new:
        lines += [f"+{line}" for line in new.splitlines()]
    return "\n".join(lines)


def _lang_for(path: Any) -> str:
    if not isinstance(path, str) or not path:
        return ""
    return EXT_LANG.get(Path(path).suffix.lower(), "")


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


# ── Titles and provenance ────────────────────────────────────────────────────


def _resolve_title(session: _Session, project: Optional[str]) -> str:
    """Claude Code's own AI title, else the opening prompt, else the project."""
    if session.ai_title and session.ai_title.strip():
        return session.ai_title.strip()

    for msg in session.messages:
        if msg.role != "user":
            continue
        first_line = next(
            (line.strip().strip("`") for line in msg.content.splitlines() if line.strip()),
            "",
        )
        if first_line:
            if len(first_line) > TITLE_MAX:
                first_line = first_line[: TITLE_MAX - 1].rstrip() + "…"
            return first_line

    return project or "Untitled session"


def _project_name(session: _Session, file: Path) -> Optional[str]:
    """The repo the session ran in — the most useful axis for slicing history."""
    if session.cwd:
        name = Path(session.cwd).name
        if name:
            return name
    # No cwd on any record: fall back to the (slugified) directory name.
    parent = file.parent.name
    return parent or None


# ── Import bookkeeping ───────────────────────────────────────────────────────


def _unchanged(conn: sqlite3.Connection, file: Path) -> bool:
    """True if this file is byte-identical in size and mtime to last import.

    A pure speed cache — message UUIDs already make re-import idempotent — so a
    stale or missing row here costs time, never correctness.
    """
    row = conn.execute(
        "SELECT mtime, size FROM import_sources WHERE path = ?", (str(file),)
    ).fetchone()
    if row is None:
        return False
    try:
        stat = file.stat()
    except OSError:
        return False
    mtime, size = row[0], row[1]
    if mtime is None or size is None:
        return False
    return size == stat.st_size and abs(mtime - stat.st_mtime) < 1e-6


def _record_import(conn: sqlite3.Connection, file: Path, conv_id: str) -> None:
    try:
        stat = file.stat()
    except OSError:
        return
    conn.execute(
        """INSERT OR REPLACE INTO import_sources(path, conv_id, mtime, size, imported_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            str(file),
            conv_id,
            stat.st_mtime,
            stat.st_size,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
