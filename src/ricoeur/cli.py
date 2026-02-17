"""ricoeur CLI — Click-based command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from . import __version__
from .config import get_home, load_config, save_config, set_config_value, DEFAULT_CONFIG
from .db import get_connection, init_db

console = Console()


@click.group()
@click.version_option(__version__, prog_name="ricoeur")
def cli():
    """ricoeur — a local-first archive, search, and intelligence engine for LLM conversation history."""
    pass


# ── init ─────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--home", type=click.Path(), default=None, help="Data directory (default: ~/.ricoeur)")
def init(home: Optional[str]):
    """Initialize a new ricoeur database."""
    home_path = Path(home) if home else get_home()
    home_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir in ["analytics", "embeddings", "models", "attachments"]:
        (home_path / subdir).mkdir(exist_ok=True)

    # Create config if it doesn't exist
    config_file = home_path / "config.toml"
    if not config_file.exists():
        cfg = DEFAULT_CONFIG.copy()
        cfg["general"] = {**cfg["general"], "home": str(home_path)}
        save_config(cfg)

    # Initialize database
    init_db(home_path)

    console.print(f"[green]Initialized ricoeur at[/green] {home_path}")
    console.print(f"  Database:    {home_path / 'ricoeur.db'}")
    console.print(f"  Config:      {config_file}")


# ── import ───────────────────────────────────────────────────────────────


@cli.group("import")
def import_cmd():
    """Import conversation data from LLM platforms."""
    pass


@import_cmd.command("chatgpt")
@click.argument("path", type=click.Path(exists=True))
@click.option("--update", is_flag=True, help="Update existing conversations if reimporting")
@click.option("--since", default=None, help="Only import conversations after this date (ISO format)")
@click.option("--dry-run", is_flag=True, help="Parse and validate without writing to database")
def import_chatgpt(path: str, update: bool, since: Optional[str], dry_run: bool):
    """Import from a ChatGPT export (conversations.json or .zip)."""
    from .importers.chatgpt import import_chatgpt as do_import

    conn = get_connection()
    file_path = Path(path)

    with Progress(console=console) as progress:
        stats = do_import(conn, file_path, update=update, since=since, dry_run=dry_run, progress=progress)

    conn.close()
    _print_import_stats("ChatGPT", stats, dry_run)


@import_cmd.command("claude")
@click.argument("path", type=click.Path(exists=True))
@click.option("--update", is_flag=True, help="Update existing conversations if reimporting")
@click.option("--since", default=None, help="Only import conversations after this date (ISO format)")
@click.option("--dry-run", is_flag=True, help="Parse and validate without writing to database")
def import_claude_cmd(path: str, update: bool, since: Optional[str], dry_run: bool):
    """Import from a Claude export (conversations.json or .zip)."""
    from .importers.claude import import_claude as do_import

    conn = get_connection()
    file_path = Path(path)

    with Progress(console=console) as progress:
        stats = do_import(conn, file_path, update=update, since=since, dry_run=dry_run, progress=progress)

    conn.close()
    _print_import_stats("Claude", stats, dry_run)


def _print_import_stats(platform: str, stats, dry_run: bool):
    prefix = "[dim](dry run)[/dim] " if dry_run else ""
    console.print()
    console.print(f"{prefix}[bold]Import from {platform}[/bold]")
    console.print(f"  Parsed:       {stats.parsed:,} conversations ({stats.messages:,} messages)")
    console.print(f"  New:          {stats.new:,}")
    console.print(f"  Updated:      {stats.updated:,}")
    console.print(f"  Skipped:      {stats.skipped:,}")
    if stats.code_blocks:
        console.print(f"  Code blocks:  {stats.code_blocks:,} extracted")


# ── search ───────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--platform", default=None, help="Filter by platform")
@click.option("--lang", default=None, help="Filter by language")
@click.option("--since", default=None, help="Only conversations after this date")
@click.option("--until", default=None, help="Only conversations before this date")
@click.option("--model", default=None, help="Filter by LLM model")
@click.option("--topic", default=None, type=int, help="Filter by topic ID")
@click.option("--role", default=None, help="Filter by message role")
@click.option("--code", is_flag=True, help="Search only in code blocks")
@click.option("--limit", default=20, help="Max results")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "full", "ids"]))
def search(query, platform, lang, since, until, model, topic, role, code, limit, fmt):
    """Search across all conversations."""
    from .search import fts_search

    conn = get_connection()
    results = fts_search(
        conn,
        query,
        platform=platform,
        lang=lang,
        since=since,
        until=until,
        model=model,
        topic=topic,
        role=role,
        code=code,
        limit=limit,
    )
    conn.close()

    if not results:
        console.print(f"No results for \"{query}\"")
        return

    console.print(f"Found {len(results)} results for \"{query}\"\n")

    if fmt == "ids":
        for r in results:
            click.echo(r.conv_id)
    elif fmt == "json":
        click.echo(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    elif fmt == "full":
        for r in results:
            console.print(f"[bold]{r.title}[/bold]  ({r.platform}, {r.created_at})")
            if r.snippet:
                console.print(f"  {r.snippet}")
            console.print()
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", width=5)
        table.add_column("Date", width=10)
        table.add_column("Platform", width=8)
        table.add_column("Title")
        for i, r in enumerate(results, 1):
            date = r.created_at[:10] if r.created_at else "?"
            table.add_row(
                str(i),
                f"{r.score:.2f}",
                date,
                r.platform,
                r.title or "Untitled",
            )
        console.print(table)
        console.print("\nUse [bold]ricoeur show <id>[/bold] to read a conversation.")


# ── show ─────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("conversation_id")
@click.option("--messages", "msg_filter", default=None, type=click.Choice(["user", "assistant"]))
@click.option("--summary", is_flag=True, help="Show summary + metadata only")
@click.option("--code", is_flag=True, help="Only show code blocks")
@click.option("--format", "fmt", default="rich", type=click.Choice(["rich", "md", "json"]))
def show(conversation_id, msg_filter, summary, code, fmt):
    """Display a full conversation."""
    conn = get_connection()

    conv = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    if not conv:
        console.print(f"[red]Conversation not found:[/red] {conversation_id}")
        return

    if fmt == "json":
        msgs = conn.execute(
            "SELECT * FROM messages WHERE conv_id = ? ORDER BY timestamp", (conversation_id,)
        ).fetchall()
        data = {
            "conversation": dict(conv),
            "messages": [dict(m) for m in msgs],
        }
        click.echo(json.dumps(data, indent=2, default=str))
        conn.close()
        return

    # Rich / Markdown output
    console.rule(conv["title"] or "Untitled")
    console.print(f" Platform:  {conv['platform']:<16} Model:    {conv['model'] or '?'}")
    console.print(f" Date:      {(conv['created_at'] or '?')[:10]:<16} Messages: {conv['message_count'] or '?'}")
    if conv["language"]:
        console.print(f" Language:  {conv['language']}")
    console.rule()

    if summary:
        s = conn.execute(
            "SELECT summary FROM summaries WHERE conv_id = ?", (conversation_id,)
        ).fetchone()
        if s:
            console.print(f"\n{s['summary']}\n")
        else:
            console.print("\n[dim]No summary available. Run ricoeur index --summarize.[/dim]\n")
        conn.close()
        return

    if code:
        rows = conn.execute(
            """SELECT cb.language, cb.code FROM code_blocks cb
               JOIN messages m ON cb.msg_id = m.id
               WHERE m.conv_id = ?""",
            (conversation_id,),
        ).fetchall()
        for r in rows:
            console.print(f"\n[dim]```{r['language']}[/dim]")
            console.print(r["code"])
            console.print("[dim]```[/dim]")
        conn.close()
        return

    # Full message display
    query = "SELECT * FROM messages WHERE conv_id = ? ORDER BY timestamp"
    msgs = conn.execute(query, (conversation_id,)).fetchall()

    for msg in msgs:
        if msg_filter and msg["role"] != msg_filter:
            continue
        role_label = "[bold cyan]You[/bold cyan]" if msg["role"] == "user" else "[bold green]Assistant[/bold green]"
        ts = msg["timestamp"][:16] if msg["timestamp"] else ""
        console.print(f"\n {role_label}  [dim]{ts}[/dim]")
        console.print(f" {msg['content'][:2000]}")

    conn.close()


# ── stats ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--platform", default=None)
@click.option("--lang", default=None)
@click.option("--since", default=None)
@click.option("--until", default=None)
@click.option("--format", "fmt", default="rich", type=click.Choice(["rich", "json", "csv"]))
def stats(platform, lang, since, until, fmt):
    """Analytics dashboard."""
    conn = get_connection()

    where_clauses = []
    params: list = []
    if platform:
        where_clauses.append("platform = ?")
        params.append(platform)
    if lang:
        where_clauses.append("language = ?")
        params.append(lang)
    if since:
        where_clauses.append("created_at >= ?")
        params.append(since)
    if until:
        where_clauses.append("created_at <= ?")
        params.append(until)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    total_convs = conn.execute(f"SELECT COUNT(*) as n FROM conversations {where}", params).fetchone()["n"]
    total_msgs = conn.execute(
        f"SELECT COUNT(*) as n FROM messages WHERE conv_id IN (SELECT id FROM conversations {where})", params
    ).fetchone()["n"]

    platforms = conn.execute(
        f"SELECT platform, COUNT(*) as n FROM conversations {where} GROUP BY platform ORDER BY n DESC", params
    ).fetchall()
    models = conn.execute(
        f"SELECT model, COUNT(*) as n FROM conversations {where} GROUP BY model ORDER BY n DESC LIMIT 10", params
    ).fetchall()
    languages = conn.execute(
        f"SELECT language, COUNT(*) as n FROM conversations {where} GROUP BY language ORDER BY n DESC", params
    ).fetchall()

    if fmt == "json":
        data = {
            "total_conversations": total_convs,
            "total_messages": total_msgs,
            "platforms": {r["platform"]: r["n"] for r in platforms},
            "top_models": {r["model"] or "unknown": r["n"] for r in models},
            "languages": {r["language"] or "unknown": r["n"] for r in languages},
        }
        click.echo(json.dumps(data, indent=2))
        conn.close()
        return

    console.rule("[bold]ricoeur — your narrative in numbers[/bold]")
    console.print()
    console.print(f" Conversations    {total_convs:,}      Messages       {total_msgs:,}")
    console.print(f" Platforms        {len(platforms)}")

    if languages:
        lang_str = "  ".join(f"{r['language'] or '?'}: {r['n']}" for r in languages[:5])
        console.print(f" Languages        {lang_str}")

    console.print()
    if models:
        console.print(" [bold]Top models[/bold]")
        for r in models[:8]:
            console.print(f"   {r['model'] or 'unknown':<20} {r['n']:>5}")

    console.print()
    conn.close()


# ── index ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--embeddings", "do_embeddings", is_flag=True, help="Only generate embeddings")
@click.option("--topics", "do_topics", is_flag=True, help="Only run topic modeling")
@click.option("--analytics", "do_analytics", is_flag=True, help="Only export analytics")
@click.option("--force", is_flag=True, help="Redo everything, even if cached")
@click.option("--embed-model", default=None, help="Embedding model (e.g. st:model-name, ollama:model-name)")
@click.option("--batch-size", default=None, type=int, help="Batch size for embedding")
@click.option("--device", default=None, type=click.Choice(["auto", "cpu", "cuda", "mps"]), help="Device for embeddings")
def index(do_embeddings, do_topics, do_analytics, force, embed_model, batch_size, device):
    """Build the intelligence layer: language detection, embeddings, analytics."""
    from .indexer import run_index

    home = get_home()
    cfg = load_config()

    # Resolve defaults from config
    embed_cfg = cfg.get("embeddings", {})
    if embed_model is None:
        embed_model = embed_cfg.get("model", "st:paraphrase-multilingual-mpnet-base-v2")
    if batch_size is None:
        batch_size = embed_cfg.get("batch_size", 64)
    if device is None:
        device = embed_cfg.get("device", "auto")

    # Flag semantics: no flags → run all available layers
    any_flag = do_embeddings or do_topics or do_analytics
    do_languages = not any_flag
    if not any_flag:
        do_embeddings = True
        do_topics = True
        do_analytics = True

    conn = get_connection()

    # Rich progress tracking
    with Progress(console=console) as progress:
        tasks = {}

        def progress_cb(layer: str, current: int, total: int):
            if layer not in tasks:
                if total > 0:
                    tasks[layer] = progress.add_task(
                        f"[cyan]{layer}", total=total
                    )
            if layer in tasks:
                progress.update(tasks[layer], completed=current)

        stats = run_index(
            conn,
            home,
            do_languages=do_languages,
            do_embeddings=do_embeddings,
            do_topics=do_topics,
            do_analytics=do_analytics,
            embed_model=embed_model,
            batch_size=batch_size,
            device=device,
            force=force,
            progress_cb=progress_cb,
        )

    conn.close()
    _print_index_stats(stats)


def _print_index_stats(stats):
    """Print a summary table of indexing results."""
    from .indexer import IndexStats

    console.print()
    console.rule("[bold]Index summary[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Layer", style="cyan")
    table.add_column("Result", justify="right")

    if stats.languages_detected > 0 or stats.languages_skipped == 0:
        table.add_row("Languages", f"{stats.languages_detected:,} detected")
    if stats.embeddings_generated > 0 or stats.embeddings_skipped > 1:
        embed_detail = f"{stats.embeddings_generated:,} generated"
        if stats.embeddings_skipped > 1:
            embed_detail += f", {stats.embeddings_skipped:,} cached"
        table.add_row("Embeddings", embed_detail)
        if stats.embedding_model:
            table.add_row("  model", stats.embedding_model)
        if stats.embedding_dimensions:
            table.add_row("  dimensions", str(stats.embedding_dimensions))
    if stats.analytics_conversations > 0:
        table.add_row(
            "Analytics",
            f"{stats.analytics_conversations:,} convs, {stats.analytics_messages:,} msgs",
        )

    console.print(table)

    # Print errors / warnings
    for err in stats.errors:
        if "not yet implemented" in err:
            console.print(f"  [dim]{err}[/dim]")
        elif "not installed" in err:
            console.print(f"  [yellow]⚠ {err}[/yellow]")
        else:
            console.print(f"  [red]✗ {err}[/red]")

    console.print()


# ── config ───────────────────────────────────────────────────────────────


@cli.group()
def config():
    """View and edit configuration."""
    pass


@config.command("show")
def config_show():
    """Print current configuration."""
    import pprint
    cfg = load_config()
    console.print_json(json.dumps(cfg, default=str))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a config value (e.g. ricoeur config set embed_model ollama:nomic-embed-text)."""
    try:
        set_config_value(key, value)
        console.print(f"[green]Set[/green] {key} = {value}")
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")


# ── version ──────────────────────────────────────────────────────────────


@cli.command()
def version():
    """Print version info."""
    console.print(f"ricoeur {__version__}")
