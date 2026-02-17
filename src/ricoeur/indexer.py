"""Indexing pipeline for ricoeur — language detection, embeddings, analytics."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class IndexStats:
    """Results from an indexing run."""

    languages_detected: int = 0
    languages_skipped: int = 0
    embeddings_generated: int = 0
    embeddings_skipped: int = 0
    embedding_model: str = ""
    embedding_dimensions: int = 0
    analytics_conversations: int = 0
    analytics_messages: int = 0
    errors: list[str] = field(default_factory=list)


# ── Dependency checking ──────────────────────────────────────────────────


def check_dependency(name: str) -> bool:
    """Check if optional dependencies for a layer are importable."""
    try:
        if name == "langdetect":
            import fast_langdetect  # noqa: F401

            return True
        elif name == "embeddings":
            import numpy  # noqa: F401
            import sentence_transformers  # noqa: F401

            return True
        elif name == "analytics":
            import pyarrow  # noqa: F401

            return True
        elif name == "ollama":
            import ollama  # noqa: F401

            return True
    except ImportError:
        pass
    return False


_INSTALL_HINTS = {
    "langdetect": "uv pip install 'ricoeur[langdetect]'",
    "embeddings": "uv pip install 'ricoeur[embeddings]'",
    "analytics": "uv pip install 'ricoeur[analytics]'",
    "ollama": "pip install ollama  # and ensure ollama server is running",
}


# ── Language detection ───────────────────────────────────────────────────


_MIN_TEXT_LENGTH = 20


def _detect_language(text: str) -> str:
    """Detect the primary language of a text snippet.

    Returns an ISO 639-1 code (e.g. 'en', 'fr') or 'unknown'.
    """
    if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
        return "unknown"
    try:
        from fast_langdetect import detect

        result = detect(text)
        if result and len(result) > 0:
            return result[0]["lang"]
    except Exception:
        pass
    return "unknown"


def detect_languages(
    conn: sqlite3.Connection,
    *,
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Detect languages for conversations and update the DB.

    Args:
        conn: SQLite connection.
        force: If True, re-detect even if language is already set.
        progress: Callback(current, total) for progress reporting.

    Returns:
        Number of conversations updated.
    """
    if force:
        rows = conn.execute("SELECT id FROM conversations").fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM conversations WHERE language IS NULL"
        ).fetchall()

    conv_ids = [r["id"] for r in rows]
    total = len(conv_ids)

    if progress:
        progress(0, total)

    updated = 0
    for i, conv_id in enumerate(conv_ids):
        # Concatenate user messages for this conversation
        msgs = conn.execute(
            "SELECT content FROM messages WHERE conv_id = ? AND role = 'user' ORDER BY timestamp",
            (conv_id,),
        ).fetchall()
        text = " ".join(m["content"] for m in msgs if m["content"])

        lang = _detect_language(text)
        conn.execute(
            "UPDATE conversations SET language = ? WHERE id = ?",
            (lang, conv_id),
        )
        updated += 1

        # Commit in batches of 500
        if updated % 500 == 0:
            conn.commit()

        if progress:
            progress(i + 1, total)

    conn.commit()
    return updated


# ── Embeddings ───────────────────────────────────────────────────────────


def _build_conversation_text(
    conn: sqlite3.Connection, conv_id: str, max_chars: int = 10_000
) -> str:
    """Concatenate messages into a single text block for embedding."""
    msgs = conn.execute(
        "SELECT role, content FROM messages WHERE conv_id = ? ORDER BY timestamp",
        (conv_id,),
    ).fetchall()
    parts = []
    char_count = 0
    for m in msgs:
        if not m["content"]:
            continue
        line = f"{m['role']}: {m['content']}"
        if char_count + len(line) > max_chars:
            remaining = max_chars - char_count
            if remaining > 0:
                parts.append(line[:remaining])
            break
        parts.append(line)
        char_count += len(line)
    return "\n".join(parts)


def _encode_sentence_transformers(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> Any:  # numpy.ndarray
    """Encode texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device if device != "auto" else None)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def _encode_ollama(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> Any:  # numpy.ndarray
    """Encode texts using Ollama's embedding API."""
    import numpy as np
    import ollama

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Add clustering prefix per document (matching NLP sprint pattern)
        prefixed = [f"clustering: {t}" for t in batch]
        response = ollama.embed(model=model_name, input=prefixed)
        all_embeddings.extend(response["embeddings"])
    return np.array(all_embeddings, dtype=np.float32)


def _parse_model_spec(model_str: str) -> tuple[str, str]:
    """Parse model spec like 'st:model-name' or 'ollama:model-name'.

    Returns (backend, model_name).
    """
    if ":" in model_str:
        backend, name = model_str.split(":", 1)
        return backend, name
    # Default to sentence-transformers
    return "st", model_str


def generate_embeddings(
    conn: sqlite3.Connection,
    home: Path,
    *,
    model: str,
    batch_size: int = 64,
    device: str = "auto",
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, int]:
    """Generate conversation embeddings.

    Args:
        conn: SQLite connection.
        home: Ricoeur home directory.
        model: Model spec like 'st:model-name' or 'ollama:model-name'.
        batch_size: Encoding batch size.
        device: Device for sentence-transformers ('auto', 'cpu', 'cuda', 'mps').
        force: If True, regenerate all embeddings.
        progress: Callback(current, total) for progress reporting.

    Returns:
        (generated_count, skipped_count)
    """
    import numpy as np

    embed_dir = home / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    meta_path = embed_dir / "meta.json"
    npy_path = embed_dir / "embeddings.npy"

    backend, model_name = _parse_model_spec(model)

    # Load existing metadata
    existing_meta: dict[str, Any] = {}
    existing_ids: set[str] = set()
    existing_embeddings = None

    if meta_path.exists() and not force:
        with open(meta_path) as f:
            existing_meta = json.load(f)
        # If model changed, force full rebuild
        if existing_meta.get("model") != model:
            force = True
        else:
            existing_ids = set(existing_meta.get("conv_ids", []))
            if npy_path.exists():
                existing_embeddings = np.load(str(npy_path))

    if force:
        existing_ids = set()
        existing_embeddings = None

    # Find conversations that need embedding
    all_convs = conn.execute("SELECT id FROM conversations").fetchall()
    all_ids = [r["id"] for r in all_convs]
    new_ids = [cid for cid in all_ids if cid not in existing_ids]

    skipped = len(all_ids) - len(new_ids)

    if not new_ids:
        if progress:
            progress(0, 0)
        return 0, skipped

    total = len(new_ids)
    if progress:
        progress(0, total)

    # Build texts for new conversations
    texts = []
    valid_ids = []
    for i, conv_id in enumerate(new_ids):
        text = _build_conversation_text(conn, conv_id)
        if text.strip():
            texts.append(text)
            valid_ids.append(conv_id)
        if progress:
            progress(i + 1, total)

    if not texts:
        return 0, skipped

    # Encode
    if backend == "ollama":
        new_embeddings = _encode_ollama(texts, model_name, batch_size)
    else:
        new_embeddings = _encode_sentence_transformers(
            texts, model_name, batch_size, device
        )

    # Ensure numpy array
    new_embeddings = np.array(new_embeddings, dtype=np.float32)

    # Merge with existing
    if existing_embeddings is not None and len(existing_embeddings) > 0:
        final_embeddings = np.vstack([existing_embeddings, new_embeddings])
        final_ids = list(existing_meta["conv_ids"]) + valid_ids
    else:
        final_embeddings = new_embeddings
        final_ids = valid_ids

    # Save
    np.save(str(npy_path), final_embeddings)
    meta = {
        "model": model,
        "dimensions": int(final_embeddings.shape[1]),
        "count": len(final_ids),
        "conv_ids": final_ids,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return len(valid_ids), skipped


# ── Analytics cache ──────────────────────────────────────────────────────


def export_analytics(
    conn: sqlite3.Connection,
    home: Path,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, int]:
    """Export conversations and messages to parquet files.

    Returns:
        (conversation_count, message_count)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    analytics_dir = home / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0, 2)

    # Export conversations
    conv_rows = conn.execute("SELECT * FROM conversations").fetchall()
    if conv_rows:
        columns = conv_rows[0].keys()
        data = {col: [r[col] for r in conv_rows] for col in columns}
        table = pa.table(data)
        pq.write_table(table, str(analytics_dir / "conversations.parquet"))
    conv_count = len(conv_rows)

    if progress:
        progress(1, 2)

    # Export messages
    msg_rows = conn.execute("SELECT * FROM messages").fetchall()
    if msg_rows:
        columns = msg_rows[0].keys()
        data = {col: [r[col] for r in msg_rows] for col in columns}
        table = pa.table(data)
        pq.write_table(table, str(analytics_dir / "messages.parquet"))
    msg_count = len(msg_rows)

    if progress:
        progress(2, 2)

    return conv_count, msg_count


# ── Orchestrator ─────────────────────────────────────────────────────────


def run_index(
    conn: sqlite3.Connection,
    home: Path,
    *,
    do_languages: bool = True,
    do_embeddings: bool = True,
    do_topics: bool = True,
    do_analytics: bool = True,
    embed_model: str = "st:paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 64,
    device: str = "auto",
    force: bool = False,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> IndexStats:
    """Run the full indexing pipeline.

    Args:
        conn: SQLite connection.
        home: Ricoeur home directory.
        do_languages: Run language detection.
        do_embeddings: Run embedding generation.
        do_topics: Run topic modeling (stub).
        do_analytics: Run analytics export.
        embed_model: Model spec for embeddings.
        batch_size: Batch size for embedding.
        device: Device for sentence-transformers.
        force: Redo everything even if cached.
        progress_cb: Callback(layer_name, current, total).

    Returns:
        IndexStats with results from all layers.
    """
    stats = IndexStats()

    def _progress(layer: str):
        def cb(current: int, total: int):
            if progress_cb:
                progress_cb(layer, current, total)

        return cb

    # ── Language detection ────────────────────────────────────────────
    if do_languages:
        if check_dependency("langdetect"):
            try:
                count = detect_languages(
                    conn, force=force, progress=_progress("languages")
                )
                stats.languages_detected = count
            except Exception as e:
                stats.errors.append(f"Language detection failed: {e}")
        else:
            hint = _INSTALL_HINTS["langdetect"]
            stats.errors.append(
                f"Language detection skipped: fast-langdetect not installed. "
                f"Install with: {hint}"
            )
            stats.languages_skipped = 1

    # ── Embeddings ───────────────────────────────────────────────────
    if do_embeddings:
        backend, _ = _parse_model_spec(embed_model)
        dep_name = "ollama" if backend == "ollama" else "embeddings"

        if check_dependency(dep_name):
            try:
                generated, skipped = generate_embeddings(
                    conn,
                    home,
                    model=embed_model,
                    batch_size=batch_size,
                    device=device,
                    force=force,
                    progress=_progress("embeddings"),
                )
                stats.embeddings_generated = generated
                stats.embeddings_skipped = skipped
                stats.embedding_model = embed_model

                # Read back dimensions from meta
                meta_path = home / "embeddings" / "meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    stats.embedding_dimensions = meta.get("dimensions", 0)
            except Exception as e:
                stats.errors.append(f"Embedding generation failed: {e}")
        else:
            hint = _INSTALL_HINTS[dep_name]
            stats.errors.append(
                f"Embeddings skipped: dependencies not installed. "
                f"Install with: {hint}"
            )
            stats.embeddings_skipped = 1

    # ── Topics (stub) ────────────────────────────────────────────────
    if do_topics:
        stats.errors.append("Topic modeling: not yet implemented (coming in issue #4)")

    # ── Analytics cache ──────────────────────────────────────────────
    if do_analytics:
        if check_dependency("analytics"):
            try:
                conv_count, msg_count = export_analytics(
                    conn, home, progress=_progress("analytics")
                )
                stats.analytics_conversations = conv_count
                stats.analytics_messages = msg_count
            except Exception as e:
                stats.errors.append(f"Analytics export failed: {e}")
        else:
            hint = _INSTALL_HINTS["analytics"]
            stats.errors.append(
                f"Analytics skipped: pyarrow not installed. "
                f"Install with: {hint}"
            )

    return stats
