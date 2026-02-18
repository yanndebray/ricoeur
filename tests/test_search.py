"""Tests for semantic, hybrid, and dispatch search functionality."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ricoeur.db import SCHEMA_SQL
from ricoeur.search import (
    SearchResult,
    _load_embeddings,
    _encode_query,
    fts_search,
    semantic_search,
    hybrid_search,
    search_dispatch,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_home(tmp_path):
    """Create a temp ricoeur home directory."""
    (tmp_path / "embeddings").mkdir()
    return tmp_path


@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database with schema and sample data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)

    # Insert sample conversations
    convs = [
        ("conv-1", "Python data analysis", "chatgpt", "gpt-4o", "2025-01-15T10:00:00", "en", None),
        ("conv-2", "Deploiement Kubernetes", "chatgpt", "gpt-4o", "2025-02-20T14:00:00", "fr", None),
        ("conv-3", "React frontend setup", "claude", "claude-3-opus", "2025-03-10T09:00:00", "en", None),
        ("conv-4", "Docker deployment strategies", "chatgpt", "gpt-4", "2025-04-01T12:00:00", "en", 1),
        ("conv-5", "Machine learning pipelines", "chatgpt", "gpt-4o", "2025-05-05T08:00:00", "en", None),
    ]
    for c in convs:
        conn.execute(
            "INSERT INTO conversations (id, title, platform, model, created_at, language, topic_id) VALUES (?,?,?,?,?,?,?)",
            c,
        )

    # Insert messages (needed for FTS)
    msgs = [
        ("msg-1", "conv-1", "user", "How do I analyze data with pandas in python?", "2025-01-15T10:00:01"),
        ("msg-2", "conv-1", "assistant", "You can use pandas DataFrame to analyze data.", "2025-01-15T10:00:02"),
        ("msg-3", "conv-2", "user", "Comment deployer sur Kubernetes?", "2025-02-20T14:00:01"),
        ("msg-4", "conv-2", "assistant", "Utilisez kubectl apply pour deployer.", "2025-02-20T14:00:02"),
        ("msg-5", "conv-3", "user", "How to set up a React project with TypeScript?", "2025-03-10T09:00:01"),
        ("msg-6", "conv-3", "assistant", "Use create-react-app with the TypeScript template.", "2025-03-10T09:00:02"),
        ("msg-7", "conv-4", "user", "What are the best deployment strategies for Docker?", "2025-04-01T12:00:01"),
        ("msg-8", "conv-4", "assistant", "Blue-green, canary, and rolling deployment strategies.", "2025-04-01T12:00:02"),
        ("msg-9", "conv-5", "user", "How to build ML pipelines with scikit-learn?", "2025-05-05T08:00:01"),
        ("msg-10", "conv-5", "assistant", "Use sklearn Pipeline to chain transformers.", "2025-05-05T08:00:02"),
    ]
    for m in msgs:
        conn.execute(
            "INSERT INTO messages (id, conv_id, role, content, timestamp) VALUES (?,?,?,?,?)",
            m,
        )
    conn.commit()
    return conn


@pytest.fixture
def fake_embeddings(tmp_home):
    """Create fake normalized embeddings (5 conversations, 8 dims)."""
    rng = np.random.RandomState(42)
    emb = rng.randn(5, 8).astype(np.float32)
    # L2-normalize each row
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms

    conv_ids = ["conv-1", "conv-2", "conv-3", "conv-4", "conv-5"]
    meta = {
        "model": "st:test-model",
        "dimensions": 8,
        "count": 5,
        "conv_ids": conv_ids,
    }

    np.save(str(tmp_home / "embeddings" / "embeddings.npy"), emb)
    with open(tmp_home / "embeddings" / "meta.json", "w") as f:
        json.dump(meta, f)

    return emb, meta


# ── _load_embeddings ─────────────────────────────────────────────────────


class TestLoadEmbeddings:
    def test_load_success(self, tmp_home, fake_embeddings):
        emb, meta = _load_embeddings(tmp_home)
        assert emb is not None
        assert meta is not None
        assert emb.shape == (5, 8)
        assert len(meta["conv_ids"]) == 5

    def test_missing_npy(self, tmp_home):
        # meta exists but npy doesn't
        meta = {"model": "st:test", "conv_ids": ["a"], "dimensions": 8, "count": 1}
        with open(tmp_home / "embeddings" / "meta.json", "w") as f:
            json.dump(meta, f)
        emb, meta_out = _load_embeddings(tmp_home)
        assert emb is None
        assert meta_out is None

    def test_missing_meta(self, tmp_home):
        # npy exists but meta doesn't
        np.save(str(tmp_home / "embeddings" / "embeddings.npy"), np.zeros((2, 4)))
        emb, meta = _load_embeddings(tmp_home)
        assert emb is None
        assert meta is None

    def test_mismatch_count(self, tmp_home):
        # meta says 3 conv_ids but npy has 2 rows
        np.save(str(tmp_home / "embeddings" / "embeddings.npy"), np.zeros((2, 4)))
        meta = {"model": "st:test", "conv_ids": ["a", "b", "c"], "dimensions": 4, "count": 3}
        with open(tmp_home / "embeddings" / "meta.json", "w") as f:
            json.dump(meta, f)
        emb, meta_out = _load_embeddings(tmp_home)
        assert emb is None
        assert meta_out is None

    def test_no_embeddings_dir(self, tmp_path):
        emb, meta = _load_embeddings(tmp_path)
        assert emb is None
        assert meta is None


# ── _encode_query ────────────────────────────────────────────────────────


class TestEncodeQuery:
    def test_sentence_transformers_backend(self):
        """Test that the ST backend loads and encodes a query."""
        from ricoeur.search import _model_cache
        _model_cache.clear()

        fake_model = MagicMock()
        fake_vec = np.random.randn(8).astype(np.float32)
        fake_vec = fake_vec / np.linalg.norm(fake_vec)
        fake_model.encode.return_value = fake_vec

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_cls:
            result = _encode_query("test query", "st:test-model", "cpu")

        mock_cls.assert_called_once_with("test-model", device="cpu")
        fake_model.encode.assert_called_once_with("test query", normalize_embeddings=True)
        assert result.shape == (8,)
        # Should be L2-normalized
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5
        _model_cache.clear()

    def test_ollama_backend(self):
        """Test that the ollama backend sends the right prefix."""
        import sys
        # Ensure ollama module exists for patching
        mock_ollama_mod = MagicMock()
        fake_vec = np.random.randn(8).astype(np.float32).tolist()
        mock_ollama_mod.embed.return_value = {"embeddings": [fake_vec]}

        with patch.dict(sys.modules, {"ollama": mock_ollama_mod}):
            result = _encode_query("my query", "ollama:nomic-embed-text", "auto")

        mock_ollama_mod.embed.assert_called_once_with(
            model="nomic-embed-text",
            input=["search_query: my query"],
        )
        assert result.shape == (8,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_auto_device_passes_none(self):
        """When device='auto', SentenceTransformer should get device=None."""
        from ricoeur.search import _model_cache
        _model_cache.clear()

        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones(4, dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_cls:
            _encode_query("q", "st:model", "auto")

        mock_cls.assert_called_once_with("model", device=None)
        _model_cache.clear()

    def test_model_cache(self):
        """Second call with same model_spec should reuse cached model."""
        from ricoeur.search import _model_cache
        _model_cache.clear()

        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones(4, dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_cls:
            _encode_query("q1", "st:cached-model", "cpu")
            _encode_query("q2", "st:cached-model", "cpu")

        # Constructor called only once
        assert mock_cls.call_count == 1
        # But encode called twice
        assert fake_model.encode.call_count == 2

        _model_cache.clear()


# ── semantic_search ──────────────────────────────────────────────────────


class TestSemanticSearch:
    def test_basic_semantic_search(self, db_conn, tmp_home, fake_embeddings):
        emb, meta = fake_embeddings
        # Make conv-4's embedding very similar to query_vec
        query_vec = emb[3].copy()  # This is conv-4's embedding

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = semantic_search(
                db_conn, "deployment strategies", tmp_home,
                model_spec="st:test-model", limit=3,
            )

        assert len(results) > 0
        assert len(results) <= 3
        # conv-4 should be the top result (identical vector)
        assert results[0].conv_id == "conv-4"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[0].snippet is None  # semantic results have no snippet

    def test_no_embeddings_raises(self, db_conn, tmp_path):
        with pytest.raises(RuntimeError, match="No embeddings found"):
            semantic_search(
                db_conn, "query", tmp_path,
                model_spec="st:test-model",
            )

    def test_platform_filter(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = semantic_search(
                db_conn, "test", tmp_home,
                model_spec="st:test-model", platform="claude",
            )

        for r in results:
            assert r.platform == "claude"

    def test_lang_filter(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = semantic_search(
                db_conn, "test", tmp_home,
                model_spec="st:test-model", lang="fr",
            )

        for r in results:
            assert r.language == "fr"

    def test_since_until_filter(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = semantic_search(
                db_conn, "test", tmp_home,
                model_spec="st:test-model",
                since="2025-03-01",
                until="2025-04-30",
            )

        for r in results:
            assert r.created_at >= "2025-03-01"
            assert r.created_at <= "2025-04-30"

    def test_topic_filter(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = semantic_search(
                db_conn, "test", tmp_home,
                model_spec="st:test-model", topic=1,
            )

        for r in results:
            assert r.topic_id == 1


# ── hybrid_search ────────────────────────────────────────────────────────


class TestHybridSearch:
    def test_rrf_fusion(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[3].copy()  # conv-4's vector

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = hybrid_search(
                db_conn, "deployment", tmp_home,
                model_spec="st:test-model", limit=5,
            )

        assert len(results) > 0
        # RRF scores should be positive small floats
        for r in results:
            assert r.score > 0
        # Results should be sorted descending by RRF score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_graceful_fallback_no_embeddings(self, db_conn, tmp_path):
        """When embeddings are missing, hybrid falls back to FTS-only."""
        results = hybrid_search(
            db_conn, "python", tmp_path,
            model_spec="st:test-model", limit=5,
        )
        # Should still return FTS results (keyword match on "python")
        assert len(results) > 0

    def test_hybrid_preserves_fts_snippet(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = hybrid_search(
                db_conn, "pandas", tmp_home,
                model_spec="st:test-model", limit=5,
            )

        # If a result came from FTS, it should have a snippet
        fts_hits = fts_search(db_conn, "pandas", limit=5)
        fts_conv_ids = {r.conv_id for r in fts_hits}
        for r in results:
            if r.conv_id in fts_conv_ids:
                # FTS-sourced results should carry their snippet through
                assert r.snippet is not None or r.conv_id not in fts_conv_ids

    def test_hybrid_with_filters(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results = hybrid_search(
                db_conn, "python", tmp_home,
                model_spec="st:test-model", lang="en", limit=5,
            )

        for r in results:
            assert r.language == "en"


# ── search_dispatch ──────────────────────────────────────────────────────


class TestSearchDispatch:
    def test_keyword_flag(self, db_conn, tmp_home, fake_embeddings):
        results, mode = search_dispatch(
            db_conn, "python", tmp_home,
            keyword=True, model_spec="st:test-model",
        )
        assert mode == "keyword"
        assert len(results) > 0

    def test_code_flag_forces_keyword(self, db_conn, tmp_home, fake_embeddings):
        results, mode = search_dispatch(
            db_conn, "pandas", tmp_home,
            code=True, model_spec="st:test-model",
        )
        assert mode == "keyword"

    def test_role_flag_forces_keyword(self, db_conn, tmp_home, fake_embeddings):
        results, mode = search_dispatch(
            db_conn, "python", tmp_home,
            role="user", model_spec="st:test-model",
        )
        assert mode == "keyword"

    def test_semantic_flag(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results, mode = search_dispatch(
                db_conn, "test", tmp_home,
                semantic=True, model_spec="st:test-model",
            )

        assert mode == "semantic"

    def test_default_hybrid_with_embeddings(self, db_conn, tmp_home, fake_embeddings):
        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        with patch("ricoeur.search._encode_query", return_value=query_vec):
            results, mode = search_dispatch(
                db_conn, "python", tmp_home,
                model_spec="st:test-model",
            )

        assert mode == "hybrid"

    def test_default_keyword_without_embeddings(self, db_conn, tmp_path):
        results, mode = search_dispatch(
            db_conn, "python", tmp_path,
            model_spec="st:test-model",
        )
        assert mode == "keyword"

    def test_semantic_without_embeddings_raises(self, db_conn, tmp_path):
        with pytest.raises(RuntimeError, match="No embeddings found"):
            search_dispatch(
                db_conn, "test", tmp_path,
                semantic=True, model_spec="st:test-model",
            )


# ── CLI integration ──────────────────────────────────────────────────────


class TestCLISearch:
    def test_mutual_exclusion(self):
        """--semantic and --keyword together should error."""
        from click.testing import CliRunner
        from ricoeur.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--semantic", "--keyword"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_keyword_mode_label(self, db_conn, tmp_path):
        """Output should show (keyword) label."""
        from click.testing import CliRunner
        from ricoeur.cli import cli

        runner = CliRunner()
        with patch("ricoeur.cli.get_connection", return_value=db_conn), \
             patch("ricoeur.cli.get_home", return_value=tmp_path):
            result = runner.invoke(cli, ["search", "python", "--keyword"])

        assert "(keyword)" in result.output

    def test_semantic_mode_label(self, db_conn, tmp_home, fake_embeddings):
        """Output should show (semantic) label."""
        from click.testing import CliRunner
        from ricoeur.cli import cli

        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        runner = CliRunner()
        with patch("ricoeur.cli.get_connection", return_value=db_conn), \
             patch("ricoeur.cli.get_home", return_value=tmp_home), \
             patch("ricoeur.search._encode_query", return_value=query_vec):
            result = runner.invoke(cli, ["search", "test", "--semantic"])

        assert "(semantic)" in result.output

    def test_hybrid_mode_label(self, db_conn, tmp_home, fake_embeddings):
        """Default with embeddings should show (hybrid) label."""
        from click.testing import CliRunner
        from ricoeur.cli import cli

        emb, _ = fake_embeddings
        query_vec = emb[0].copy()

        runner = CliRunner()
        with patch("ricoeur.cli.get_connection", return_value=db_conn), \
             patch("ricoeur.cli.get_home", return_value=tmp_home), \
             patch("ricoeur.search._encode_query", return_value=query_vec):
            result = runner.invoke(cli, ["search", "python"])

        assert "(hybrid)" in result.output

    def test_no_results_shows_mode(self, db_conn, tmp_path):
        """Even 'no results' should show the mode."""
        from click.testing import CliRunner
        from ricoeur.cli import cli

        runner = CliRunner()
        with patch("ricoeur.cli.get_connection", return_value=db_conn), \
             patch("ricoeur.cli.get_home", return_value=tmp_path):
            result = runner.invoke(cli, ["search", "zzzznonexistent", "--keyword"])

        assert "(keyword)" in result.output
        assert "No results" in result.output
