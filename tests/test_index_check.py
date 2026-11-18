"""Startup index check: a real (temporary) Chroma index built with the fake embedder."""

import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from tests.conftest import FakeEmbedder, run_ingest
from tools.retrieval.index_check import IndexMissing, check_index
from tools.retrieval.index_files import chunks_path


@pytest.fixture
def built(rag_settings):
    (rag_settings.docs_dir / "policy.md").write_text(
        "# Policy\n\n## Returns\n\nJeans: 11.0%.\n\n## Shipping\n\nTwo days.\n", encoding="utf-8"
    )
    report = run_ingest(rag_settings, FakeEmbedder(dim=rag_settings.embedding_dim))
    assert report.chunks > 0
    return rag_settings


def check(s, **overrides):
    args = dict(
        chroma_dir=s.chroma_dir,
        collection_name=s.rag_collection,
        embedding_model=s.embedding_model,
        embedding_dim=s.embedding_dim,
    )
    return check_index(**{**args, **overrides})


def test_built_index_passes(built):
    assert check(built) == len(json.loads(chunks_path(built.chroma_dir).read_text()))


def test_missing_folder_fails(rag_settings):
    with pytest.raises(IndexMissing, match="chunks.json not found.*make ingest"):
        check(rag_settings)


def test_missing_chunks_file_fails(built):
    chunks_path(built.chroma_dir).unlink()
    with pytest.raises(IndexMissing, match="not found"):
        check(built)


def test_empty_chunks_file_fails(built):
    chunks_path(built.chroma_dir).write_text("[]")
    with pytest.raises(IndexMissing, match="empty"):
        check(built)


def test_broken_chunks_file_fails(built):
    chunks_path(built.chroma_dir).write_text("{not json")
    with pytest.raises(IndexMissing, match="unreadable"):
        check(built)


def test_missing_vector_store_fails(tmp_path, built):
    other = tmp_path / "only-chunks"
    other.mkdir()
    chunks_path(other).write_text(chunks_path(built.chroma_dir).read_text())
    with pytest.raises(IndexMissing, match="chroma.sqlite3 not found"):
        check(built, chroma_dir=other)


def test_missing_collection_fails(built):
    with pytest.raises(IndexMissing, match="no collection 'other'"):
        check(built, collection_name="other")


def test_count_mismatch_fails(built):
    path = chunks_path(built.chroma_dir)
    path.write_text(json.dumps(json.loads(path.read_text())[:1]))
    with pytest.raises(IndexMissing, match="vectors but chunks.json has 1"):
        check(built)


def test_different_embedding_model_fails(built):
    with pytest.raises(IndexMissing, match="EMBEDDING_MODEL='another-model'"):
        check(built, embedding_model="another-model")


def test_app_refuses_to_start_without_index(monkeypatch, rag_settings):
    monkeypatch.setattr(main, "get_settings", lambda: rag_settings)  # chroma_dir is an empty tmp folder
    with pytest.raises(IndexMissing):
        with TestClient(main.app):
            pass
