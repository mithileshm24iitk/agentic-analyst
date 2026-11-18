"""Shared test fakes: settings, a fake BigQuery client, a fake ToolContext and a fake embedder."""

import hashlib
from types import SimpleNamespace

import pytest

from app.config import Settings
from tools import bigquery_tools
from tools.retrieval.bm25 import tokenize
from tools.retrieval.ingest import IngestReport, ingest

DATASET = "bigquery-public-data.thelook_ecommerce"


@pytest.fixture(autouse=True)
def app_state_defaults():
    """Values the startup hook normally sets. Most API tests skip startup (it needs GCP).

    No API key = development mode (open API); test_auth.py sets a key itself.
    Tests may assign app.state values directly: the whole state is restored afterwards.
    (Not monkeypatch: it cannot patch the same app.state key twice in one test, because
    Starlette keeps state in a private dict rather than the instance __dict__.)
    """
    from app.main import app

    saved = dict(app.state._state)
    app.state.api_key = None
    app.state.max_question_chars = 2000
    yield
    app.state._state.clear()
    app.state._state.update(saved)


@pytest.fixture
def settings(monkeypatch) -> Settings:
    """Test settings, independent of the real .env file."""
    test_settings = Settings(
        _env_file=None,
        google_cloud_project="test-project",
        google_cloud_location="global",
        worker_model="test-model",
        embedding_model="fake-embedding",
        embedding_location="global",
        bq_dataset=DATASET,
        bq_max_bytes_billed=1_000,
        bq_max_rows=3,
        sql_max_attempts=2,
    )
    monkeypatch.setattr(bigquery_tools, "get_settings", lambda: test_settings)
    bigquery_tools._schema_cache.clear()
    return test_settings


class FakeJob:
    def __init__(self, rows: list[dict], total_bytes_processed: int, error: Exception | None):
        self._rows = rows
        self.total_bytes_processed = total_bytes_processed
        self._error = error

    def result(self, timeout=None):
        if self._error:
            raise self._error
        return iter(self._rows)


class FakeBigQueryClient:
    """Records every query and its job config. Rows are plain dicts."""

    def __init__(self, rows=None, dry_run_bytes: int = 100, error: Exception | None = None):
        self.rows = rows or []
        self.dry_run_bytes = dry_run_bytes
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def query(self, sql, job_config=None):
        self.calls.append((sql, job_config))
        if job_config is not None and job_config.dry_run:
            return FakeJob([], self.dry_run_bytes, None)
        return FakeJob(self.rows, self.dry_run_bytes, self.error)


@pytest.fixture
def fake_client(monkeypatch, settings) -> FakeBigQueryClient:
    client = FakeBigQueryClient()
    monkeypatch.setattr(bigquery_tools, "get_client", lambda: client)
    return client


def make_tool_context(invocation_id: str = "inv-1", state: dict | None = None):
    """Stands in for ADK's ToolContext: run_query only uses .state and .invocation_id."""
    return SimpleNamespace(state={} if state is None else state, invocation_id=invocation_id)


# ---------------------------------------------------------------- RAG fakes


class FakeEmbedder:
    """Deterministic bag-of-words embedder: each word adds 1 to a hashed dimension.

    Texts sharing words get similar vectors, which is enough to test retrieval
    plumbing without calling Vertex AI. Uses hashlib (Python's hash() is random per run).
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls = 0  # number of embed_documents calls

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0  # cosine distance needs a non-zero vector
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def rag_settings(tmp_path) -> Settings:
    """Settings pointing at a temporary docs dir and Chroma dir."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return Settings(
        _env_file=None,
        google_cloud_project="test-project",
        google_cloud_location="global",
        worker_model="test-model",
        embedding_model="fake-embedding",
        embedding_location="global",
        embedding_dim=256,
        docs_dir=docs_dir,
        chroma_dir=tmp_path / "chroma",
        rag_collection="test_docs",
        chunk_size=300,
        chunk_overlap=50,
        rag_top_k=3,
    )


def run_ingest(settings: Settings, embedder) -> IngestReport:
    return ingest(
        docs_dir=settings.docs_dir,
        chroma_dir=settings.chroma_dir,
        collection_name=settings.rag_collection,
        embedder=embedder,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
