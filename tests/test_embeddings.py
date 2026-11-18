"""GeminiEmbedder: batching, retries with backoff, and the vector-count check. No GCP."""

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from tools.retrieval import embeddings
from tools.retrieval.embeddings import GeminiEmbedder


def api_error(code: int) -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": "x", "status": "X"}})


class FakeModels:
    """Records embed_content calls; fails with the queued errors first."""

    def __init__(self, errors=(), vectors_per_call=None):
        self.errors = list(errors)
        self.vectors_per_call = vectors_per_call  # None = one vector per input text
        self.calls: list[tuple[list[str], str, int]] = []

    def embed_content(self, model, contents, config):
        self.calls.append((list(contents), config.task_type, config.output_dimensionality))
        if self.errors:
            raise self.errors.pop(0)
        n = len(contents) if self.vectors_per_call is None else self.vectors_per_call
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(i), 1.0]) for i in range(n)])


@pytest.fixture
def make_embedder(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(embeddings, "_sleep", sleeps.append)  # no real waiting

    def make(models: FakeModels, batch_size: int = 2, timeout_seconds: int | None = 30):
        created = {}

        def fake_client(**kwargs):
            created.update(kwargs)
            return SimpleNamespace(models=models)

        monkeypatch.setattr(embeddings.genai, "Client", fake_client)
        embedder = GeminiEmbedder("emb-model", "proj", "global", dim=768, batch_size=batch_size,
                                  timeout_seconds=timeout_seconds)
        return embedder, created, sleeps

    return make


def test_documents_are_embedded_in_batches_with_document_task_type(make_embedder):
    models = FakeModels()
    embedder, _, _ = make_embedder(models, batch_size=2)

    vectors = embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert [c[0] for c in models.calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert {c[1] for c in models.calls} == {"RETRIEVAL_DOCUMENT"}
    assert {c[2] for c in models.calls} == {768}


def test_query_uses_query_task_type(make_embedder):
    models = FakeModels()
    embedder, _, _ = make_embedder(models)
    embedder.embed_query("jeans")
    assert models.calls == [(["jeans"], "RETRIEVAL_QUERY", 768)]


def test_retries_429_and_503_with_exponential_backoff(make_embedder):
    models = FakeModels(errors=[api_error(429), api_error(503)])
    embedder, _, sleeps = make_embedder(models)

    assert len(embedder.embed_query("q")) == 2

    assert len(models.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_tries(make_embedder):
    models = FakeModels(errors=[api_error(429)] * embeddings.MAX_TRIES)
    embedder, _, sleeps = make_embedder(models)

    with pytest.raises(genai_errors.ClientError):
        embedder.embed_query("q")
    assert len(models.calls) == embeddings.MAX_TRIES
    assert len(sleeps) == embeddings.MAX_TRIES - 1


def test_non_retryable_error_raises_immediately(make_embedder):
    models = FakeModels(errors=[api_error(403)])
    embedder, _, sleeps = make_embedder(models)

    with pytest.raises(genai_errors.ClientError):
        embedder.embed_query("q")
    assert len(models.calls) == 1 and sleeps == []


def test_merged_batch_is_rejected(make_embedder):
    # e.g. gemini-embedding-2 returned one vector for three texts
    embedder, _, _ = make_embedder(FakeModels(vectors_per_call=1), batch_size=3)
    with pytest.raises(RuntimeError, match="returned 1 vectors for 3 texts"):
        embedder.embed_documents(["a", "b", "c"])


def test_client_gets_a_timeout_in_milliseconds(make_embedder):
    _, created, _ = make_embedder(FakeModels(), timeout_seconds=30)
    assert created["http_options"].timeout == 30_000
    assert created["vertexai"] is True and created["location"] == "global"
