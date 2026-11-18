"""Real RAG retrieval test. Skipped unless RUN_RAG_INTEGRATION=1.

Needs `make ingest` to have run, gcloud ADC, and .env. Cost: one query embedding call.
"""

import os

import pytest

from tools.retrieval.search import hybrid_search

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RAG_INTEGRATION") != "1",
    reason="set RUN_RAG_INTEGRATION=1 to run against real Vertex AI embeddings",
)


def test_real_hybrid_search_finds_jeans_threshold():
    hits = hybrid_search("What is the maximum acceptable return rate for Jeans?")

    assert hits, "no results: did you run make ingest?"
    assert any(
        "Jeans: maximum acceptable return rate is 11.0%." in h["text"] for h in hits
    ), [h["chunk_id"] for h in hits]
    top = hits[0]
    assert top["vector_rank"] is not None or top["bm25_rank"] is not None
