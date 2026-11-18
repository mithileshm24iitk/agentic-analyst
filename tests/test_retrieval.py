"""Ingestion and hybrid_search with a fake embedder and a temporary ChromaDB. No GCP."""

import re
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder, run_ingest
from tools.retrieval.index_files import read_chunks
from tools.retrieval.search import IndexNotBuilt, hybrid_search
from tools.retrieval.vector_store import open_collection

REAL_DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def write_doc(settings, name: str, text: str) -> None:
    (settings.docs_dir / name).write_text(text, encoding="utf-8")


def collection_ids(settings) -> list[str]:
    col = open_collection(
        settings.chroma_dir, settings.rag_collection, settings.embedding_model, settings.embedding_dim
    )
    return sorted(col.get(include=[])["ids"])


@pytest.fixture
def small_corpus(rag_settings):
    write_doc(rag_settings, "returns.md", "# Returns\n\nItems can be returned within 30 days of delivery.")
    write_doc(rag_settings, "shipping.md", "# Shipping\n\nExpedited shipping is delivered within 2 business days.")
    write_doc(rag_settings, "loyalty.md", "# Loyalty\n\nGold members earn points and get free standard shipping.")
    return rag_settings


# ---------------------------------------------------------------- ingestion


def test_ingest_stores_chunks_in_chroma_and_json(small_corpus):
    report = run_ingest(small_corpus, FakeEmbedder())

    assert report.files == 3
    assert report.chunks == 3
    assert collection_ids(small_corpus) == ["loyalty.md:c0", "returns.md:c0", "shipping.md:c0"]
    assert sorted(c.chunk_id for c in read_chunks(small_corpus.chroma_dir)) == collection_ids(small_corpus)


def test_ingest_twice_does_not_duplicate(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())
    first = collection_ids(small_corpus)

    report = run_ingest(small_corpus, FakeEmbedder())

    assert collection_ids(small_corpus) == first
    assert report.removed == 0
    assert len(read_chunks(small_corpus.chroma_dir)) == len(first)


def test_reingest_removes_chunks_of_deleted_doc(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())
    (small_corpus.docs_dir / "loyalty.md").unlink()

    report = run_ingest(small_corpus, FakeEmbedder())

    assert report.removed == 1
    assert "loyalty.md:c0" not in collection_ids(small_corpus)
    assert all(c.source != "loyalty.md" for c in read_chunks(small_corpus.chroma_dir))


def test_reingest_updates_changed_text(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())
    write_doc(small_corpus, "returns.md", "# Returns\n\nItems can be returned within 45 days.")

    run_ingest(small_corpus, FakeEmbedder())

    texts = {c.chunk_id: c.text for c in read_chunks(small_corpus.chroma_dir)}
    assert "45 days" in texts["returns.md:c0"]


def test_changing_embedding_model_rebuilds_collection(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())
    changed = small_corpus.model_copy(update={"embedding_model": "another-model"})

    run_ingest(changed, FakeEmbedder())

    col = open_collection(changed.chroma_dir, changed.rag_collection, "another-model", changed.embedding_dim)
    assert col.metadata["embedding_model"] == "another-model"
    assert col.count() == 3


# ---------------------------------------------------------------- hybrid_search


def test_search_before_ingest_raises_clear_error(rag_settings):
    with pytest.raises(IndexNotBuilt, match="make ingest"):
        hybrid_search("anything", embedder=FakeEmbedder(), settings=rag_settings)


def test_hybrid_search_returns_ranked_chunks_with_sources_and_scores(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())

    # BM25 has no stemming, so the query uses the document's own words ("items", "returned").
    hits = hybrid_search(
        "how many days can items be returned", k=2, embedder=FakeEmbedder(), settings=small_corpus
    )

    assert len(hits) == 2
    top = hits[0]
    assert top["source"] == "returns.md"
    assert top["chunk_id"] == "returns.md:c0"
    assert set(top) == {
        "chunk_id", "source", "heading", "text", "score",
        "vector_rank", "vector_distance", "bm25_rank", "bm25_score",
    }
    assert top["vector_rank"] == 1 and top["bm25_rank"] == 1
    assert hits[0]["score"] >= hits[1]["score"]


def test_hybrid_search_uses_default_top_k(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())
    hits = hybrid_search("shipping", embedder=FakeEmbedder(), settings=small_corpus)
    assert len(hits) == small_corpus.rag_top_k


def test_hybrid_search_sees_new_ingest_without_restart(small_corpus):
    run_ingest(small_corpus, FakeEmbedder())
    hybrid_search("returns", embedder=FakeEmbedder(), settings=small_corpus)  # fills the cache
    write_doc(small_corpus, "warranty.md", "# Warranty\n\nWarranty claims need a receipt.")

    run_ingest(small_corpus, FakeEmbedder())
    hits = hybrid_search("warranty receipt", k=1, embedder=FakeEmbedder(), settings=small_corpus)

    assert hits[0]["source"] == "warranty.md"


# ---------------------------------------------------------------- real ShopCo documents


def _categories_from_returns_policy() -> list[str]:
    text = (REAL_DOCS / "returns_policy.md").read_text(encoding="utf-8")
    return re.findall(r"^(.+): maximum acceptable return rate is [\d.]+%\.$", text, re.M)


CATEGORIES = _categories_from_returns_policy()


@pytest.fixture(scope="module")
def shopco_index(tmp_path_factory):
    """Ingest the real data/docs once (fake embedder) with the production chunk settings."""
    from app.config import Settings

    tmp = tmp_path_factory.mktemp("shopco")
    settings = Settings(
        _env_file=None,
        google_cloud_project="test-project",
        google_cloud_location="global",
        worker_model="test-model",
        embedding_model="fake-embedding",
        embedding_location="global",
        embedding_dim=256,
        docs_dir=REAL_DOCS,
        chroma_dir=tmp / "chroma",
        rag_collection="shopco_test",
    )  # chunk_size, overlap, top_k, rrf_k use the real defaults
    run_ingest(settings, FakeEmbedder())
    return settings


def test_all_26_categories_found_in_returns_policy():
    assert len(CATEGORIES) == 26


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_question_retrieves_that_categorys_threshold(shopco_index, category):
    expected_line = re.compile(rf"^{re.escape(category)}: maximum acceptable return rate is [\d.]+%\.$", re.M)
    question = f"What is the maximum acceptable return rate for {category}?"

    hits = hybrid_search(question, embedder=FakeEmbedder(), settings=shopco_index)

    assert any(h["source"] == "returns_policy.md" and expected_line.search(h["text"]) for h in hits), (
        f"no top-{shopco_index.rag_top_k} chunk contains the {category} threshold: "
        f"{[h['chunk_id'] for h in hits]}"
    )


# ---------------------------------------------------------------- vector and BM25 disagree


class TopicEmbedder:
    """Vectors chosen per document so vector search and BM25 rank differently."""

    VECTORS = {
        "Sem": [1.0, 0.0, 0.0],  # same meaning as the query, but no shared words
        "Both": [0.8, 0.6, 0.0],  # close in meaning AND shares words
        "Filler": [0.0, 1.0, 0.0],
        "Kw": [-1.0, 0.0, 0.0],  # keyword-stuffed, opposite meaning
    }

    def _vector(self, text: str) -> list[float]:
        heading = text.split("\n", 1)[0]
        return self.VECTORS.get(heading, [1.0, 0.0, 0.0])  # the query maps to [1, 0, 0]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_hybrid_search_fuses_disagreeing_rankings(rag_settings):
    write_doc(rag_settings, "sem.md", "# Sem\n\nGarments may be sent back within thirty days.")
    write_doc(rag_settings, "both.md", "# Both\n\nReturn window is thirty days.")
    write_doc(rag_settings, "filler.md", "# Filler\n\nShipping takes five days.")
    write_doc(rag_settings, "kw.md", "# Kw\n\nReturn policy window. Return policy window.")
    run_ingest(rag_settings, TopicEmbedder())

    hits = hybrid_search("return policy window", k=4, embedder=TopicEmbedder(), settings=rag_settings)
    by_source = {h["source"]: h for h in hits}

    # The two retrievers disagree about what is best...
    assert by_source["sem.md"]["vector_rank"] == 1 and by_source["sem.md"]["bm25_rank"] is None
    assert by_source["kw.md"]["bm25_rank"] == 1 and by_source["kw.md"]["vector_rank"] == 4
    # ...and RRF puts the chunk that both rank 2nd on top.
    assert by_source["both.md"]["vector_rank"] == 2 and by_source["both.md"]["bm25_rank"] == 2
    assert [h["source"] for h in hits] == ["both.md", "kw.md", "sem.md", "filler.md"]
