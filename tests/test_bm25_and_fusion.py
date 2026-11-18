"""BM25 ranking and reciprocal rank fusion."""

import pytest

from tools.retrieval.bm25 import BM25Index, tokenize
from tools.retrieval.chunking import Chunk
from tools.retrieval.fusion import fuse, rrf_scores


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, source="doc.md", heading="", text=text)


CHUNKS = [
    chunk("returns", "Jeans returns are accepted within 30 days. Jeans return rate threshold is 11%."),
    chunk("shipping", "Standard shipping takes 5 business days."),
    chunk("promo", "Back to School: 25% off Jeans and Pants."),
    chunk("loyalty", "Gold members get free standard shipping."),
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Tops & Tees: 11.0%!") == ["tops", "tees", "11", "0"]


def test_bm25_ranks_more_relevant_chunk_first():
    hits = BM25Index(CHUNKS).search("jeans return rate", n=10)
    assert hits[0][0] == "returns"
    assert [cid for cid, _ in hits] == ["returns", "promo"]  # only chunks sharing a word


def test_bm25_skips_chunks_without_query_words():
    assert BM25Index(CHUNKS).search("warehouse latitude", n=10) == []


def test_bm25_empty_index_and_empty_query():
    assert BM25Index([]).search("jeans", n=5) == []
    assert BM25Index(CHUNKS).search("?!", n=5) == []


def test_bm25_respects_n():
    assert len(BM25Index(CHUNKS).search("shipping jeans", n=1)) == 1


def test_rrf_scores_formula():
    scores = rrf_scores([["a", "b"], ["b"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_when_rankings_disagree_consensus_wins():
    # Vector search loves "v_only", BM25 loves "b_only"; "both" is 2nd in each list.
    vector = ["v_only", "both", "x"]
    bm25 = ["b_only", "both", "y"]

    ranked = [item for item, _ in fuse([vector, bm25], k=60)]

    assert ranked[0] == "both"  # 2/62 beats 1/61
    assert set(ranked[1:3]) == {"v_only", "b_only"}  # tied: each was #1 in one list
    assert set(ranked[3:]) == {"x", "y"}


def test_rrf_item_in_only_one_list_still_counts():
    ranked = fuse([["a"], []], k=60)
    assert ranked == [("a", pytest.approx(1 / 61))]


def test_rrf_constant_controls_how_much_top_rank_matters():
    # With a small k, being #1 in one list beats being #3 in two lists; with a large k it does not.
    lists = [["solo", "p", "both"], ["q", "r", "both"]]
    assert fuse(lists, k=1)[0][0] == "solo" or fuse(lists, k=1)[0][0] == "q"
    assert fuse(lists, k=60)[0][0] == "both"
