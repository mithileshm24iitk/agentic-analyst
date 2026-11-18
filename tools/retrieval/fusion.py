"""Reciprocal Rank Fusion (RRF): merge several rankings using only the ranks, not the scores.

Vector distances and BM25 scores are on different scales, so they cannot be added.
RRF gives each item 1 / (k + rank) from every list it appears in and sums them.
Items ranked well by both lists beat items ranked first by only one.
"""


def rrf_scores(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(item) = sum over lists of 1 / (k + rank). Ranks start at 1.

    An item missing from a list gets nothing from that list.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Return (item, rrf score) pairs, best first. Ties keep first-seen order."""
    scores = rrf_scores(ranked_lists, k)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
