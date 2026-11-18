"""BM25 keyword search over the chunks written to chunks.json at ingest time."""

import re

from rank_bm25 import BM25Okapi

from tools.retrieval.chunking import Chunk


def tokenize(text: str) -> list[str]:
    """Lowercase and split on anything that is not a letter or digit."""
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [tokenize(c.text) for c in chunks]
        self._token_sets = [set(t) for t in tokenized]
        # BM25Okapi divides by the number of documents, so it fails on an empty corpus.
        self._bm25 = BM25Okapi(tokenized) if chunks else None

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Return up to n (chunk_id, score) pairs, best first. Higher = better.

        Only chunks sharing at least one word with the query are returned,
        so a chunk with no keyword overlap never gets a BM25 rank.
        """
        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        wanted = set(query_tokens)
        matches = [
            (chunk_id, float(scores[i]))
            for i, chunk_id in enumerate(self.chunk_ids)
            if wanted & self._token_sets[i]
        ]
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches[:n]
