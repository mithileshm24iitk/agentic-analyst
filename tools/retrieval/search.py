"""hybrid_search: vector search + BM25, merged with Reciprocal Rank Fusion."""

from dataclasses import dataclass

from app.config import Settings, get_settings
from tools.retrieval.bm25 import BM25Index
from tools.retrieval.chunking import Chunk
from tools.retrieval.embeddings import Embedder, get_embedder
from tools.retrieval.fusion import fuse
from tools.retrieval.index_files import chunks_path, read_chunks
from tools.retrieval.vector_store import open_collection, vector_search

CANDIDATE_MULTIPLIER = 3  # each retriever proposes 3*k candidates before fusion


class IndexNotBuilt(Exception):
    """Raised when there is no chunks.json yet (run `make ingest`)."""


@dataclass
class _LoadedIndex:
    version: int  # chunks.json modification time; changes after every ingest
    chunks: dict[str, Chunk]
    bm25: BM25Index


_cache: dict[str, _LoadedIndex] = {}


def _load_index(settings: Settings) -> _LoadedIndex:
    """Load chunks.json and build BM25 once; reload only after a new ingest."""
    path = chunks_path(settings.chroma_dir)
    if not path.exists():
        raise IndexNotBuilt(f"No index at {path}. Run `make ingest` first.")
    version = path.stat().st_mtime_ns
    key = str(path.resolve())
    cached = _cache.get(key)
    if cached is None or cached.version != version:
        chunks = read_chunks(settings.chroma_dir)
        cached = _LoadedIndex(version, {c.chunk_id: c for c in chunks}, BM25Index(chunks))
        _cache[key] = cached
    return cached


def hybrid_search(
    query: str,
    k: int | None = None,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """Return the top k chunks for the query, best first.

    Each result has: chunk_id, source, heading, text, score (fused RRF score),
    vector_rank, vector_distance, bm25_rank, bm25_score. A rank is None when
    that retriever did not propose the chunk.
    """
    settings = settings or get_settings()
    embedder = embedder or get_embedder()
    k = k or settings.rag_top_k
    n = k * CANDIDATE_MULTIPLIER

    index = _load_index(settings)
    collection = open_collection(
        settings.chroma_dir, settings.rag_collection, settings.embedding_model, settings.embedding_dim
    )
    vector_hits = [
        (cid, dist)
        for cid, dist in vector_search(collection, embedder.embed_query(query), n)
        if cid in index.chunks
    ]
    bm25_hits = index.bm25.search(query, n)

    vector_rank = {cid: (rank, dist) for rank, (cid, dist) in enumerate(vector_hits, start=1)}
    bm25_rank = {cid: (rank, score) for rank, (cid, score) in enumerate(bm25_hits, start=1)}
    fused = fuse([[cid for cid, _ in vector_hits], [cid for cid, _ in bm25_hits]], settings.rrf_k)

    results = []
    for chunk_id, score in fused[:k]:
        chunk = index.chunks[chunk_id]
        v_rank, v_dist = vector_rank.get(chunk_id, (None, None))
        b_rank, b_score = bm25_rank.get(chunk_id, (None, None))
        results.append(
            {
                "chunk_id": chunk_id,
                "source": chunk.source,
                "heading": chunk.heading,
                "text": chunk.text,
                "score": round(score, 6),
                "vector_rank": v_rank,
                "vector_distance": None if v_dist is None else round(v_dist, 6),
                "bm25_rank": b_rank,
                "bm25_score": None if b_score is None else round(b_score, 6),
            }
        )
    return results
