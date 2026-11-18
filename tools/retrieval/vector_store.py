"""ChromaDB persistent vector store. We pass embeddings in ourselves (no Chroma embedder)."""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings


def get_client(chroma_dir: Path) -> chromadb.ClientAPI:
    """One client per folder, reused across searches (cheaper than re-opening each time)."""
    return _client_for(str(chroma_dir.resolve()))


@lru_cache
def _client_for(path: str) -> chromadb.ClientAPI:
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


def open_collection(chroma_dir: Path, name: str, model: str, dim: int) -> Collection:
    """Open (or create) the collection. Cosine distance; model and dim recorded in metadata."""
    return get_client(chroma_dir).get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": model, "embedding_dim": dim},
        embedding_function=None,
    )


def open_collection_for_ingest(chroma_dir: Path, name: str, model: str, dim: int) -> Collection:
    """Like open_collection, but rebuild it if it was built with a different model or dim.

    Vectors from different embedding models are not comparable, so they must never mix.
    """
    client = get_client(chroma_dir)
    if name in [c.name for c in client.list_collections()]:
        meta = client.get_collection(name, embedding_function=None).metadata or {}
        if meta.get("embedding_model") != model or meta.get("embedding_dim") != dim:
            client.delete_collection(name)
    return open_collection(chroma_dir, name, model, dim)


def vector_search(collection: Collection, query_embedding: list[float], n: int) -> list[tuple[str, float]]:
    """Return up to n (chunk_id, cosine distance) pairs, closest first. Lower = closer."""
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, collection.count()),
        include=["distances"],
    )
    return list(zip(result["ids"][0], result["distances"][0]))
