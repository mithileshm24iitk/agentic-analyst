"""Startup check: refuse to serve if the pre-built retrieval index is missing or broken.

The container never runs ingest. It ships data/chroma (chroma.sqlite3 + chunks.json) built
locally by `make ingest`. If that is missing, RAG would quietly answer "the documents do
not contain this", so we fail at startup instead.
"""

import json
from pathlib import Path

from tools.retrieval.index_files import chunks_path, read_chunks
from tools.retrieval.vector_store import get_client

HINT = "Run `make ingest` locally, then rebuild the image (data/chroma must be copied into it)."


class IndexMissing(RuntimeError):
    """The retrieval index is missing, empty or does not match the settings."""


def check_index(chroma_dir: Path, collection_name: str, embedding_model: str, embedding_dim: int) -> int:
    """Return the number of chunks, or raise IndexMissing with a clear message."""
    path = chunks_path(chroma_dir)
    if not path.is_file():
        raise IndexMissing(f"Retrieval index missing: {path} not found. {HINT}")
    try:
        chunks = read_chunks(chroma_dir)
    except (ValueError, TypeError) as e:  # bad JSON or wrong fields
        raise IndexMissing(f"Retrieval index unreadable: {path} ({e}). {HINT}") from e
    if not chunks:
        raise IndexMissing(f"Retrieval index is empty: {path} has no chunks. {HINT}")

    if not (chroma_dir / "chroma.sqlite3").is_file():
        raise IndexMissing(f"Vector store missing: {chroma_dir / 'chroma.sqlite3'} not found. {HINT}")
    client = get_client(chroma_dir)
    if collection_name not in [c.name for c in client.list_collections()]:
        raise IndexMissing(f"Vector store has no collection {collection_name!r} (RAG_COLLECTION). {HINT}")

    collection = client.get_collection(collection_name, embedding_function=None)
    meta = collection.metadata or {}
    if meta.get("embedding_model") != embedding_model or meta.get("embedding_dim") != embedding_dim:
        built_with = json.dumps({k: meta.get(k) for k in ("embedding_model", "embedding_dim")})
        raise IndexMissing(
            f"Vector store was built with {built_with}, but settings say "
            f"EMBEDDING_MODEL={embedding_model!r}, EMBEDDING_DIM={embedding_dim}. {HINT}"
        )
    count = collection.count()
    if count != len(chunks):
        raise IndexMissing(f"Vector store has {count} vectors but chunks.json has {len(chunks)} chunks. {HINT}")
    return count
