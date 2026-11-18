"""Ingest data/docs into ChromaDB + chunks.json. Safe to re-run: `make ingest`.

Re-running never duplicates chunks:
  - chunk IDs are stable ("<file>:c<n>"), and we upsert by ID;
  - IDs that exist in the store but not in this run (deleted docs, shorter docs) are removed.
"""

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from tools.retrieval.chunking import Chunk, chunk_markdown
from tools.retrieval.embeddings import Embedder, get_embedder
from tools.retrieval.index_files import write_chunks
from tools.retrieval.vector_store import open_collection_for_ingest


@dataclass
class IngestReport:
    files: int
    chunks: int
    removed: int  # stale chunk IDs deleted from the store


def load_and_chunk(docs_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8"), chunk_size, overlap))
    return chunks


def ingest(
    docs_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    embedder: Embedder,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int,
    overlap: int,
) -> IngestReport:
    chunks = load_and_chunk(docs_dir, chunk_size, overlap)
    collection = open_collection_for_ingest(chroma_dir, collection_name, embedding_model, embedding_dim)

    new_ids = [c.chunk_id for c in chunks]
    stale = sorted(set(collection.get(include=[])["ids"]) - set(new_ids))
    if stale:
        collection.delete(ids=stale)

    if chunks:
        embeddings = embedder.embed_documents([c.text for c in chunks])
        collection.upsert(
            ids=new_ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "heading": c.heading} for c in chunks],
        )

    # Written last, so BM25 only ever sees chunks that are also in the vector store.
    write_chunks(chroma_dir, chunks)
    files = len({c.source for c in chunks})
    return IngestReport(files=files, chunks=len(chunks), removed=len(stale))


def main() -> None:
    s = get_settings()
    report = ingest(
        docs_dir=s.docs_dir,
        chroma_dir=s.chroma_dir,
        collection_name=s.rag_collection,
        embedder=get_embedder(),
        embedding_model=s.embedding_model,
        embedding_dim=s.embedding_dim,
        chunk_size=s.chunk_size,
        overlap=s.chunk_overlap,
    )
    print(
        f"Ingested {report.files} files into {report.chunks} chunks "
        f"({report.removed} stale chunks removed) -> {s.chroma_dir}"
    )


if __name__ == "__main__":
    main()
