"""Read and write chunks.json, the chunk list that BM25 is built from."""

import json
import os
from pathlib import Path

from tools.retrieval.chunking import Chunk

CHUNKS_FILE = "chunks.json"


def chunks_path(chroma_dir: Path) -> Path:
    return chroma_dir / CHUNKS_FILE


def write_chunks(chroma_dir: Path, chunks: list[Chunk]) -> None:
    """Write atomically (temp file, then rename) so readers never see half a file."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_path(chroma_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([c.to_dict() for c in chunks], indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_chunks(chroma_dir: Path) -> list[Chunk]:
    data = json.loads(chunks_path(chroma_dir).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]
