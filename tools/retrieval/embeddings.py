"""Text embeddings. GeminiEmbedder calls Vertex AI; tests use a fake with the same methods."""

import logging
import time
from functools import lru_cache
from typing import Protocol

from google import genai
from google.genai import errors, types

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_TRIES = 5
BASE_DELAY_S = 1.0
RETRYABLE_STATUS = {429, 503}
_sleep = time.sleep  # tests replace this so they do not wait


class Embedder(Protocol):
    """Anything with these two methods can be used for retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Gemini embeddings on Vertex AI, batched, with retries on 429/503."""

    def __init__(self, model: str, project: str, location: str, dim: int, batch_size: int) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = genai.Client(vertexai=True, project=project, location=location)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        for attempt in range(1, MAX_TRIES + 1):
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                break
            except errors.APIError as e:
                if e.code not in RETRYABLE_STATUS or attempt == MAX_TRIES:
                    raise
                delay = BASE_DELAY_S * 2 ** (attempt - 1)
                logger.warning("Embedding HTTP %s, retry %d/%d in %.0fs", e.code, attempt, MAX_TRIES, delay)
                _sleep(delay)
        vectors = [list(e.values) for e in response.embeddings]
        if len(vectors) != len(texts):
            # Some models merge a batch into one vector; fail loudly instead of mis-indexing.
            raise RuntimeError(f"{self.model} returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors


@lru_cache
def get_embedder() -> GeminiEmbedder:
    """The configured embedder, created once."""
    s = get_settings()
    return GeminiEmbedder(
        model=s.embedding_model,
        project=s.google_cloud_project,
        location=s.embedding_location,
        dim=s.embedding_dim,
        batch_size=s.embedding_batch_size,
    )
