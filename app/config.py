"""App settings, loaded from environment variables and the .env file."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config lives here. Values come from env vars or .env (env vars win)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_cloud_project: str
    google_cloud_location: str
    google_genai_use_enterprise: bool = True
    worker_model: str

    # BigQuery: queries are billed to google_cloud_project and read this dataset.
    bq_dataset: str = "bigquery-public-data.thelook_ecommerce"
    bq_location: str = "US"  # the public dataset lives in the US multi-region
    bq_max_bytes_billed: int = 1_000_000_000  # 1 GB
    bq_max_rows: int = 100
    bq_query_timeout_seconds: int = 30
    sql_max_attempts: int = 3

    # RAG: embeddings (Vertex AI) and hybrid retrieval over data/docs.
    embedding_model: str
    embedding_location: str
    embedding_dim: int = 768
    embedding_batch_size: int = 50  # Vertex allows up to 250 texts per request
    docs_dir: Path = Path("data/docs")
    chroma_dir: Path = Path("data/chroma")
    rag_collection: str = "shopco_docs"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 100  # characters
    rag_top_k: int = 5
    rrf_k: int = 60  # reciprocal rank fusion constant

    # Orchestrator and critic. Empty = use worker_model.
    orchestrator_model: str | None = None
    critic_model: str | None = None
    critic_max_revisions: int = 1
    reports_dir: Path = Path("reports")

    # Timeouts. One model or embedding HTTP call, and one whole API request
    # (kept under Cloud Run's default 300s so we answer with clean JSON first).
    llm_timeout_seconds: int = 90
    request_timeout_seconds: int = 270

    @model_validator(mode="after")
    def _default_models_to_worker(self) -> "Settings":
        self.orchestrator_model = self.orchestrator_model or self.worker_model
        self.critic_model = self.critic_model or self.worker_model
        return self

    def export_to_environ(self) -> None:
        """Copy the Google settings into os.environ.

        The Google GenAI client used inside ADK reads these from the process
        environment, not from our Settings object.
        """
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = self.google_cloud_location
        os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = str(self.google_genai_use_enterprise).lower()


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them."""
    return Settings()
