"""App settings, loaded from environment variables and the .env file."""

import os
from functools import lru_cache
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    """USD per 1M tokens. Thinking tokens are billed as output tokens."""

    input_per_1m: float
    output_per_1m: float  # "Text output (response and reasoning)" on the pricing page
    cached_input_per_1m: float | None = None  # None = charge cached tokens at the input rate


# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
# Read on 2026-11-19, "Gemini 3 Standard", Global endpoint, prompts <= 200K tokens.
# gemini-3.8-flash is on introductory pricing "through December 31, 2026";
# from January 1, 2027 the page lists $1.50 input / $0.15 cached input / $7.50 output.
# Only models whose rows could be read unambiguously are listed; others report cost null.
# Not costed: query embeddings (gemini-embedding-001, a few tokens per question) are not
# LLM calls and are not in the per-request cost.
DEFAULT_MODEL_PRICES: dict[str, ModelPrice | None] = {
    "gemini-3.8-flash": ModelPrice(input_per_1m=0.75, output_per_1m=3.75, cached_input_per_1m=0.075),
}


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

    # Gemini thinking effort for every agent, sent as thinking_config.thinking_level.
    # gemini-3.8-flash supports LOW, MEDIUM, HIGH (its own default is MEDIUM).
    # Empty = don't send it, so the model uses its default. Decided by `make bench`.
    thinking_level: Literal["", "LOW", "MEDIUM", "HIGH"] = ""

    # Metrics, tracing, benchmark, evaluation.
    metrics_path: Path = Path("data/metrics/requests.jsonl")
    tracing_exporter: Literal["none", "console", "gcp"] = "none"
    bench_runs: int = 2
    judge_model: str | None = None
    strong_orchestrator_model: str = "gemini-4-argon"
    eval_concurrency: int = 4
    eval_max_drop_points: float = 10.0  # percentage points (0-100) allowed drop vs baseline.json
    # USD per 1M tokens per model. A model missing here, or mapped to null, gets
    # cost_usd = null: we never guess a price. Override with MODEL_PRICES='{"model": {...}}'.
    model_prices: dict[str, ModelPrice | None] = Field(default_factory=lambda: dict(DEFAULT_MODEL_PRICES))

    @model_validator(mode="after")
    def _default_models_to_worker(self) -> "Settings":
        self.orchestrator_model = self.orchestrator_model or self.worker_model
        self.critic_model = self.critic_model or self.worker_model
        self.judge_model = self.judge_model or self.worker_model
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
