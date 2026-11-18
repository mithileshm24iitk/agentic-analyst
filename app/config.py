"""App settings, loaded from environment variables and the .env file."""

import os
from functools import lru_cache

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
