"""Shared test fakes: settings, a fake BigQuery client and a fake ToolContext."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from tools import bigquery_tools

DATASET = "bigquery-public-data.thelook_ecommerce"


@pytest.fixture
def settings(monkeypatch) -> Settings:
    """Test settings, independent of the real .env file."""
    test_settings = Settings(
        _env_file=None,
        google_cloud_project="test-project",
        google_cloud_location="global",
        worker_model="test-model",
        bq_dataset=DATASET,
        bq_max_bytes_billed=1_000,
        bq_max_rows=3,
        sql_max_attempts=2,
    )
    monkeypatch.setattr(bigquery_tools, "get_settings", lambda: test_settings)
    bigquery_tools._schema_cache.clear()
    return test_settings


class FakeJob:
    def __init__(self, rows: list[dict], total_bytes_processed: int, error: Exception | None):
        self._rows = rows
        self.total_bytes_processed = total_bytes_processed
        self._error = error

    def result(self, timeout=None):
        if self._error:
            raise self._error
        return iter(self._rows)


class FakeBigQueryClient:
    """Records every query and its job config. Rows are plain dicts."""

    def __init__(self, rows=None, dry_run_bytes: int = 100, error: Exception | None = None):
        self.rows = rows or []
        self.dry_run_bytes = dry_run_bytes
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def query(self, sql, job_config=None):
        self.calls.append((sql, job_config))
        if job_config is not None and job_config.dry_run:
            return FakeJob([], self.dry_run_bytes, None)
        return FakeJob(self.rows, self.dry_run_bytes, self.error)


@pytest.fixture
def fake_client(monkeypatch, settings) -> FakeBigQueryClient:
    client = FakeBigQueryClient()
    monkeypatch.setattr(bigquery_tools, "get_client", lambda: client)
    return client


def make_tool_context(invocation_id: str = "inv-1", state: dict | None = None):
    """Stands in for ADK's ToolContext: run_query only uses .state and .invocation_id."""
    return SimpleNamespace(state={} if state is None else state, invocation_id=invocation_id)
