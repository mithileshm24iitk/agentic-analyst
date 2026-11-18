"""Real BigQuery test. Skipped unless RUN_BQ_INTEGRATION=1 (needs .env and gcloud ADC).

Cost: one INFORMATION_SCHEMA query plus one small COUNT(*), a few MB of scan at most.
"""

import os

import pytest

from tests.conftest import make_tool_context
from tools import bigquery_tools

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BQ_INTEGRATION") != "1",
    reason="set RUN_BQ_INTEGRATION=1 to run against real BigQuery",
)


def test_schema_and_query_against_real_bigquery():
    schema = bigquery_tools.get_schema(make_tool_context())
    assert schema["ok"] is True, schema
    assert "orders" in schema["tables"]

    result = bigquery_tools.run_query(
        "SELECT status, COUNT(*) AS n "
        "FROM `bigquery-public-data`.`thelook_ecommerce`.`orders` GROUP BY status",
        make_tool_context(),
    )
    assert result["ok"] is True, result
    assert result["row_count"] > 0
    assert result["bytes_processed"] is not None
