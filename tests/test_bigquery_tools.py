"""run_query and get_schema with a fake BigQuery client. No GCP access."""

import concurrent.futures
import datetime
import decimal

from google.api_core.exceptions import BadRequest

from tests.conftest import make_tool_context
from tools.bigquery_tools import (
    STATE_ATTEMPTS,
    STATE_LAST_ROWS,
    STATE_LAST_SQL,
    get_schema,
    run_query,
)

GOOD_SQL = "SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`"
RESULT_KEYS = {"ok", "rows", "row_count", "error", "error_type", "bytes_processed"}


def test_success_returns_rows_and_records_state(fake_client):
    fake_client.rows = [{"order_id": 1}, {"order_id": 2}]
    ctx = make_tool_context()

    result = run_query(GOOD_SQL, ctx)

    assert set(result) == RESULT_KEYS
    assert result["ok"] is True
    assert result["rows"] == [{"order_id": 1}, {"order_id": 2}]
    assert result["row_count"] == 2
    assert ctx.state[STATE_ATTEMPTS] == 1
    assert ctx.state[STATE_LAST_SQL].endswith("LIMIT 3")  # LIMIT from settings.bq_max_rows
    assert ctx.state[STATE_LAST_ROWS] == result["rows"]


def test_dry_run_happens_first_then_real_run_with_byte_cap(fake_client, settings):
    run_query(GOOD_SQL, make_tool_context())

    (dry_sql, dry_cfg), (real_sql, real_cfg) = fake_client.calls
    assert dry_cfg.dry_run is True
    assert dry_sql == real_sql
    assert not real_cfg.dry_run
    assert real_cfg.maximum_bytes_billed == settings.bq_max_bytes_billed
    # The client stores int64 API fields as strings.
    assert int(real_cfg.job_timeout_ms) == settings.bq_query_timeout_seconds * 1000


def test_too_expensive_query_rejected_after_dry_run(fake_client, settings):
    fake_client.dry_run_bytes = settings.bq_max_bytes_billed + 1

    result = run_query(GOOD_SQL, make_tool_context())

    assert result["ok"] is False
    assert result["error_type"] == "too_expensive"
    assert len(fake_client.calls) == 1  # only the dry run; real query never ran


def test_row_cap_applied_even_with_large_limit(fake_client):
    fake_client.rows = [{"order_id": i} for i in range(10)]

    result = run_query(GOOD_SQL + " LIMIT 1000", make_tool_context())

    assert result["row_count"] == 3  # settings.bq_max_rows
    assert len(result["rows"]) == 3


def test_validation_failure_returned_not_raised(fake_client):
    result = run_query("DROP TABLE `bigquery-public-data.thelook_ecommerce.orders`", make_tool_context())

    assert set(result) == RESULT_KEYS
    assert result["ok"] is False
    assert result["error_type"] == "validation"
    assert fake_client.calls == []  # never reached BigQuery


def test_bigquery_error_returned_not_raised(fake_client):
    fake_client.error = BadRequest("Unrecognized name: ordr_id")

    result = run_query(GOOD_SQL, make_tool_context())

    assert result["ok"] is False
    assert result["error_type"] == "bigquery_error"
    assert "ordr_id" in result["error"]


def test_timeout_returned_not_raised(fake_client):
    fake_client.error = concurrent.futures.TimeoutError()

    result = run_query(GOOD_SQL, make_tool_context())

    assert result["error_type"] == "timeout"


def test_attempt_cap_stops_further_queries(fake_client):
    ctx = make_tool_context()  # settings.sql_max_attempts == 2

    run_query(GOOD_SQL, ctx)
    run_query("DROP TABLE x", ctx)  # rejected queries still count as attempts
    calls_before = len(fake_client.calls)
    third = run_query(GOOD_SQL, ctx)

    assert third["ok"] is False
    assert third["error_type"] == "attempt_limit"
    assert ctx.state[STATE_ATTEMPTS] == 2
    assert len(fake_client.calls) == calls_before  # nothing sent to BigQuery


def test_attempt_counter_is_session_scoped_across_invocations(fake_client):
    # Same session, two invocations (e.g. the orchestrator's draft and revision turns):
    # the cap covers the whole request, it does not reset per invocation.
    state: dict = {}
    run_query(GOOD_SQL, make_tool_context("draft-turn", state))
    run_query(GOOD_SQL, make_tool_context("draft-turn", state))

    result = run_query(GOOD_SQL, make_tool_context("revision-turn", state))

    assert result["error_type"] == "attempt_limit"
    assert state[STATE_ATTEMPTS] == 2


def test_rows_made_json_safe(fake_client):
    fake_client.rows = [
        {"created_at": datetime.datetime(2024, 1, 2, 3, 4, 5), "price": decimal.Decimal("9.50")}
    ]

    result = run_query(GOOD_SQL, make_tool_context())

    assert result["rows"] == [{"created_at": "2024-01-02T03:04:05", "price": 9.5}]


def test_get_schema_groups_columns_and_caches(fake_client):
    fake_client.rows = [
        {"table_name": "orders", "column_name": "order_id", "data_type": "INT64"},
        {"table_name": "orders", "column_name": "status", "data_type": "STRING"},
        {"table_name": "users", "column_name": "id", "data_type": "INT64"},
    ]

    first = get_schema(make_tool_context())
    second = get_schema(make_tool_context())

    assert first["ok"] is True
    assert first["tables"] == {
        "orders": [{"name": "order_id", "type": "INT64"}, {"name": "status", "type": "STRING"}],
        "users": [{"name": "id", "type": "INT64"}],
    }
    assert second == first
    assert len(fake_client.calls) == 1  # second call served from cache
    assert "INFORMATION_SCHEMA.COLUMNS" in fake_client.calls[0][0]


def test_get_schema_error_returned_and_not_cached(fake_client):
    fake_client.error = BadRequest("boom")
    assert get_schema(make_tool_context())["ok"] is False

    fake_client.error = None
    fake_client.rows = [{"table_name": "t", "column_name": "c", "data_type": "INT64"}]
    assert get_schema(make_tool_context())["ok"] is True


def test_every_attempt_is_recorded_in_sql_history(fake_client):
    fake_client.rows = [{"order_id": 1}]
    ctx = make_tool_context()

    run_query("DROP TABLE x", ctx)
    run_query(GOOD_SQL, ctx)

    history = ctx.state["sql_history"]
    assert [h["ok"] for h in history] == [False, True]
    assert history[0]["error_type"] == "validation" and history[0]["sql"] == "DROP TABLE x"
    assert history[1]["sql"].endswith("LIMIT 3") and history[1]["rows"] == [{"order_id": 1}]


def test_failed_follow_up_query_keeps_last_successful_sql_and_rows(fake_client):
    fake_client.rows = [{"order_id": 1}]
    ctx = make_tool_context()

    run_query(GOOD_SQL, ctx)
    run_query("DROP TABLE x", ctx)  # second attempt fails

    assert ctx.state[STATE_LAST_SQL].startswith("SELECT order_id")
    assert ctx.state[STATE_LAST_ROWS] == [{"order_id": 1}]
    assert [h["ok"] for h in ctx.state["sql_history"]] == [True, False]


def test_no_last_sql_when_nothing_succeeded(fake_client):
    ctx = make_tool_context()
    run_query("DROP TABLE x", ctx)
    assert STATE_LAST_SQL not in ctx.state


def test_auth_error_is_fatal_and_recorded(fake_client):
    from google.api_core.exceptions import Forbidden

    fake_client.error = Forbidden("Access Denied: user does not have bigquery.jobs.create")
    ctx = make_tool_context()

    result = run_query(GOOD_SQL, ctx)

    assert result["ok"] is False
    assert result["error_type"] == "auth_error"
    assert "Do not retry" in result["error"]
    assert ctx.state["upstream_error"] == {"error_type": "auth_error", "tool": "run_query"}


def test_quota_error_in_get_schema_is_fatal_and_recorded(fake_client):
    from google.api_core.exceptions import TooManyRequests

    fake_client.error = TooManyRequests("Quota exceeded")
    ctx = make_tool_context()

    result = get_schema(ctx)

    assert result["error_type"] == "quota_exceeded"
    assert ctx.state["upstream_error"]["tool"] == "get_schema"


def test_ordinary_bigquery_error_is_not_fatal(fake_client):
    fake_client.error = BadRequest("Unrecognized name: ordr_id")
    ctx = make_tool_context()

    result = run_query(GOOD_SQL, ctx)

    assert result["error_type"] == "bigquery_error"
    assert "ordr_id" in result["error"]  # the agent needs the real message to fix its SQL
    assert "upstream_error" not in ctx.state
