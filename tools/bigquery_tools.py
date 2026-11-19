"""BigQuery tools for the SQL agent: get_schema() and run_query().

Safety rules are enforced here in code, not only in the agent prompt:
  - exactly one read-only statement (SELECT / WITH ... SELECT / UNION of SELECTs)
  - only tables from the configured dataset
  - dry run first; reject queries that would scan too many bytes
  - only built-in functions (no qualified user-defined functions from other projects)
  - LIMIT added when missing and clamped when larger; returned rows capped
  - a per-session (= per request) cap on how many times run_query may be called
"""

import concurrent.futures
import datetime
import decimal
import logging
from functools import lru_cache
from itertools import islice
from typing import Any

import sqlglot
from google.adk.tools import ToolContext
from google.cloud import bigquery
from sqlglot import exp

from app.config import get_settings
from app.errors import current_request_id
from tools.upstream import (
    FATAL_ERROR_TYPES,
    fatal_error_type,
    fatal_message,
    record_fatal_error,
    sanitize_tool_error,
)

logger = logging.getLogger(__name__)

# Session state keys (read later for metrics).
STATE_ATTEMPTS = "sql_attempts"
STATE_LAST_SQL = "last_sql"
STATE_LAST_ROWS = "last_sql_rows"
STATE_HISTORY = "sql_history"  # every attempt: {sql, ok, error_type, rows}; evidence for the critic

# Statement types that write data or change the schema. Rejected anywhere in the tree.
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Export,
    exp.Command,  # anything sqlglot can't parse properly, e.g. CALL
)


class SqlRejected(Exception):
    """Raised by validate_sql when a query breaks a safety rule."""


# ---------------------------------------------------------------- validation


def validate_sql(sql: str, dataset: str, max_rows: int) -> str:
    """Check the SQL against the safety rules and return the SQL to execute.

    `dataset` is "project.dataset", e.g. "bigquery-public-data.thelook_ecommerce".
    Raises SqlRejected with a message the agent can act on.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SqlRejected(f"Could not parse SQL: {e}") from e

    if len(statements) != 1:
        raise SqlRejected(f"Exactly one statement is allowed, got {len(statements)}.")
    statement = statements[0]

    if not isinstance(statement, (exp.Select, exp.SetOperation)):
        raise SqlRejected("Only SELECT queries are allowed.")
    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlRejected(f"{type(node).__name__} is not allowed. Only SELECT queries are allowed.")

    _check_tables(statement, dataset)
    _check_functions(statement)

    # Add LIMIT when missing, and clamp a larger one, so BigQuery never builds a huge result.
    limit = statement.args.get("limit")
    limit_value = limit.expression if limit is not None else None
    if (
        limit is None
        or not isinstance(limit_value, exp.Literal)
        or not limit_value.is_int
        or int(limit_value.this) > max_rows
    ):
        statement = statement.limit(max_rows)
    return statement.sql(dialect="bigquery")


# Built-in BigQuery function namespaces (e.g. SAFE.DIVIDE, HLL_COUNT.MERGE).
BUILTIN_FUNCTION_NAMESPACES = {"SAFE", "NET", "HLL_COUNT", "KLL_QUANTILES"}


def _check_functions(statement: exp.Expression) -> None:
    """Reject qualified user-defined functions (e.g. `other-project.ds.udf`(x)) and AI.* functions.

    A persistent, remote, or AI.* function could read, send data, or invoke Vertex AI outside
    the allowed dataset and byte cap, so only built-in SQL functions may be called.
    """
    for func in statement.find_all(exp.Anonymous):
        name = func.name
        qualifier = None
        if "." in name:  # `project.dataset.fn`(...) written as one quoted identifier
            qualifier = name.rsplit(".", 1)[0]
        elif isinstance(func.parent, exp.Dot) and func.parent.expression is func:  # ds.fn(...)
            qualifier = func.parent.this.sql(dialect="bigquery").replace("`", "")
        if qualifier is not None and qualifier.upper() not in BUILTIN_FUNCTION_NAMESPACES:
            raise SqlRejected(
                f"Function `{qualifier}.{name.rsplit('.', 1)[-1]}` is not allowed. "
                "Only built-in BigQuery functions may be used."
            )

    # Also catch qualified functions that sqlglot parses as dedicated Func subclasses
    # (e.g. AI.GENERATE -> Dot(AI, AIGenerate), AI.IF -> Dot(AI, If)).
    for dot in statement.find_all(exp.Dot):
        if isinstance(dot.expression, exp.Func):
            qualifier = dot.this.sql(dialect="bigquery").replace("`", "")
            if qualifier.upper() not in BUILTIN_FUNCTION_NAMESPACES:
                fn_name = dot.expression.sql_name() if hasattr(dot.expression, "sql_name") else type(dot.expression).__name__
                raise SqlRejected(
                    f"Function `{qualifier}.{fn_name}` is not allowed. "
                    "Only built-in BigQuery functions may be used."
                )

    for node in statement.find_all(exp.AIGenerate):
        raise SqlRejected(
            f"Function `{node.sql_name()}` is not allowed. Only built-in BigQuery functions may be used."
        )


def _check_tables(statement: exp.Expression, dataset: str) -> None:
    """Every table must be <project>.<dataset>.<table> from the allowed dataset.

    sqlglot splits `a.b.c` and `a`.`b`.`c` into the same parts
    (catalog=a, db=b, name=c), so both backtick styles are handled the same way.
    """
    project, dataset_name = dataset.split(".", 1)
    cte_names = {cte.alias for cte in statement.find_all(exp.CTE)}

    for table in statement.find_all(exp.Table):
        if not table.catalog and not table.db and table.name in cte_names:
            continue  # reference to a WITH clause, not a real table
        if not table.catalog or not table.db:
            raise SqlRejected(
                f"Table '{table.name}' must be fully qualified, e.g. `{dataset}.{table.name}`."
            )
        if table.catalog != project or table.db != dataset_name:
            raise SqlRejected(
                f"Table `{table.catalog}.{table.db}.{table.name}` is not allowed. "
                f"Only tables from `{dataset}` may be queried."
            )


# ---------------------------------------------------------------- BigQuery client


@lru_cache
def get_client() -> bigquery.Client:
    """One BigQuery client, billed to our project, running jobs in BQ_LOCATION."""
    settings = get_settings()
    return bigquery.Client(project=settings.google_cloud_project, location=settings.bq_location)


def _to_json_safe(value: Any) -> Any:
    """BigQuery returns dates, decimals and bytes; tools must return JSON-friendly data."""
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------- tools

_schema_cache: dict[str, Any] = {}


def get_schema(tool_context: ToolContext) -> dict:
    """Returns every table in the dataset with its columns (name and type).

    Call this before writing SQL so you use real table and column names.
    """
    settings = get_settings()
    if settings.bq_dataset in _schema_cache:
        return _schema_cache[settings.bq_dataset]

    sql = (
        "SELECT table_name, column_name, data_type "
        f"FROM `{settings.bq_dataset}`.INFORMATION_SCHEMA.COLUMNS "
        "ORDER BY table_name, ordinal_position"
    )
    try:
        rows = get_client().query(sql).result(timeout=settings.bq_query_timeout_seconds)
    except Exception as e:  # never raise to the agent
        fatal = fatal_error_type(e)
        if fatal:
            record_fatal_error(tool_context.state, fatal, "get_schema")
            return {"ok": False, "error": fatal_message(fatal), "error_type": fatal}
        logger.warning("request_id=%s get_schema failed", current_request_id(), exc_info=e)
        return {"ok": False, "error": sanitize_tool_error(e), "error_type": "bigquery_error"}

    tables: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tables.setdefault(row["table_name"], []).append(
            {"name": row["column_name"], "type": row["data_type"]}
        )
    result = {"ok": True, "dataset": settings.bq_dataset, "tables": tables}
    _schema_cache[settings.bq_dataset] = result
    return result


def _failure(error: str, error_type: str, bytes_processed: int | None = None) -> dict:
    return {
        "ok": False,
        "rows": [],
        "row_count": 0,
        "error": error,
        "error_type": error_type,
        "bytes_processed": bytes_processed,
    }


def run_query(sql: str, tool_context: ToolContext) -> dict:
    """Runs one read-only BigQuery SELECT on the dataset and returns the rows.

    Use fully qualified table names in backticks, e.g.
    `bigquery-public-data.thelook_ecommerce.orders`.

    Returns a dict with: ok, rows, row_count, error, error_type, bytes_processed.
    If ok is false, read `error`, fix the SQL and try again.
    If error_type is "attempt_limit", stop and explain what went wrong.
    """
    settings = get_settings()
    state = tool_context.state

    # Attempt counter, scoped to the session. Each API request uses a fresh session,
    # so the cap is per request (including the orchestrator's revision turn).
    attempts = state.get(STATE_ATTEMPTS, 0)
    if attempts >= settings.sql_max_attempts:
        return _failure(
            f"Attempt limit reached ({settings.sql_max_attempts} queries). "
            "Do not call run_query again. Explain to the user what you tried and why it failed.",
            "attempt_limit",
        )
    state[STATE_ATTEMPTS] = attempts + 1

    result, executed_sql = _validate_and_execute(sql, settings)

    # Record the attempt (assign new values so ADK tracks the state change).
    # last_sql / last_sql_rows only move on success, so a failed follow-up query
    # never hides the rows the answer was built from. Failures stay in sql_history.
    if result["ok"]:
        state[STATE_LAST_SQL] = executed_sql
        state[STATE_LAST_ROWS] = result["rows"]
    elif result["error_type"] in FATAL_ERROR_TYPES:
        record_fatal_error(state, result["error_type"], "run_query")
    state[STATE_HISTORY] = state.get(STATE_HISTORY, []) + [
        {
            "sql": executed_sql,
            "ok": result["ok"],
            "error_type": result["error_type"],
            "rows": result["rows"],
        }
    ]
    return result


def _validate_and_execute(sql: str, settings) -> tuple[dict, str]:
    """Validate, dry-run, then run. Returns (result dict, the SQL that was checked/run)."""
    # 1. Validate.
    try:
        safe_sql = validate_sql(sql, settings.bq_dataset, settings.bq_max_rows)
    except SqlRejected as e:
        return _failure(str(e), "validation"), sql

    client = get_client()
    try:
        # 2. Dry run: estimate cost without running.
        dry_job = client.query(
            safe_sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        )
        estimated = dry_job.total_bytes_processed or 0
        if estimated > settings.bq_max_bytes_billed:
            return _failure(
                f"Query would scan {estimated:,} bytes, over the limit of "
                f"{settings.bq_max_bytes_billed:,}. Select fewer columns or filter more.",
                "too_expensive",
                estimated,
            ), safe_sql

        # 3. Real run, with a hard byte cap and a timeout.
        job = client.query(
            safe_sql,
            job_config=bigquery.QueryJobConfig(
                maximum_bytes_billed=settings.bq_max_bytes_billed,
                job_timeout_ms=settings.bq_query_timeout_seconds * 1000,
            ),
        )
        result = job.result(timeout=settings.bq_query_timeout_seconds)
        # Stop reading at BQ_MAX_ROWS even if the query's own LIMIT is larger.
        rows = [_to_json_safe(dict(row.items())) for row in islice(result, settings.bq_max_rows)]
    except concurrent.futures.TimeoutError:
        return _failure(
            f"Query timed out after {settings.bq_query_timeout_seconds}s. Simplify it.", "timeout"
        ), safe_sql
    except Exception as e:
        fatal = fatal_error_type(e)
        if fatal:  # auth / quota: retrying with different SQL cannot help
            return _failure(fatal_message(fatal), fatal), safe_sql
        # Other BigQuery errors go back to the agent (sanitized) so it can fix the SQL.
        logger.warning("request_id=%s run_query BigQuery error", current_request_id(), exc_info=e)
        return _failure(sanitize_tool_error(e), "bigquery_error"), safe_sql

    return {
        "ok": True,
        "rows": rows,
        "row_count": len(rows),
        "error": None,
        "error_type": None,
        "bytes_processed": job.total_bytes_processed,
    }, safe_sql
