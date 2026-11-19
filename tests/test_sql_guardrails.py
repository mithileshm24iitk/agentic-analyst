"""validate_sql guardrails. Pure logic, no BigQuery."""

import pytest

from tests.conftest import DATASET
from tools.bigquery_tools import SqlRejected, validate_sql


def check(sql: str, max_rows: int = 100) -> str:
    return validate_sql(sql, DATASET, max_rows)


# --- statement type ------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE `bigquery-public-data.thelook_ecommerce.orders`",
        "INSERT INTO `bigquery-public-data.thelook_ecommerce.orders` (order_id) VALUES (1)",
        "UPDATE `bigquery-public-data.thelook_ecommerce.orders` SET status = 'x' WHERE true",
        "DELETE FROM `bigquery-public-data.thelook_ecommerce.orders` WHERE true",
        "MERGE `bigquery-public-data.thelook_ecommerce.orders` t USING "
        "`bigquery-public-data.thelook_ecommerce.users` s ON t.user_id = s.id WHEN MATCHED THEN DELETE",
        "CREATE TABLE `bigquery-public-data.thelook_ecommerce.x` AS SELECT 1 AS a",
        "ALTER TABLE `bigquery-public-data.thelook_ecommerce.orders` ADD COLUMN x INT64",
        "TRUNCATE TABLE `bigquery-public-data.thelook_ecommerce.orders`",
        "EXPORT DATA OPTIONS(uri='gs://bucket/*.csv', format='CSV') AS "
        "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders`",
        "CALL `bigquery-public-data.thelook_ecommerce.some_proc`()",
    ],
)
def test_non_select_statements_rejected(sql):
    with pytest.raises(SqlRejected):
        check(sql)


def test_drop_table_rejected_with_message():
    with pytest.raises(SqlRejected, match="Only SELECT"):
        check("DROP TABLE `bigquery-public-data.thelook_ecommerce.orders`")


def test_two_statements_rejected():
    sql = (
        "SELECT 1 FROM `bigquery-public-data.thelook_ecommerce.orders`; "
        "DROP TABLE `bigquery-public-data.thelook_ecommerce.orders`"
    )
    with pytest.raises(SqlRejected, match="Exactly one statement"):
        check(sql)


def test_two_selects_rejected():
    with pytest.raises(SqlRejected, match="Exactly one statement"):
        check("SELECT 1; SELECT 2")


def test_trailing_semicolon_allowed():
    check("SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`;")


def test_unparseable_sql_rejected():
    with pytest.raises(SqlRejected):
        check("SELEC order_id FRM orders WHERE (")


# --- table allowlist and backtick forms ------------------------------------


@pytest.mark.parametrize(
    "table_ref",
    [
        "`bigquery-public-data.thelook_ecommerce.orders`",  # one quoted identifier
        "`bigquery-public-data`.`thelook_ecommerce`.`orders`",  # each part quoted
        "`bigquery-public-data`.thelook_ecommerce.orders",  # only the hyphenated part quoted
    ],
)
def test_allowed_dataset_accepted_in_every_backtick_form(table_ref):
    out = check(f"SELECT order_id FROM {table_ref} LIMIT 5")
    assert "thelook_ecommerce" in out


@pytest.mark.parametrize(
    "table_ref",
    [
        "`other-project.other_dataset.orders`",
        "`other-project`.`other_dataset`.`orders`",
        "`bigquery-public-data.other_dataset.orders`",  # same project, wrong dataset
        "`bigquery-public-data`.`other_dataset`.`orders`",
        "`my-project.thelook_ecommerce.orders`",  # right dataset name, wrong project
    ],
)
def test_other_dataset_rejected_in_every_backtick_form(table_ref):
    with pytest.raises(SqlRejected, match="not allowed"):
        check(f"SELECT * FROM {table_ref}")


def test_other_dataset_hidden_in_join_rejected():
    sql = (
        "SELECT o.order_id FROM `bigquery-public-data.thelook_ecommerce.orders` o "
        "JOIN `other-project.secret.users` u ON o.user_id = u.id"
    )
    with pytest.raises(SqlRejected, match="not allowed"):
        check(sql)


def test_other_dataset_hidden_in_subquery_rejected():
    sql = (
        "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders` "
        "WHERE user_id IN (SELECT id FROM `other-project.secret.users`)"
    )
    with pytest.raises(SqlRejected, match="not allowed"):
        check(sql)


def test_unqualified_table_rejected():
    with pytest.raises(SqlRejected, match="fully qualified"):
        check("SELECT * FROM orders")


def test_cte_names_allowed():
    sql = (
        "WITH recent AS (SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`) "
        "SELECT * FROM recent"
    )
    assert "recent" in check(sql)


def test_union_of_allowed_tables_accepted():
    check(
        "SELECT id FROM `bigquery-public-data.thelook_ecommerce.users` "
        "UNION ALL SELECT user_id FROM `bigquery-public-data.thelook_ecommerce.orders`"
    )


# --- LIMIT ------------------------------------------------------------------


def test_limit_added_when_missing():
    out = check("SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`", max_rows=100)
    assert out.endswith("LIMIT 100")


def test_limit_added_to_cte_query():
    out = check(
        "WITH x AS (SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`) "
        "SELECT * FROM x",
        max_rows=50,
    )
    assert out.endswith("LIMIT 50")


def test_existing_limit_kept():
    out = check("SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 5")
    assert out.endswith("LIMIT 5")
    assert out.count("LIMIT") == 1


# --- functions: only built-ins ------------------------------------------------

USERS = "`bigquery-public-data.thelook_ecommerce.users`"


@pytest.mark.parametrize(
    "call",
    [
        "`other-project.dataset.remote_udf`(email)",  # one quoted identifier
        "`other-project`.dataset.remote_udf(email)",  # split, project quoted
        "other_dataset.my_udf(email)",  # dataset-qualified
        "`bigquery-public-data.thelook_ecommerce.some_udf`(email)",  # even in the allowed dataset
        "AI.GENERATE('prompt')",  # BigQuery AI.* invokes Vertex AI outside byte cap
        "AI.GENERATE('prompt', connection_id => 'other-proj.us.conn')",
        "AI.IF(TRUE, 1, 2)",
    ],
)
def test_qualified_user_defined_functions_rejected(call):
    with pytest.raises(SqlRejected, match="Only built-in BigQuery functions"):
        check(f"SELECT {call} FROM {USERS}")


def test_table_function_from_other_project_rejected():
    with pytest.raises(SqlRejected):
        check("SELECT * FROM `other-project.ds.tvf`(1)")


def test_builtin_functions_and_namespaces_allowed():
    out = check(
        f"SELECT SAFE_DIVIDE(1, 2), SAFE.PARSE_DATE('%Y', '2024'), NET.HOST('http://x.com'), "
        f"HLL_COUNT.EXTRACT(HLL_COUNT.INIT(id)), APPROX_TOP_COUNT(id, 5), COUNT(*) FROM {USERS}"
    )
    assert out.startswith("SELECT")


def test_region_information_schema_rejected():
    with pytest.raises(SqlRejected):
        check("SELECT query FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT")


# --- LIMIT clamping -----------------------------------------------------------


def test_larger_limit_clamped_to_max_rows():
    out = check(f"SELECT id FROM {USERS} LIMIT 10000000", max_rows=100)
    assert out.endswith("LIMIT 100") and "10000000" not in out


def test_limit_equal_to_max_rows_kept():
    assert check(f"SELECT id FROM {USERS} LIMIT 100", max_rows=100).endswith("LIMIT 100")


def test_union_outer_limit_clamped_and_inner_limit_valid():
    out = check(
        f"SELECT id FROM {USERS} UNION ALL (SELECT id FROM {USERS} LIMIT 5) LIMIT 999", max_rows=50
    )
    assert out.endswith("LIMIT 50")
    assert "LIMIT 5)" in out  # inner LIMIT untouched, one valid outer LIMIT
    assert out.count("LIMIT") == 2


def test_non_literal_limit_replaced():
    out = check(f"SELECT id FROM {USERS} LIMIT 10 * 1000", max_rows=100)
    assert out.endswith("LIMIT 100")
