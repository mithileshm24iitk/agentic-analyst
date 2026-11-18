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
