"""Percentiles, JSONL reading/writing and GET /metrics/summary on a fixture file."""

import json

import pytest
from fastapi.testclient import TestClient

from app.metrics import append_record
from app.metrics_summary import build_summary, percentile, read_records
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------- percentiles


@pytest.mark.parametrize(
    "values, p, expected",
    [
        ([], 50, None),
        ([7], 95, 7),
        ([1, 2, 3, 4], 50, 2.5),  # linear interpolation between ranks
        ([1, 2, 3, 4, 5], 50, 3),
        ([10, 20, 30, 40, 50], 95, 48),  # position 3.8 -> 40 + 0.8 * 10
        ([50, 10, 40, 20, 30], 95, 48),  # order does not matter
        ([1, 2, 3, 4, 5], 0, 1),
        ([1, 2, 3, 4, 5], 100, 5),
    ],
)
def test_percentile(values, p, expected):
    assert percentile(values, p) == (pytest.approx(expected) if expected is not None else None)


def test_percentile_matches_numpy_linear_method():
    values = [120.0, 3.5, 88.0, 41.0, 41.0, 9.0, 300.0]
    # numpy.percentile(values, [50, 95]) with the default 'linear' method:
    assert percentile(values, 50) == pytest.approx(41.0)
    assert percentile(values, 95) == pytest.approx(246.0)


# ---------------------------------------------------------------- JSONL


def record(route="docs", status="ok", latency=1000.0, cost=0.001, revisions=0, sql_attempts=0):
    return {
        "request_id": "r", "route": route, "status": status, "latency_ms": latency,
        "llm_calls": 4, "input_tokens": 1000, "output_tokens": 100, "thinking_tokens": 50,
        "cost_usd": cost, "sql_attempts": sql_attempts, "revisions": revisions,
    }


def test_append_record_writes_one_json_line_each(tmp_path):
    path = tmp_path / "metrics" / "requests.jsonl"  # folder does not exist yet

    append_record(path, record(route="docs"))
    append_record(path, record(route="sql"))

    lines = path.read_text().splitlines()
    assert [json.loads(line)["route"] for line in lines] == ["docs", "sql"]


def test_read_records_handles_missing_empty_and_broken_lines(tmp_path):
    assert read_records(tmp_path / "missing.jsonl") == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert read_records(empty) == []
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(record()) + "\n{not json\n\n" + json.dumps(record(route="sql")) + "\n")
    assert [r["route"] for r in read_records(mixed)] == ["docs", "sql"]


# ---------------------------------------------------------------- summary


@pytest.fixture
def fixture_file(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [
        record(route="docs", latency=1000, cost=0.001),
        record(route="docs", latency=3000, cost=0.003, revisions=1),
        record(route="sql", latency=2000, cost=None, sql_attempts=2),  # unknown price
        record(route="sql", status="timeout", latency=9000, cost=0.002, sql_attempts=3),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_summary_overall_and_by_route(fixture_file):
    summary = build_summary(fixture_file)
    overall, docs, sql = summary["overall"], summary["by_route"]["docs"], summary["by_route"]["sql"]

    assert overall["requests"] == 4
    assert overall["error_rate"] == 0.25
    assert overall["latency_ms"] == {"avg": 3750.0, "p50": 2500.0, "p95": 8100.0}
    assert overall["avg_cost_usd"] == pytest.approx(0.002)  # mean of the 3 known costs
    assert overall["requests_with_unknown_cost"] == 1
    assert overall["revision_rate"] == 0.25
    assert overall["avg_llm_calls"] == 4

    assert docs["requests"] == 2 and docs["error_rate"] == 0 and docs["revision_rate"] == 0.5
    assert sql["requests"] == 2 and sql["error_rate"] == 0.5 and sql["avg_sql_attempts"] == 2.5
    assert sorted(summary["by_route"]) == ["docs", "sql"]


def test_summary_of_missing_file_has_zero_requests(tmp_path):
    summary = build_summary(tmp_path / "nothing.jsonl")
    assert summary["overall"]["requests"] == 0
    assert summary["overall"]["latency_ms"] == {"avg": None, "p50": None, "p95": None}
    assert summary["by_route"] == {}


def test_summary_endpoint_reads_metrics_path(fixture_file, monkeypatch):
    monkeypatch.setattr(app.state, "metrics_path", fixture_file, raising=False)
    body = client.get("/metrics/summary").json()
    assert body["overall"]["requests"] == 4
    assert set(body["by_route"]) == {"docs", "sql"}


def test_summary_endpoint_with_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(app.state, "metrics_path", tmp_path / "none.jsonl", raising=False)
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    assert response.json()["overall"]["requests"] == 0
