"""Structured metrics line on stdout (Cloud Logging) and the /metrics/summary scope note."""

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.agent_service import AgentResult
from app.main import SUMMARY_SCOPE, app, get_orchestration
from app.metrics import LOG_TYPE, log_record

client = TestClient(app)


def record(**overrides):
    base = {"request_id": "abc", "endpoint": "/ask", "status": "ok", "route": "sql", "latency_ms": 12.5}
    return {**base, **overrides}


def test_log_record_writes_one_json_line_with_cloud_logging_fields():
    out = io.StringIO()

    log_record(record(), stream=out)

    lines = out.getvalue().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["severity"] == "INFO"
    assert entry["log_type"] == LOG_TYPE
    assert entry["message"].startswith(LOG_TYPE) and "route=sql" in entry["message"]
    assert entry["request_id"] == "abc" and entry["latency_ms"] == 12.5  # the full record is there


def test_failed_request_is_a_warning():
    out = io.StringIO()
    log_record(record(status="timeout"), stream=out)
    assert json.loads(out.getvalue())["severity"] == "WARNING"


def test_log_record_never_raises():
    class Broken:
        def write(self, _):
            raise OSError("stdout closed")

    log_record(record(), stream=Broken())  # no exception


@pytest.fixture
def fake_app(monkeypatch, tmp_path):
    class Fake:
        async def ask(self, question):
            return AgentResult(answer="ok", state={"route": "docs"})

    app.state.request_timeout_seconds = 5
    app.state.metrics_path = tmp_path / "requests.jsonl"
    app.dependency_overrides[get_orchestration] = Fake
    yield tmp_path / "requests.jsonl"
    app.dependency_overrides.clear()


def test_each_request_goes_to_file_and_stdout(fake_app, capsys):
    response = client.post("/ask", json={"question": "Return policy?"})
    request_id = response.headers["X-Request-ID"]

    stdout_entries = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    ours = [e for e in stdout_entries if e.get("request_id") == request_id]
    assert len(ours) == 1 and ours[0]["log_type"] == LOG_TYPE and ours[0]["route"] == "docs"

    file_lines = fake_app.read_text().splitlines()
    assert len(file_lines) == 1 and json.loads(file_lines[0])["request_id"] == request_id


def test_summary_states_its_scope(fake_app):
    body = client.get("/metrics/summary").json()
    assert body["scope"] == SUMMARY_SCOPE
    assert "current instance" in body["scope"] and "Cloud Logging" in body["scope"]
    assert "overall" in body and "by_route" in body


def test_cancelled_request_cleans_up_active_metrics(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from app import metrics
    from app.main import run_agent

    class CancellingService:
        async def ask(self, question):
            raise asyncio.CancelledError()

    req = SimpleNamespace(
        state=SimpleNamespace(request_id="cancel-test-id"),
        app=SimpleNamespace(state=SimpleNamespace(max_question_chars=2000, request_timeout_seconds=5, metrics_path=tmp_path / "m.jsonl")),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_agent(req, CancellingService(), "hello", "/ask"))

    assert "cancel-test-id" not in metrics._active
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "cancelled"
