"""Metrics across the whole request: orchestrator + sub-agents + critic (own Runner and session).

Real ADK agents, runners, plugin and sessions; the models are ScriptedLlm fakes that
report token usage, and BigQuery / search are faked. No GCP.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agents.critic import build_critic
from agents.orchestrator import build_orchestrator
from app.agent_service import AgentResult, AgentService
from app.config import ModelPrice
from app.errors import set_request_id
from app.main import app, get_orchestration, get_sql_service
from app.metrics import finish_request, start_request
from app.metrics_plugin import MetricsPlugin
from app.metrics_summary import read_records
from app.orchestration import AgentCritic, Orchestration
from tests.conftest import DATASET
from tests.scripted_llm import ScriptedLlm
from tools import rag_tools

CHUNK = {
    "chunk_id": "returns_policy.md:c4", "source": "returns_policy.md", "heading": "H",
    "text": "Underwear: maximum acceptable return rate is 10.0%.", "score": 0.03,
    "vector_rank": 1, "vector_distance": 0.1, "bm25_rank": 1, "bm25_score": 3.0,
}
VERDICT_OK = '{"supported": true, "unsupported_claims": []}'


def build_flow(prices: dict, orch_usage=(100, 10, 5), worker_usage=(200, 20, 7), critic_usage=(300, 30, 9)):
    """Orchestrator -> rag_agent (single-turn) -> retrieve, then the critic in its own Runner."""
    plugins = [MetricsPlugin(prices)]
    orchestrator = build_orchestrator(
        model=ScriptedLlm(
            model="orch-model", usage=orch_usage,
            script=[("rag_agent", {"request": "Underwear limit?"}), "Limit is 10.0% [returns_policy.md]."],
        ),
        worker_model=ScriptedLlm(
            model="worker-model", usage=worker_usage,
            script=[("retrieve", {"query": "Underwear limit"}), "10.0% [returns_policy.md]."],
        ),
        dataset=DATASET,
        sql_max_attempts=2,
    )
    critic = build_critic(ScriptedLlm(model="critic-model", usage=critic_usage, script=[VERDICT_OK]))
    return Orchestration(
        AgentService(orchestrator, plugins), AgentCritic(AgentService(critic, plugins)), max_revisions=1
    )


async def run_request(request_id: str, flow: Orchestration) -> dict:
    """What main.run_agent does, minus HTTP: set the id, start, run, finish."""
    set_request_id(request_id)
    start_request(request_id, "Underwear limit?", "/ask")
    result = await flow.ask("Underwear limit?")
    return finish_request(request_id, "ok", result.state, None)


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    monkeypatch.setattr(rag_tools, "hybrid_search", lambda query: [CHUNK])


def test_orchestrator_sub_agent_and_critic_calls_all_count_in_one_request():
    prices = {
        "orch-model": ModelPrice(input_per_1m=1.0, output_per_1m=2.0),
        "worker-model": ModelPrice(input_per_1m=1.0, output_per_1m=2.0),
        "critic-model": ModelPrice(input_per_1m=1.0, output_per_1m=2.0),
    }
    record = asyncio.run(run_request("req-all", build_flow(prices)))

    calls = record["llm_detail"]
    assert sorted(c["agent"] for c in calls) == ["critic", "orchestrator", "orchestrator", "rag_agent", "rag_agent"]
    assert record["llm_calls"] == 5
    # 2 orchestrator calls + 2 sub-agent calls + 1 critic call (critic has its own Runner and session)
    assert record["input_tokens"] == 2 * 100 + 2 * 200 + 300
    assert record["output_tokens"] == 2 * 10 + 2 * 20 + 30
    assert record["thinking_tokens"] == 2 * 5 + 2 * 7 + 9
    # cost = input * 1 + (output + thinking) * 2, per 1M tokens
    expected = (900 * 1.0 + (90 + 33) * 2.0) / 1e6
    assert record["cost_usd"] == pytest.approx(expected)

    for c in calls:
        assert c["latency_ms"] >= 0 and c["model"].endswith("-model")
    tools = {t["name"]: t for t in record["tool_detail"]}
    assert tools["retrieve"]["ok"] is True and tools["retrieve"]["is_agent"] is False
    assert tools["rag_agent"]["is_agent"] is True  # sub-agent wrapper, excluded from tool_time_ms
    assert record["agents_called"] == ["rag_agent"] and record["route"] == "docs"
    assert record["critic_supported"] is True


def test_cost_is_null_when_any_model_has_no_price():
    prices = {"orch-model": ModelPrice(input_per_1m=1.0, output_per_1m=2.0)}  # worker and critic unknown
    record = asyncio.run(run_request("req-null", build_flow(prices)))

    assert record["cost_usd"] is None
    by_agent = {c["agent"]: c["cost_usd"] for c in record["llm_detail"]}
    assert by_agent["orchestrator"] is not None
    assert by_agent["rag_agent"] is None and by_agent["critic"] is None


def test_concurrent_requests_are_kept_apart_by_the_request_id():
    async def both():
        # Each task gets its own copy of the context, so each sees only its own request id.
        return await asyncio.gather(
            run_request("req-A", build_flow({}, orch_usage=(1, 1, 1), worker_usage=(1, 1, 1), critic_usage=(1, 1, 1))),
            run_request("req-B", build_flow({}, orch_usage=(1000, 1, 1), worker_usage=(1000, 1, 1), critic_usage=(1000, 1, 1))),
        )

    a, b = asyncio.run(both())

    assert a["request_id"] == "req-A" and a["llm_calls"] == 5 and a["input_tokens"] == 5
    assert b["request_id"] == "req-B" and b["llm_calls"] == 5 and b["input_tokens"] == 5000


def test_calls_outside_a_request_are_not_recorded():
    flow = build_flow({})

    async def no_request():
        set_request_id("not-started")  # no start_request for this id
        await flow.ask("q")

    asyncio.run(no_request())  # must not raise
    assert finish_request("not-started", "ok", {}, None)["llm_calls"] == 0


# ---------------------------------------------------------------- through the API


class FakeFlow:
    def __init__(self, answer: str, state: dict | None = None, exc: Exception | None = None):
        self.answer, self.state, self.exc = answer, state or {}, exc

    async def ask(self, question: str) -> AgentResult:
        if self.exc:
            raise self.exc
        return AgentResult(answer=self.answer, state=self.state)


@pytest.fixture
def api(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(app.state, "metrics_path", path, raising=False)
    monkeypatch.setattr(app.state, "request_timeout_seconds", 5, raising=False)
    yield TestClient(app), path
    app.dependency_overrides.clear()


def test_every_request_appends_a_jsonl_line_ok_and_error(api):
    client, path = api
    app.dependency_overrides[get_orchestration] = lambda: FakeFlow(
        "ok [a.md]", {"route": "docs", "agents_called": ["rag_agent"], "revisions": 1,
                      "critic_verdict": {"supported": True, "unsupported_claims": []}}
    )
    ok = client.post("/ask", json={"question": "q1"})
    app.dependency_overrides[get_sql_service] = lambda: FakeFlow("", exc=TimeoutError())
    failed = client.post("/ask/sql", json={"question": "q2"})

    records = read_records(path)
    assert [r["status"] for r in records] == ["ok", "timeout"]
    assert records[0]["request_id"] == ok.headers["X-Request-ID"]
    assert records[0]["route"] == "docs" and records[0]["revisions"] == 1
    assert records[0]["critic_supported"] is True
    assert records[1]["request_id"] == failed.json()["request_id"]
    assert records[1]["route"] == "sql" and records[1]["endpoint"] == "/ask/sql"
    assert ok.json()["metrics"]["revisions"] == 1
